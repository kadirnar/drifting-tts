"""Low-latency synthesis: CUDA graphs over length buckets, and a streaming vocoder.

* **Acoustic model** (:class:`GraphedAcoustic`): the text encoder (with pitch) and the DiT run as CUDA graphs
  captured once per length bucket. Token ids are padded to a multiple of ``token_bucket`` and frames to a multiple of
  ``frame_bucket``, and the padding is masked, so the output equals the eager model's (bit for bit in our tests).
  Two options trade that exactness for speed: ``tf32`` (TF32 matmuls; mel SNR 74 dB against fp32) and ``compile``
  (``torch.compile`` of the DiT; one dynamic-shape compilation, about 20 s on first use).
* **Vocoder** (:func:`stream_vocoder`): the first ``first`` frames are vocoded with ``context`` frames of right
  context, then windows of ``chunk`` frames with ``context`` frames on each side. BigVGAN-v2's receptive field is
  about 24 frames, so with 32 frames of context the chunks match vocoding the whole utterance up to the TF32 noise
  floor (> 95 dB SNR in fp32). Each vocoder of the registry has its own measured context (``Vocoder.context``,
  docs/VOCODERS.md).
  Fixed-size windows run as CUDA graphs. Time to first audio no longer grows with the sentence.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor

from .alignment import sequence_mask
from .models.text_encoder import durations_to_alignment


@contextlib.contextmanager
def tf32_matmul(enabled: bool = True):
    """TF32 for matrix multiplies (PyTorch already uses it for convolutions by default)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class Graph:
    """A CUDA graph of ``fn`` over static input buffers: ``__call__`` copies the inputs in and replays it."""

    def __init__(self, fn: Callable, inputs: list[Tensor], pool=None, tf32: bool = False, warmup: int = 3):
        self.inputs = inputs
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), tf32_matmul(tf32):
            for _ in range(warmup):
                fn(*inputs)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool), tf32_matmul(tf32):
            self.outputs = fn(*inputs)

    def __call__(self, *args: Tensor):
        for buf, x in zip(self.inputs, args):
            buf.copy_(x)
        self.graph.replay()
        return self.outputs


def _round_up(n: int, k: int) -> int:
    return -(-n // k) * k


class GraphedAcoustic:
    """:meth:`DriftingTTS.synthesize` (one step, one utterance) as bucketed CUDA graphs, with the same random draws."""

    def __init__(self, model, token_bucket: int = 32, frame_bucket: int = 64, compile: bool = False,
                 tf32: bool = False):
        self.model, self.token_bucket, self.frame_bucket, self.tf32 = model, token_bucket, frame_bucket, tf32
        self.device = next(model.parameters()).device
        self.pool = torch.cuda.graph_pool_handle()
        self.encoders: dict[int, Graph] = {}
        self.generators: dict[int, Graph] = {}
        self.generate = model.generate
        if compile:
            logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)  # harmless dynamic-shape noise
            self.generate = torch.compile(model.generate, dynamic=True)

    def _encoder(self, n: int) -> Graph:
        nb = _round_up(n, self.token_bucket)
        if nb not in self.encoders:
            m, dev = self.model, self.device

            def encode(ids: Tensor, length: Tensor, spk: Tensor):
                h, mu, logw, x_mask = m.encoder(ids, length, spk)
                if m.pitch_enabled:
                    h, _ = m.pitch_condition(h, x_mask, spk)
                return h, mu, logw, x_mask

            bufs = [torch.zeros(1, nb, dtype=torch.long, device=dev), torch.full((1,), nb, device=dev),
                    torch.zeros(1, dtype=torch.long, device=dev)]
            self.encoders[nb] = Graph(encode, bufs, self.pool, self.tf32)
        return self.encoders[nb]

    def _generator(self, t: int) -> Graph:
        tb = _round_up(t, self.frame_bucket)
        if tb not in self.generators:
            m, dev, g = self.model, self.device, self.model.generator

            def generate(z: Tensor, cond: Tensor, mask: Tensor, spk: Tensor, alpha: Tensor, labels: Tensor) -> Tensor:
                return self.generate(z, cond, spk, alpha, mask=mask, noise_labels=labels)

            bufs = [torch.zeros(1, m.n_mels, tb, device=dev), torch.zeros(1, m.n_mels + m.encoder.d, tb, device=dev),
                    torch.ones(1, tb, dtype=torch.bool, device=dev), torch.zeros(1, dtype=torch.long, device=dev),
                    torch.ones(1, device=dev), torch.zeros(1, max(1, g.noise_coords), dtype=torch.long, device=dev)]
            self.generators[tb] = Graph(generate, bufs, self.pool, self.tf32)
        return self.generators[tb]

    def warmup(self, max_tokens: int = 384, max_frames: int = 1664) -> None:
        """Capture every bucket up to these lengths now rather than on first use."""
        for n in range(self.token_bucket, max_tokens + 1, self.token_bucket):
            self._encoder(n)
        for t in range(self.frame_bucket, max_frames + 1, self.frame_bucket):
            self._generator(t)
        torch.cuda.synchronize()

    @torch.no_grad()
    def __call__(self, ids: Tensor, spk: Tensor, cfg_scale: float, temperature: float, length_scale: float,
                 generator: torch.Generator | None = None) -> Tensor:
        """Token ids ``[1, N]`` -> normalised mel ``[1, n_mels, T]``, as ``synthesize`` with ``steps=1``."""
        n = ids.shape[1]
        enc = self._encoder(n)
        padded = torch.zeros(1, enc.inputs[0].shape[1], dtype=torch.long, device=self.device)
        padded[:, :n] = ids
        h, mu, logw, x_mask = enc(padded, torch.tensor([n], device=self.device), spk)
        attn, y_len = durations_to_alignment(logw, x_mask, length_scale)  # reads T on the host
        cond = self.model.frame_condition(h, mu, attn)
        t = cond.shape[-1]
        # the same draws, in the same order, as DriftingTTS.synthesize / rollout
        z = torch.randn(1, self.model.n_mels, t, device=self.device, generator=generator)
        net = self.model.generator
        labels = torch.randint(0, net.noise_classes, (1, max(1, net.noise_coords)), device=self.device,
                               generator=generator)
        gen = self._generator(t)
        tb = gen.inputs[0].shape[-1]
        z_pad = torch.zeros(1, self.model.n_mels, tb, device=self.device)
        z_pad[..., :t] = z * temperature
        cond_pad = torch.zeros(1, cond.shape[1], tb, device=self.device)
        cond_pad[..., :t] = cond
        mask = sequence_mask(y_len, tb)
        alpha = torch.full((1,), float(cfg_scale), device=self.device)
        return gen(z_pad, cond_pad, mask, spk, alpha, labels)[..., :t].clone()


def stream_vocoder(vocode: Callable[[Tensor], Tensor], mel: Tensor, hop: int = 256, first: int = 32,
                   chunk: int = 256, context: int | None = 32, graphs: dict | None = None) -> Iterator[Tensor]:
    """Vocode ``mel`` ``[1, n_mels, T]`` window by window; yields the waveform in pieces (``[samples]``).

    ``context``: ``None`` vocodes the whole mel in one piece (a vocoder that cannot stream, e.g. Griffin-Lim).
    ``graphs``: a dict that caches CUDA graphs of ``vocode`` for the two fixed window sizes (``None``: eager)."""
    t = mel.shape[-1]
    if context is None or t <= first + context:
        yield vocode(mel)[0]
        return

    def run(a: int, b: int) -> Tensor:
        x = mel[..., a:b]
        if graphs is None or b - a not in (first + context, chunk + 2 * context):
            return vocode(x)[0]
        if b - a not in graphs:
            graphs[b - a] = Graph(vocode, [torch.zeros_like(x)], pool=graphs.get("pool"))
        return graphs[b - a](x)[0]

    yield run(0, first + context)[: first * hop].clone()
    for s in range(first, t, chunk):
        a, b = max(0, s - context), min(t, s + chunk + context)
        yield run(a, b)[(s - a) * hop: (s - a + min(chunk, t - s)) * hop].clone()
