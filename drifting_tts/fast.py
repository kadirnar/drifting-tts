"""Low-latency synthesis: CUDA graphs over length buckets, and a streaming vocoder.

* **Acoustic model** (:class:`GraphedAcoustic`): the text encoder (with pitch) and the DiT run as CUDA graphs
  captured once per length bucket. Token ids are padded to a multiple of ``token_bucket`` and frames to a multiple of
  ``frame_bucket``, and the padding is masked, so the output equals the eager model's (bit for bit in our tests).
  A one-pass stochastic prosody predictor (:class:`~drifting_tts.models.prosody_net.ProsodyPredictor`, ``drift`` or
  ``mse``) runs inside the encoder's graph, its noise drawn outside it in the eager order (:func:`graphable`).
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
from .models.prosody_net import boundary_tokens
from .models.text_encoder import durations_to_alignment, frames_to_alignment


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


def graphable(prosody, spread: float = 1.0) -> bool:
    """Whether :class:`GraphedAcoustic` can run a prosody predictor: one network pass (``drift`` / ``mse``), no
    word features (BERT runs on the host), no output-space spread (a batch of samples)."""
    return prosody is None or (prosody.kind in ("drift", "mse") and not prosody.net.word_dim and spread == 1.0)


class GraphedAcoustic:
    """:meth:`DriftingTTS.synthesize` (one step, one utterance) as bucketed CUDA graphs, with the same random draws.

    ``prosody``: a stochastic prosody predictor (:func:`graphable`) whose durations (``prosody_durations="sampled"``)
    and token pitch replace the regressors', as ``ProsodyPredictor.predict`` followed by ``synthesize``.
    ``duration_row``: the sampled durations get their own row in the predictor's batch (a duration temperature of
    their own, another speaker's rhythm: ``ProsodyPredictor.sample``); every call then runs both rows."""

    def __init__(self, model, token_bucket: int = 32, frame_bucket: int = 64, compile: bool = False,
                 tf32: bool = False, prosody=None, prosody_durations: str = "sampled", duration_row: bool = False):
        if not graphable(prosody):
            raise ValueError("this prosody predictor cannot run in a CUDA graph (flow matching, word features)")
        self.prosody, self.prosody_durations = prosody, prosody_durations
        self.duration_row = bool(duration_row and prosody is not None and prosody_durations == "sampled")
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
            m, dev, pros = self.model, self.device, self.prosody

            def encode(ids: Tensor, length: Tensor, spk: Tensor):
                h, mu, logw, x_mask = m.encoder(ids, length, spk)
                if m.pitch_enabled:
                    h, _ = m.pitch_condition(h, x_mask, spk)
                return h, mu, logw, x_mask

            def encode_prosody(ids: Tensor, length: Tensor, spk: Tensor, z_tok: Tensor, z_glob: Tensor,
                               scale: Tensor):
                """ProsodyPredictor.predict (noise given), then the encoder half of synthesize."""
                h, mu, logw, x_mask = m.encoder(ids, length, spk)
                cond, base = pros.condition(m, h, x_mask, spk, logw, pros.stats)
                out = pros.net(cond, x_mask, z_tok, z_glob) if pros.kind == "drift" else pros.net(cond, x_mask)
                frames, pitch = pros.frames_and_pitch((base + out[:, :2]) * x_mask, out[:, 2], x_mask, scale)
                h, _ = m.pitch_condition(h, x_mask, spk, pitch)
                return h, mu, logw, x_mask, frames

            def encode_prosody2(ids: Tensor, length: Tensor, spk: Tensor, z_tok: Tensor, z_glob: Tensor,
                                scale: Tensor, temps: Tensor, rspk: Tensor):
                """As encode_prosody with the durations' row (ProsodyPredictor._draw): unit noise, the temperatures
                (pitch, durations) and the speaker whose rhythm the durations follow."""
                h, mu, logw, x_mask = m.encoder(ids, length, spk)
                cond, base = pros.condition(m, h, x_mask, spk, logw, pros.stats)
                hd, _, logwd, _ = m.encoder(ids, length, rspk)  # as the eager path: one encoder pass per speaker
                cd, bd = pros.condition(m, hd, x_mask, rspk, logwd, pros.stats)
                xm2 = torch.cat([x_mask, x_mask])
                if pros.kind == "drift":
                    out = pros.net(torch.cat([cond, cd]), xm2, torch.cat([z_tok * temps[0], z_tok * temps[1]]),
                                   torch.cat([z_glob * temps[0], z_glob * temps[1]]))
                else:
                    out = pros.net(torch.cat([cond, cd]), xm2)
                y = (torch.cat([base, bd]) + out[:, :2]) * xm2
                own = (rspk == spk)[:, None, None]  # no borrowed rhythm: the pauses keep the first row's (predict)
                ld = torch.where(boundary_tokens(ids)[:, None] & own, y[:1, :1], y[1:, :1])
                y = torch.cat([ld, y[:1, 1:]], 1)
                frames, pitch = pros.frames_and_pitch(y, out[:1, 2], x_mask, scale)
                h, _ = m.pitch_condition(h, x_mask, spk, pitch)
                return h, mu, logw, x_mask, frames

            bufs = [torch.zeros(1, nb, dtype=torch.long, device=dev), torch.full((1,), nb, device=dev),
                    torch.zeros(1, dtype=torch.long, device=dev)]
            if pros is not None:
                net = pros.net
                bufs += [torch.zeros(1, net.noise_tok, nb, device=dev), torch.zeros(1, net.noise_glob, device=dev),
                         torch.ones(1, device=dev)]
            if self.duration_row:
                bufs += [torch.ones(2, device=dev), torch.zeros(1, dtype=torch.long, device=dev)]
            fn = encode if pros is None else encode_prosody2 if self.duration_row else encode_prosody
            self.encoders[nb] = Graph(fn, bufs, self.pool, self.tf32)
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

    @torch.no_grad()
    def warmup(self, max_tokens: int = 384, max_frames: int = 1664) -> None:
        """Capture every bucket up to these lengths now rather than on first use. Without ``no_grad`` each graph's
        outputs would keep its autograd activations alive (~0.25 GB per frame bucket, ~6 GB in all)."""
        for n in range(self.token_bucket, max_tokens + 1, self.token_bucket):
            self._encoder(n)
        for t in range(self.frame_bucket, max_frames + 1, self.frame_bucket):
            self._generator(t)
        torch.cuda.synchronize()

    @torch.no_grad()
    def __call__(self, ids: Tensor, spk: Tensor, cfg_scale: float, temperature: float, length_scale: float,
                 generator: torch.Generator | None = None, prosody_temperature: float = 1.0,
                 duration_temperature: float | None = None, rhythm: Tensor | None = None) -> Tensor:
        """Token ids ``[1, N]`` -> normalised mel ``[1, n_mels, T]``, as ``synthesize`` with ``steps=1`` (with a
        prosody predictor: as ``ProsodyPredictor.predict`` at ``prosody_temperature`` / ``duration_temperature``,
        for the rhythm of speaker ``rhythm``, then ``synthesize``). The last two need ``duration_row``."""
        if (duration_temperature is not None or rhythm is not None) and self.prosody_durations == "sampled" \
                and not self.duration_row:
            raise ValueError("a duration temperature or rhythm of its own needs GraphedAcoustic(duration_row=True)")
        n = ids.shape[1]
        enc = self._encoder(n)
        nb = enc.inputs[0].shape[1]
        padded = torch.zeros(1, nb, dtype=torch.long, device=self.device)
        padded[:, :n] = ids
        if self.prosody is None:
            h, mu, logw, x_mask = enc(padded, torch.tensor([n], device=self.device), spk)
            attn, y_len = durations_to_alignment(logw, x_mask, length_scale)  # reads T on the host
        else:  # the prosody noise first, as ProsodyPredictor._draw (one pass, spread 1)
            net = self.prosody.net
            z_tok = torch.zeros(1, net.noise_tok, nb, device=self.device)
            z_glob = torch.zeros(1, net.noise_glob, device=self.device)
            if self.prosody.kind == "drift":
                z_tok[..., :n] = torch.randn(1, net.noise_tok, n, device=self.device, generator=generator)
                z_glob = torch.randn(1, net.noise_glob, device=self.device, generator=generator)
                if not self.duration_row:  # the two-row graph scales the unit noise per row
                    z_tok *= prosody_temperature
                    z_glob = z_glob * prosody_temperature
            scale = torch.full((1,), float(length_scale), device=self.device)
            extra = []
            if self.duration_row:
                dt = prosody_temperature if duration_temperature is None else duration_temperature
                extra = [torch.tensor([prosody_temperature, dt], dtype=torch.float32, device=self.device),
                         spk if rhythm is None else rhythm]
            h, mu, logw, x_mask, frames = enc(padded, torch.tensor([n], device=self.device), spk, z_tok, z_glob, scale,
                                              *extra)
            if self.prosody_durations == "sampled":
                attn, y_len = frames_to_alignment(frames, x_mask)
            else:
                attn, y_len = durations_to_alignment(logw, x_mask, length_scale)
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
