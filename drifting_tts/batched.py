"""Batched synthesis for many concurrent requests (opt-in; ``scripts/bench_concurrency.py``, docs/LATENCY.md).

* :func:`acoustic_batch`: one sentence of each of ``B`` requests in one padded pass of the text encoder, the prosody
  predictor and the DiT. Each request draws its noise from its own generator, in the order of the single-request path
  (prosody noise, then the DiT's noise and style codes), and the padding is masked, so each row equals
  :meth:`Synthesizer._mel` with the same seed up to float rounding.
* :func:`vocode_masked`: a Vocos vocoder on a batch of mels of different lengths, each row as if vocoded alone (the
  padding is zeroed before every convolution and left out of the inverse STFT); other vocoders run row by row.
* :func:`stream_batched`: the streaming windows of :func:`drifting_tts.fast.stream_vocoder` for many requests,
  one batch per round: every request gets its first piece after the first round, then one more piece per round.

The batched passes run eagerly (no CUDA graphs: the shapes change with every batch).
"""

from __future__ import annotations

import atexit
import logging
import random
import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .alignment import sequence_mask
from .audio import HOP_LENGTH, N_FFT
from .chunked import Chunking, _take, autocast, to_device  # noqa: F401 (_take re-exported)
from .noise import DiTNoise, Philox, philox_normal, philox_randint
from .text import normalize, split_sentences, text_to_ids


@dataclass(frozen=True)
class Serving:
    """Speed options of the batched passes; the defaults are the reference behaviour (docs/LATENCY.md).

    ``buckets``: a batch's sentences are sorted by length and split into up to this many groups of at least
    ``min_bucket`` rows for the text encoder and the prosody predictor (and, without chunking, for the DiT), so less
    of each pass is padding; in the first round the text frontend of each group runs while the GPU works on the
    groups before it. Each row is as without them, up to float rounding.
    ``dit_dtype`` / ``text_dtype`` / ``prosody_dtype``: the DiT / the text encoder / the prosody predictor's network
    under autocast in ``"fp16"`` or ``"bf16"`` (the duration and pitch regressors and the durations' rounding stay
    fp32). They change the output: a quality check is needed (docs/LATENCY.md).
    ``vocoder_dtype``: the batched Vocos under autocast (``"fp16"``: log-mel distance 0.014 dB against fp32).
    ``compile`` / ``compile_text``: ``torch.compile`` the DiT / the text pass (dynamic shapes; compiled on the first
    batch, about half a minute each); float rounding changes, so a duration can round to another frame.
    ``graphs`` (with them): CUDA graphs of the compiled passes (mode ``reduce-overhead``), the shapes padded to a few
    sizes; it saves most of the host's time per pass.
    ``compile_vocoder``: ``torch.compile`` the batched Vocos (float rounding only). ``autotune``: compile the DiT with
    ``max-autotune`` (its matrix multiplies tuned for the GPU; minutes of compilation the first time).
    ``frontend_workers``: processes that run the first round's text frontend (normalisation, sentence split) for
    the length groups in parallel (0: in this process, group by group).
    ``pipeline`` (with buckets): the first round's DiT window and vocoder run group by group too, and each group's
    first pieces are yielded as soon as they are ready (lower median, higher maximum first-audio time); else one
    acoustic batch follows the groups' text passes."""

    buckets: int = 1
    min_bucket: int = 64
    dit_dtype: str | None = None
    prosody_dtype: str | None = None
    compile: bool = False
    text_dtype: str | None = None
    compile_text: bool = False
    vocoder_dtype: str | None = None
    graphs: bool = False
    pipeline: bool = False
    frontend_workers: int = 0
    compile_vocoder: bool = False
    autotune: bool = False


def length_groups(lengths: list[int], buckets: int, min_rows: int = 1) -> list[list[int]]:
    """Row indices in up to ``buckets`` groups of consecutive lengths, each of at least ``min_rows`` rows (one group,
    in order, for ``buckets`` <= 1)."""
    k = min(buckets, len(lengths) // max(1, min_rows))
    if k <= 1:
        return [list(range(len(lengths)))]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[j * len(order) // k: (j + 1) * len(order) // k] for j in range(k)]


def _randn(out: Tensor, g: torch.Generator) -> None:
    """``out[...] = torch.randn(out.shape, generator=g)``. On CUDA in place (one kernel, no copy): the generator gives
    each element by its index, so a slice of a padded batch gets the same values (not so on the CPU)."""
    if out.is_cuda:
        out.normal_(generator=g)
    else:
        out.copy_(torch.randn(out.shape, generator=g))


def _text_pass(model, prosody, text: Tensor, text_len: Tensor, spk: Tensor, z_tok: Tensor | None,
               z_glob: Tensor | None, prosody_temperature: float, length_scale: float, regressor_durations: bool,
               text_dtype: str | None = None, prosody_dtype: str | None = None) -> tuple:
    """The text encoder, the prosody predictor, the token pitch and the integer durations of a padded batch (no host
    synchronisation; the noise is given): ``(h, mu, x_mask, durations)``. ``text_dtype``: the encoder under
    autocast; the duration and pitch regressors and the rounding of the durations stay fp32 (on its output)."""
    dt = text_dtype not in (None, "fp32")
    with autocast(text_dtype, text.device.type):
        h, mu, logw, x_mask = model.encoder(text, text_len, spk)
    if dt:  # the durations from the fp32 regressor on the encoder's output
        h, mu = h.float(), mu.float()
        s = model.encoder.spk(spk)[:, :, None].expand(-1, -1, h.shape[-1])
        logw = model.encoder.duration(torch.cat([h, s], 1), x_mask)
    frames = None
    if prosody is None:
        if model.pitch_enabled:
            h, _ = model.pitch_condition(h, x_mask, spk)
    else:
        cond, base = prosody.condition(model, h, x_mask, spk, logw, prosody.stats)
        with autocast(prosody_dtype, text.device.type):
            out = (prosody.net(cond, x_mask, z_tok * prosody_temperature, z_glob * prosody_temperature)
                   if prosody.kind == "drift" else prosody.net(cond, x_mask))
        out = out.float()
        frames, pitch = prosody.frames_and_pitch((base + out[:, :2]) * x_mask, out[:, 2], x_mask, length_scale)
        h, _ = model.pitch_condition(h, x_mask, spk, pitch)
    if frames is None or regressor_durations:  # durations_to_alignment's frames
        frames = torch.ceil(torch.exp(logw) * x_mask * length_scale).clamp_min(0)[:, 0]
    return h, mu, x_mask, frames


def _encode(model, ids: list[list[int]], spk: int, length_scale: float, generators: list[torch.Generator],
            prosody, prosody_temperature: float, prosody_durations: str, serving: Serving | None = None) -> tuple:
    """The text half of :func:`prepare_batch` for one group, queued on the GPU without reading anything back: the
    prosody noise (each row's first draws), then :func:`_text_pass`. Returns ``(h, mu, x_mask, durations)``."""
    sv = serving or Serving()
    dev = next(model.parameters()).device
    lens = [len(x) for x in ids]
    B, n = len(ids), max(lens)
    text = to_device([x + [0] * (n - len(x)) for x in ids], dev)
    text_len = to_device(lens, dev)
    spk = torch.full((B,), int(spk), dtype=torch.long, device=dev)
    z_tok = z_glob = None
    if prosody is not None and prosody.kind == "drift":  # ProsodyPredictor._draw: token noise, then global noise
        net = prosody.net
        if isinstance(generators, Philox):  # drifting_tts.noise: one launch per stream for the whole group
            seeds, sentences = generators.tensors(dev)
            z_tok = philox_normal(seeds, sentences, "prosody_tok", net.noise_tok, n, lengths=text_len)
            z_glob = philox_normal(seeds, sentences, "prosody_glob", net.noise_glob, 1)[:, :, 0]
        else:
            z_tok = torch.zeros(B, net.noise_tok, n, device=dev)
            z_glob = torch.empty(B, net.noise_glob, device=dev)
            for b, g in enumerate(generators):
                _randn(z_tok[b, :, : lens[b]], g)
                _randn(z_glob[b], g)
    base = _text_pass
    if not sv.compile_text:
        return base(model, prosody, text, text_len, spk, z_tok, z_glob, float(prosody_temperature),
                    float(length_scale), prosody_durations == "regressor", sv.text_dtype, sv.prosody_dtype)
    fn = _compiled(model, "text_pass", base, sv.graphs)
    if sv.graphs:  # a few shapes: rows to a multiple of 16 (one-token dummies), tokens to a multiple of 32
        Bp, Np = _round_up(B, 16), _round_up(n, 32)
        text, text_len, spk = _pad(text, (Bp, Np)), _pad(text_len, (Bp,), 1), _repeat_first(spk, Bp)
        if z_tok is not None:
            z_tok, z_glob = _pad(z_tok, (Bp, z_tok.shape[1], Np)), _pad(z_glob, (Bp, z_glob.shape[1]))
    h, mu, x_mask, frames = fn(model, prosody, text, text_len, spk, z_tok, z_glob, float(prosody_temperature),
                               float(length_scale), prosody_durations == "regressor", sv.text_dtype, sv.prosody_dtype)
    if sv.graphs:  # out of the graphs' memory, unpadded
        h, mu, x_mask, frames = (h[:B, :, :n].clone(), mu[:B, :, :n].clone(), x_mask[:B, :, :n].clone(),
                                 frames[:B, :n].clone())
    return h, mu, x_mask, frames


def expand_by_durations(tok: Tensor, durations: Tensor, frames: int) -> Tensor:
    """Token features ``[B, C, N]`` repeated by their integer durations ``[B, N]`` -> ``[B, C, frames]`` (zero after
    each row's total): ``DriftingTTS.frame_condition`` with the hard alignment of ``frames_to_alignment``, as one
    gather instead of a batched matrix multiply with a ``[B, N, frames]`` 0/1 matrix (the same values)."""
    end = torch.cumsum(durations, 1)  # token n covers frames [end[n - 1], end[n])
    t = torch.arange(frames, device=tok.device, dtype=end.dtype)[None].expand(tok.shape[0], frames).contiguous()
    idx = torch.searchsorted(end, t, right=True).clamp_max(tok.shape[-1] - 1)
    out = tok.gather(2, idx[:, None].expand(-1, tok.shape[1], -1))
    return out * (t < end[:, -1:])[:, None]


@torch.no_grad()
def _finish(model, groups: list[list[int]], encoded: list[tuple], spk: int, cfg_scale: float, temperature: float,
            generators: list[torch.Generator]) -> tuple[Tensor, ...]:
    """The rest of :func:`prepare_batch`: each group's alignment and aligned condition, back in the order of the rows
    (``groups`` holds row indices), then every row's noise and style codes."""
    dev = next(model.parameters()).device
    B = sum(len(r) for r in groups)
    # the token features [mu; h] and integer durations of every row, in the order of the rows
    toks = [torch.cat([mu, h], 1) for h, mu, _, _ in encoded]
    durs = [frames.to(x_mask.dtype) * x_mask[:, 0] for _, _, x_mask, frames in encoded]
    if len(groups) == 1 and groups[0] == list(range(B)):
        tok, dur = toks[0], durs[0]
    else:
        tok = torch.zeros(B, toks[0].shape[1], max(x.shape[-1] for x in toks), device=dev)
        dur = torch.zeros(B, tok.shape[-1], device=dev)
        for rows, x, w in zip(groups, toks, durs):
            idx = to_device(rows, dev)
            tok[idx, :, : x.shape[-1]] = x
            dur[idx, : w.shape[-1]] = w
    y_len = dur.sum(1).clamp_min(1).long()  # frames_to_alignment's lengths
    t = y_len.tolist()
    cond = expand_by_durations(tok, dur, max(t))
    T = cond.shape[-1]
    gen = model.generator
    spk = torch.full((B,), int(spk), dtype=torch.long, device=dev)
    alpha = torch.full((B,), float(cfg_scale), device=dev)
    if isinstance(generators, Philox):  # the DiT's noise is drawn window by window (DiTNoise.window / full)
        seeds, sentences = generators.tensors(dev)
        labels = philox_randint(seeds, sentences, "style", max(1, gen.noise_coords), gen.noise_classes)
        return DiTNoise(generators, model.n_mels, y_len, temperature, T), cond, spk, alpha, labels, y_len
    # DriftingTTS.synthesize: the mel noise, then the style codes (rollout); zero after each row's length, as the
    # single-request path pads a partial patch with zeros
    z = torch.zeros(B, model.n_mels, T, device=dev)
    labels = torch.empty(B, max(1, gen.noise_coords), dtype=torch.long, device=dev)
    for b, g in enumerate(generators):
        _randn(z[b, :, : t[b]], g)
        if z.is_cuda:
            labels[b].random_(0, gen.noise_classes, generator=g)
        else:
            labels[b] = torch.randint(0, gen.noise_classes, labels[b].shape, generator=g)
    return z * temperature, cond, spk, alpha, labels, y_len


@torch.no_grad()
def prepare_batch(model, ids: list[list[int]], spk: int, cfg_scale: float, temperature: float, length_scale: float,
                  generators: list[torch.Generator] | Philox, prosody=None, prosody_temperature: float = 1.0,
                  prosody_durations: str = "sampled", serving: Serving | None = None) -> tuple[Tensor, ...]:
    """The DiT's inputs of :func:`acoustic_batch` (the text encoder, the prosody predictor, the alignment and every
    row's draws): the noise scaled by ``temperature`` ``[B, n_mels, T]`` and the aligned condition ``[B, C, T]``
    (both zero after each row's length), the speaker ids ``[B]``, the CFG scales ``[B]``, the style codes and the
    lengths ``[B]``. ``serving``: :class:`Serving` (length buckets, precision, compilation of the text pass)."""
    sv = serving or Serving()
    groups = length_groups([len(x) for x in ids], sv.buckets, sv.min_bucket)
    def sub(rows):
        return generators.subset(rows) if isinstance(generators, Philox) else [generators[r] for r in rows]

    encoded = [_encode(model, [ids[r] for r in rows], spk, length_scale, sub(rows), prosody, prosody_temperature,
                       prosody_durations, serving) for rows in groups]
    return _finish(model, groups, encoded, spk, cfg_scale, temperature, generators)


def _round_up(n: int, k: int) -> int:
    return -(-n // k) * k


def _pad(x: Tensor, sizes: tuple[int, ...], value=0) -> Tensor:
    """``x`` padded at the end of each dimension to ``sizes`` with ``value``."""
    if tuple(x.shape) == tuple(sizes):
        return x
    out = torch.full(sizes, value, dtype=x.dtype, device=x.device)
    out[tuple(slice(0, n) for n in x.shape)] = x
    return out


def _repeat_first(x: Tensor, rows: int) -> Tensor:
    """``x`` with its first row repeated up to ``rows`` rows (no host synchronisation)."""
    return x if x.shape[0] == rows else torch.cat([x, x[:1].expand(rows - x.shape[0], *x.shape[1:])])


def _compiled(model, name: str, fn, graphs: bool = False, autotune: bool = False):
    """``torch.compile(fn)`` with dynamic shapes, cached on ``model``; ``graphs``: mode ``reduce-overhead`` (CUDA
    graphs, one per input shape: the callers pad the shapes to a few sizes); ``autotune``: ``max-autotune``."""
    mode = ("max-autotune" if graphs else "max-autotune-no-cudagraphs") if autotune else \
        ("reduce-overhead" if graphs else None)
    attr = f"_compiled_{name}_{mode}"
    if getattr(model, attr, None) is None:
        logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)  # harmless dynamic-shape noise
        setattr(model, attr, torch.compile(fn, dynamic=True, mode=mode))
    return getattr(model, attr)


def batched_generator(model, dtype: str | None = None, compile: bool = False, graphs: bool = False,
                      autotune: bool = False):
    """``generate(z, cond, spk, alpha, mask, labels)`` for a padded batch: the DiT's rollout with the valid-frame
    mask (the interface of :class:`~drifting_tts.chunked.WindowedMel`), under autocast in ``dtype`` and
    ``torch.compile``-d (dynamic shapes, compiled once per model) if asked. ``graphs`` (with ``compile``): CUDA graphs
    of the compiled DiT; the batch is padded to a multiple of 16 rows and 32 frames (masked), so that a few graphs
    serve every batch."""
    gen = _compiled(model, "generate", model.generate, graphs, autotune) if compile else model.generate

    def generate(z, cond, spk, alpha, mask, labels):
        B, W = z.shape[0], z.shape[-1]
        if compile and graphs:
            Bp, Wp = _round_up(B, 16), _round_up(W, 32)
            z, cond = _pad(z, (Bp, z.shape[1], Wp)), _pad(cond, (Bp, cond.shape[1], Wp))
            spk, alpha = _repeat_first(spk, Bp), _pad(alpha, (Bp,), 1.0)
            mask, labels = _pad(mask, (Bp, Wp), False), _pad(labels, (Bp, labels.shape[1]))
        with autocast(dtype, z.device.type):
            x = z
            for k in range(model.generator.num_steps):  # DriftingTTS.rollout
                x = gen(x, cond, spk, alpha, mask=mask, noise_labels=labels, step=k)
        return x[:B, :, :W].float().clone() if compile and graphs else x.float()

    return generate


@torch.no_grad()
def acoustic_batch(model, ids: list[list[int]], spk: int, cfg_scale: float, temperature: float,
                   length_scale: float, generators: list[torch.Generator], prosody=None,
                   prosody_temperature: float = 1.0, prosody_durations: str = "sampled",
                   serving: Serving | None = None) -> tuple[Tensor, Tensor]:
    """Token ids of ``B`` sentences (one voice) -> normalised mels ``[B, n_mels, T]`` (zero after each row's length)
    and the lengths ``[B]``. ``generators``: one per row. ``prosody``: a one-pass predictor (``drift`` / ``mse``,
    :func:`drifting_tts.fast.graphable`) whose token pitch (and, with ``prosody_durations="sampled"``, durations)
    replace the regressors', as ``ProsodyPredictor.predict`` followed by ``DriftingTTS.synthesize``."""
    sv = serving or Serving()
    z, cond, spk, alpha, labels, y_len = prepare_batch(model, ids, spk, cfg_scale, temperature, length_scale,
                                                       generators, prosody, prosody_temperature, prosody_durations,
                                                       sv)
    generate = batched_generator(model, sv.dit_dtype, sv.compile, sv.graphs, sv.autotune)
    return _dit_whole(generate, z, cond, spk, alpha, labels, y_len, sv), y_len


def _dit_whole(generate, z: Tensor, cond: Tensor, spk: Tensor, alpha: Tensor, labels: Tensor, y_len: Tensor,
               serving: Serving) -> Tensor:
    """Whole-sentence mels of a padded batch, zero after each row's length (in length buckets: each one trimmed to
    its longest row)."""
    T, t = cond.shape[-1], y_len.tolist()
    groups = length_groups(t, serving.buckets, serving.min_bucket)
    if len(groups) == 1:
        mask = sequence_mask(y_len, T)
        return generate(z, cond, spk, alpha, mask, labels) * mask[:, None]
    mel = torch.zeros_like(z)
    for rows in groups:
        idx, m = to_device(rows, z.device), max(t[r] for r in rows)
        mask = sequence_mask(y_len[idx], m)
        mel[idx, :, :m] = generate(z[idx, :, :m], cond[idx, :, :m], spk[idx], alpha[idx], mask, labels[idx]) \
            * mask[:, None]
    return mel


def batchable(synth) -> bool:
    """Whether :func:`synth_mels` supports this Synthesizer's prosody source: none, or a one-pass predictor without
    a duration row of its own (no separate duration temperature, no borrowed rhythm) and no second pitch predictor."""
    from .fast import graphable

    return (graphable(synth.prosody, synth.prosody_spread) and not synth._duration_row()
            and getattr(synth, "prosody_pitch", None) is None)


def _batch_args(synth, sentences: list[str], speaker, prosody_temperature: float | None) -> tuple:
    if not batchable(synth):
        raise NotImplementedError("batched synthesis supports no prosody predictor or a one-pass one (drift / mse) "
                                  "without a duration temperature, rhythm or pitch predictor of its own")
    _, tempo = synth._speaker(speaker)
    pt = synth.prosody_temperature if prosody_temperature is None else prosody_temperature
    return [text_to_ids(s, normalized=True) for s in sentences], synth.speaker_id(speaker), tempo, \
        1.0 if pt is None else pt


@torch.no_grad()
def synth_mels(synth, sentences: list[str], speaker, cfg_scale: float, temperature: float,
               generators: list[torch.Generator], length_scale: float = 1.0,
               prosody_temperature: float | None = None, serving: Serving | None = None) -> tuple[Tensor, Tensor]:
    """:func:`acoustic_batch` with a :class:`~drifting_tts.synthesize.Synthesizer`'s model, prosody source and the
    voice's duration factor: normalised ``sentences`` -> normalised mels ``[B, n_mels, T]`` and lengths ``[B]``."""
    ids, spk, tempo, pt = _batch_args(synth, sentences, speaker, prosody_temperature)
    return acoustic_batch(synth.model, ids, spk, cfg_scale, temperature, length_scale * tempo, generators,
                          synth.prosody, pt, synth.prosody_durations, serving)


@torch.no_grad()
def synth_prepare(synth, sentences: list[str], speaker, cfg_scale: float, temperature: float,
                  generators: list[torch.Generator], length_scale: float = 1.0,
                  prosody_temperature: float | None = None, serving: Serving | None = None) -> tuple[Tensor, ...]:
    """:func:`prepare_batch` with a Synthesizer's model, prosody source and the voice's duration factor."""
    ids, spk, tempo, pt = _batch_args(synth, sentences, speaker, prosody_temperature)
    sv = serving or Serving()
    return prepare_batch(synth.model, ids, spk, cfg_scale, temperature, length_scale * tempo, generators,
                         synth.prosody, pt, synth.prosody_durations, sv)


def _masked_vocos(model, x: Tensor, m: Tensor) -> Tensor:
    """Vocos backbone + ISTFT head (BigVGAN-style framing, as ``Vocoder._vocos_bigvgan``) on ``x`` ``[B, C, T]``
    with the valid-frame mask ``m`` ``[B, 1, T]``: ``[B, T * hop]``, each row as vocoded alone."""
    bb, head = model.backbone, model.head
    h = bb.embed(x * m)  # each convolution sees zeros after the row's end, as its own padding
    h = bb.norm(h.transpose(1, 2)).transpose(1, 2)
    for block in bb.convnext:
        h = block(h * m)
    h = head.out(bb.final_layer_norm(h.transpose(1, 2))).transpose(1, 2).float()  # the inverse STFT in fp32
    mag, p = h.chunk(2, dim=1)
    spec = torch.exp(mag).clip(max=1e2) * (torch.cos(p) + 1j * torch.sin(p)) * m  # padded frames add nothing
    t = x.shape[-1]
    size, pad = (t - 1) * HOP_LENGTH + N_FFT, (N_FFT - HOP_LENGTH) // 2
    window = head.istft.window
    frames = torch.fft.irfft(spec, N_FFT, dim=1) * window[:, None]
    y = F.fold(frames, (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))[:, 0, 0]
    env = F.fold(window.square()[None, :, None] * m, (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))[:, 0, 0]
    return (y / env.clamp_min(1e-11))[:, pad: size - pad]


@torch.no_grad()
def vocode_masked(vocoder, mel: Tensor, lengths: Tensor, compile: bool = False) -> Tensor:
    """Unnormalised log-mels ``[B, n_mels, T]`` whose rows have ``lengths`` frames -> waveforms ``[B, T * hop]``:
    row ``b`` equals ``vocoder(mel[b:b+1, :, :lengths[b]])`` on its first ``lengths[b] * hop`` samples (zeros after).
    One batch for a Vocos with BigVGAN-style mels (vocos-ft, vocos-v2); other vocoders run row by row.
    ``compile``: the batch through ``torch.compile`` (float rounding only)."""
    B, _, T = mel.shape
    model = vocoder.model
    if (vocoder.kind == "vocos" and vocoder.mel == "bigvgan" and not vocoder.noise
            and getattr(getattr(model, "head", None), "istft", None) is not None
            and model.head.istft.padding != "center"):
        m = sequence_mask(lengths.to(mel.device), T)[:, None].float()
        fn = _compiled(model, "masked_vocos", _masked_vocos) if compile else _masked_vocos
        wav = fn(model, mel.to(vocoder.device).float(), m)
        keep = sequence_mask(lengths.to(mel.device) * HOP_LENGTH, T * HOP_LENGTH)
        return wav.clamp(-1, 1) * keep
    out = torch.zeros(B, T * HOP_LENGTH, device=mel.device)
    for b, t in enumerate(lengths.tolist()):
        out[b, : t * HOP_LENGTH] = vocoder(mel[b: b + 1, :, :t])[0]
    return out


def stream_windows(t: int, first: int = 32, chunk: int = 256, context: int | None = 32) -> list[tuple[int, int, int,
                                                                                                        int]]:
    """The windows of :func:`drifting_tts.fast.stream_vocoder` for a mel of ``t`` frames: ``(a, b, start, n)``,
    vocode frames ``[a, b)`` and keep ``n`` frames from frame ``start`` of the window."""
    if context is None or t <= first + context:
        return [(0, t, 0, t)]
    out = [(0, first + context, 0, first)]
    for s in range(first, t, chunk):
        a, b = max(0, s - context), min(t, s + chunk + context)
        out.append((a, b, s - a, min(chunk, t - s)))
    return out


_RAW_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_POOLS: dict = {}


def frontend_pool(workers: int):
    """A persistent pool of ``workers`` processes for :func:`drifting_tts.text.frontend` (spawned once; they import
    only the text frontend)."""
    if workers not in _POOLS:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        from .text import frontend

        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        list(pool.map(frontend, [["Merhaba."]] * workers))  # start the processes now
        atexit.register(pool.shutdown, wait=False, cancel_futures=True)
        _POOLS[workers] = pool
    return _POOLS[workers]


class _WholeMels:
    """Whole-sentence mels of one acoustic batch, in the interface of :class:`~drifting_tts.chunked.WindowedMel`."""

    def __init__(self, mel: Tensor, lengths: list[int]):
        self.mel, self.committed = mel, lengths

    def step(self, rows: list[int]) -> None:
        raise RuntimeError("whole-sentence mels are complete")


class _Pieces:
    """One vocoder batch's pieces on their way to the host. ``pinned``: a copy into page-locked memory is queued
    right after the vocoder, so the host can queue more GPU work before it waits for them (:meth:`result`)."""

    def __init__(self, pieces: Tensor, entries: list[tuple[int, int, Tensor | None]], pinned: bool = False):
        self.entries, self.event = entries, None
        if pinned and pieces.is_cuda:
            self.host = torch.empty(pieces.shape, dtype=pieces.dtype, pin_memory=True)
            self.host.copy_(pieces, non_blocking=True)
            self.event = torch.cuda.Event()
            self.event.record()
        else:
            self.host = pieces.cpu()

    def result(self) -> list[tuple[int, Tensor]]:
        """``[(request index, piece), ...]``, each request's pause first when its sentence starts here."""
        if self.event is not None:
            self.event.synchronize()
        out = []
        for row, (i, keep, gap) in enumerate(self.entries):
            if gap is not None:
                out.append((i, gap))
            out.append((i, self.host[row, :keep]))
        return out


@torch.no_grad()
def stream_batched(synth, texts: list[str], speaker=None, cfg_scale: float = 1.0, temperature: float = 1.0,
                   seeds: list[int] | None = None, length_scale: float = 1.0, pause=0.0, first: int = 32,
                   chunk: int = 256, prosody_temperature: float | None = None,
                   chunked: Chunking | bool | None = None,
                   serving: Serving | None = None) -> Iterator[list[tuple[int, Tensor]]]:
    """:meth:`Synthesizer.stream` for many requests at once (one voice). Yields round by round the pieces now on the
    host, ``[(request index, waveform piece), ...]``; each request's pieces join into what ``synth.stream`` gives for
    its text and seed (default seed: its index), up to float rounding.

    A round runs the acoustic model on the next sentence of every request that has used up its mel (all requests in
    the first round), then one masked vocoder batch with the next streaming window of every unfinished request.
    ``pause`` (seconds, a callable or ``"punct"``) is the silence between a request's sentences.

    ``chunked`` (a :class:`~drifting_tts.chunked.Chunking`, or ``True`` for its defaults): the DiT streams too, as
    ``synth.stream(..., chunked=...)``. A request's new sentence then runs the text encoder, the prosody predictor
    and only the DiT's first window in its round; every round runs one more DiT window per request (one batch per
    acoustic batch) before the vocoder batch, so the first round costs the first windows, not the longest sentence.

    ``serving``: speed options of the batched passes (:class:`Serving`; default: none). With length buckets, the
    first round's text side runs group by group (requests grouped by the raw length of their first sentence): a
    group's text frontend and text pass are queued while the GPU still works on the group before; with
    ``serving.pipeline`` each group's first pieces are also yielded as soon as they are on the host."""
    from .chunked import WindowedMel
    from .synthesize import DEFAULT_VOICE, silence

    chunking = Chunking() if chunked is True else chunked or None
    if chunking is not None:
        chunk = chunking.chunk
    speaker = DEFAULT_VOICE if speaker is None else speaker
    voc, dev = synth.vocoder, synth.device
    pause = synth._pause(pause, speaker)
    seeds = list(range(len(texts))) if seeds is None else seeds
    reqs = [{"text": t, "sentences": None, "k": 0, "windows": deque(), "rng": random.Random(s), "g": None}
            for t, s in zip(texts, seeds)]
    batches: list = []  # one per acoustic batch: _WholeMels or WindowedMel (normalised mels)
    sv = serving or Serving()
    generate = batched_generator(synth.model, sv.dit_dtype, sv.compile, sv.graphs, sv.autotune)
    head = first + (voc.context or 0)

    def keys(rows: list[int]):
        """The noise of these requests' next sentences: their generators, or their counter-based keys."""
        if chunking is not None and chunking.noise == "philox":
            return Philox([seeds[i] for i in rows], [reqs[i]["k"] for i in rows])
        for i in rows:
            if reqs[i]["g"] is None:
                reqs[i]["g"] = torch.Generator(device=dev).manual_seed(seeds[i])
        return [reqs[i]["g"] for i in rows]

    def add(need: list[int], inputs: tuple) -> None:
        """A new acoustic batch with the next sentence of each request in ``need``, and its streaming windows."""
        z, cond, spk, alpha, labels, y_len = inputs
        lens = y_len.tolist()
        if chunking is None:
            batches.append(_WholeMels(_dit_whole(generate, z if isinstance(z, Tensor) else z.full(), cond, spk,
                                                 alpha, labels, y_len, sv), lens))
        else:
            batches.append(WindowedMel(generate, z, cond, spk, alpha, labels, lens, chunking, head))
        for row, (i, t) in enumerate(zip(need, lens)):
            r = reqs[i]
            gap = silence(pause, r["sentences"][r["k"] - 1], r["rng"]) if r["k"] else None
            r["windows"].extend((len(batches) - 1, row, *w, k == 0)
                                for k, w in enumerate(stream_windows(t, first, chunk, voc.context)))
            r["gap"], r["k"] = gap, r["k"] + 1

    def vocode(active: list[int], pinned: bool = False) -> _Pieces:
        """The next streaming window of every request in ``active`` as one vocoder batch (after the DiT windows
        they still need: one per request and round when the chunks are aligned)."""
        ws = [reqs[i]["windows"].popleft() for i in active]
        while True:
            late = {}
            for w in ws:
                if batches[w[0]].committed[w[1]] < w[3]:
                    late.setdefault(w[0], []).append(w[1])
            if not late:
                break
            for src, rows in late.items():
                batches[src].step(rows)
        width = max(w[3] - w[2] for w in ws)
        parts, order = [], []
        for src in sorted({w[0] for w in ws}):  # one gather per acoustic batch the windows come from
            sel = [j for j, w in enumerate(ws) if w[0] == src]
            rows, starts = to_device([[ws[j][1] for j in sel], [ws[j][2] for j in sel]], dev)
            parts.append(_take(batches[src].mel[rows], starts, width))
            order += sel
        x = synth.stats.denormalize(parts[0] if len(parts) == 1 else torch.cat(parts))
        lengths = to_device([ws[j][3] - ws[j][2] for j in order], dev)
        with autocast(sv.vocoder_dtype, x.device.type):
            wav = vocode_masked(voc, x, lengths, sv.compile_vocoder).float()
        keep = [ws[j][5] * HOP_LENGTH for j in order]
        pieces = _take(wav, to_device([ws[j][4] * HOP_LENGTH for j in order], dev), max(keep))
        entries = [(active[j], keep[row], reqs[active[j]].pop("gap", None) if ws[j][6] else None)
                   for row, j in enumerate(order)]  # the pause before a sentence's first piece
        return _Pieces(pieces, entries, pinned)

    # the first round, group by group: frontend, text pass, alignment and draws, first DiT window, first vocoder
    # window; the next group's frontend and text pass are queued before the previous group's pieces are awaited
    if not batchable(synth):
        _batch_args(synth, [], speaker, prosody_temperature)  # raises
    _, tempo = synth._speaker(speaker)
    spk_id = synth.speaker_id(speaker)
    pt = synth.prosody_temperature if prosody_temperature is None else prosody_temperature
    est = [len(_RAW_SENTENCE_END.split(t.strip(), maxsplit=1)[0]) for t in texts]
    pending, rows, groups, encoded = None, [], [], []
    first_groups = length_groups(est, sv.buckets, sv.min_bucket)
    jobs = None
    if sv.frontend_workers:  # every group's frontend at once, in other processes
        from .text import frontend

        pool = frontend_pool(sv.frontend_workers)
        jobs = [pool.submit(frontend, [texts[i] for i in group]) for group in first_groups]
    for j, group in enumerate(first_groups):
        kept = []
        done = jobs[j].result() if jobs else [split_sentences(normalize(reqs[i]["text"])) for i in group]
        for i, sentences in zip(group, done):
            reqs[i]["sentences"] = sentences
            if sentences:
                kept.append(i)
        if not kept:
            continue
        gens = keys(kept)
        enc = _encode(synth.model, [text_to_ids(reqs[i]["sentences"][0], normalized=True) for i in kept], spk_id,
                      length_scale * tempo, gens, synth.prosody, 1.0 if pt is None else pt, synth.prosody_durations,
                      sv)
        if not sv.pipeline:  # one acoustic batch after the groups' text passes
            groups.append(list(range(len(rows), len(rows) + len(kept))))
            rows += kept
            encoded.append(enc)
            continue
        if pending is not None:
            yield pending.result()
        add(kept, _finish(synth.model, [list(range(len(kept)))], [enc], spk_id, cfg_scale, temperature, gens))
        pending = vocode(kept, pinned=True)
    if rows:
        add(rows, _finish(synth.model, groups, encoded, spk_id, cfg_scale, temperature, keys(rows)))
        pending = vocode(rows)
    if pending is not None:
        yield pending.result()
    while True:
        need = [i for i, r in enumerate(reqs) if not r["windows"] and r["k"] < len(r["sentences"])]
        if need:
            add(need, synth_prepare(synth, [reqs[i]["sentences"][reqs[i]["k"]] for i in need], speaker, cfg_scale,
                                    temperature, keys(need), length_scale, prosody_temperature, sv))
        active = [i for i, r in enumerate(reqs) if r["windows"]]
        if not active:
            return
        yield vocode(active).result()
