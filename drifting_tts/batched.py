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
from .chunked import Chunking, _take, autocast  # noqa: F401 (_take re-exported)
from .models.text_encoder import frames_to_alignment
from .text import normalize, split_sentences, text_to_ids


@dataclass(frozen=True)
class Serving:
    """Speed options of the batched passes; the defaults are the reference behaviour (docs/LATENCY.md).

    ``buckets``: the batch's sentences are sorted by length and split into this many groups for the text encoder and
    the prosody predictor (and, without chunking, for the DiT), so less of each pass is padding (each row as without
    them, up to float rounding). ``dit_dtype`` / ``prosody_dtype``: run the DiT / the prosody predictor's network under
    autocast in ``"fp16"`` or ``"bf16"`` (they change the output: a quality check is needed). ``compile``:
    ``torch.compile`` the DiT (dynamic shapes; the first batch compiles it, about half a minute)."""

    buckets: int = 1
    dit_dtype: str | None = None
    prosody_dtype: str | None = None
    compile: bool = False


def length_groups(lengths: list[int], buckets: int) -> list[list[int]]:
    """Row indices in ``buckets`` groups of consecutive lengths (one group, in order, for ``buckets`` <= 1)."""
    if buckets <= 1 or len(lengths) < 2:
        return [list(range(len(lengths)))]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    k = min(buckets, len(order))
    return [order[j * len(order) // k: (j + 1) * len(order) // k] for j in range(k)]


def _randn(out: Tensor, g: torch.Generator) -> None:
    """``out[...] = torch.randn(out.shape, generator=g)``. On CUDA in place (one kernel, no copy): the generator gives
    each element by its index, so a slice of a padded batch gets the same values (not so on the CPU)."""
    if out.is_cuda:
        out.normal_(generator=g)
    else:
        out.copy_(torch.randn(out.shape, generator=g))


def _encode(model, ids: list[list[int]], spk: int, length_scale: float, generators: list[torch.Generator],
            prosody, prosody_temperature: float, prosody_durations: str, prosody_dtype: str | None) -> tuple:
    """The text half of :func:`prepare_batch` for one group, queued on the GPU without reading anything back: the
    prosody noise (each row's first draws), the text encoder, the prosody predictor, the token pitch and the
    integer durations. Returns ``(h, mu, x_mask, durations)``."""
    dev = next(model.parameters()).device
    lens = [len(x) for x in ids]
    B, n = len(ids), max(lens)
    text = torch.tensor([x + [0] * (n - len(x)) for x in ids]).to(dev)
    text_len = torch.tensor(lens, device=dev)
    spk = torch.full((B,), int(spk), dtype=torch.long, device=dev)
    drift = prosody is not None and prosody.kind == "drift"
    if drift:  # ProsodyPredictor._draw: token noise, then global noise, from each request's generator
        net = prosody.net
        z_tok = torch.zeros(B, net.noise_tok, n, device=dev)
        z_glob = torch.empty(B, net.noise_glob, device=dev)
        for b, g in enumerate(generators):
            _randn(z_tok[b, :, : lens[b]], g)
            _randn(z_glob[b], g)
    h, mu, logw, x_mask = model.encoder(text, text_len, spk)
    frames = None
    if prosody is None:
        if model.pitch_enabled:
            h, _ = model.pitch_condition(h, x_mask, spk)
    else:
        cond, base = prosody.condition(model, h, x_mask, spk, logw, prosody.stats)
        with autocast(prosody_dtype, dev.type):
            out = (prosody.net(cond, x_mask, z_tok * prosody_temperature, z_glob * prosody_temperature) if drift
                   else prosody.net(cond, x_mask))
        out = out.float()
        frames, pitch = prosody.frames_and_pitch((base + out[:, :2]) * x_mask, out[:, 2], x_mask, length_scale)
        h, _ = model.pitch_condition(h, x_mask, spk, pitch)
    if frames is None or prosody_durations == "regressor":  # durations_to_alignment's frames
        frames = torch.ceil(torch.exp(logw) * x_mask * length_scale).clamp_min(0)[:, 0]
    return h, mu, x_mask, frames


@torch.no_grad()
def _finish(model, groups: list[list[int]], encoded: list[tuple], spk: int, cfg_scale: float, temperature: float,
            generators: list[torch.Generator]) -> tuple[Tensor, ...]:
    """The rest of :func:`prepare_batch`: each group's alignment and aligned condition, back in the order of the rows
    (``groups`` holds row indices), then every row's noise and style codes."""
    dev = next(model.parameters()).device
    B = sum(len(r) for r in groups)
    conds, y_lens = [], []
    for h, mu, x_mask, frames in encoded:
        attn, y_len = frames_to_alignment(frames, x_mask)
        conds.append(model.frame_condition(h, mu, attn))
        y_lens.append(y_len)
    if len(groups) == 1 and groups[0] == list(range(B)):
        cond, y_len = conds[0], y_lens[0]
    else:  # back to the order of the rows, padded to the longest
        cond = torch.zeros(B, conds[0].shape[1], max(c.shape[-1] for c in conds), device=dev)
        y_len = torch.empty(B, dtype=torch.long, device=dev)
        for rows, c, yl in zip(groups, conds, y_lens):
            idx = torch.tensor(rows, device=dev)
            cond[idx, :, : c.shape[-1]] = c
            y_len[idx] = yl
    T, t = cond.shape[-1], y_len.tolist()
    gen = model.generator
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
    alpha = torch.full((B,), float(cfg_scale), device=dev)
    return z * temperature, cond, torch.full((B,), int(spk), dtype=torch.long, device=dev), alpha, labels, y_len


@torch.no_grad()
def prepare_batch(model, ids: list[list[int]], spk: int, cfg_scale: float, temperature: float, length_scale: float,
                  generators: list[torch.Generator], prosody=None, prosody_temperature: float = 1.0,
                  prosody_durations: str = "sampled", buckets: int = 1,
                  prosody_dtype: str | None = None) -> tuple[Tensor, ...]:
    """The DiT's inputs of :func:`acoustic_batch` (the text encoder, the prosody predictor, the alignment and every
    row's draws): the noise scaled by ``temperature`` ``[B, n_mels, T]`` and the aligned condition ``[B, C, T]``
    (both zero after each row's length), the speaker ids ``[B]``, the CFG scales ``[B]``, the style codes and the
    lengths ``[B]``. ``buckets`` / ``prosody_dtype``: :class:`Serving`."""
    groups = length_groups([len(x) for x in ids], buckets)
    encoded = [_encode(model, [ids[r] for r in rows], spk, length_scale, [generators[r] for r in rows], prosody,
                       prosody_temperature, prosody_durations, prosody_dtype) for rows in groups]
    return _finish(model, groups, encoded, spk, cfg_scale, temperature, generators)


def batched_generator(model, dtype: str | None = None, compile: bool = False):
    """``generate(z, cond, spk, alpha, mask, labels)`` for a padded batch: the DiT's rollout with the valid-frame
    mask (the interface of :class:`~drifting_tts.chunked.WindowedMel`), under autocast in ``dtype`` and
    ``torch.compile``-d (dynamic shapes, compiled once per model) if asked."""
    gen = model.generate
    if compile:
        if getattr(model, "_compiled_generate", None) is None:
            logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)  # harmless dynamic-shape noise
            model._compiled_generate = torch.compile(model.generate, dynamic=True)
        gen = model._compiled_generate

    def generate(z, cond, spk, alpha, mask, labels):
        with autocast(dtype, z.device.type):
            x = z
            for k in range(model.generator.num_steps):  # DriftingTTS.rollout
                x = gen(x, cond, spk, alpha, mask=mask, noise_labels=labels, step=k)
        return x.float()

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
                                                       sv.buckets, sv.prosody_dtype)
    generate = batched_generator(model, sv.dit_dtype, sv.compile)
    return _dit_whole(generate, z, cond, spk, alpha, labels, y_len, sv.buckets), y_len


def _dit_whole(generate, z: Tensor, cond: Tensor, spk: Tensor, alpha: Tensor, labels: Tensor, y_len: Tensor,
               buckets: int) -> Tensor:
    """Whole-sentence mels of a padded batch, zero after each row's length (in length buckets: each one trimmed to
    its longest row)."""
    T, t = cond.shape[-1], y_len.tolist()
    groups = length_groups(t, buckets)
    if len(groups) == 1:
        mask = sequence_mask(y_len, T)
        return generate(z, cond, spk, alpha, mask, labels) * mask[:, None]
    mel = torch.zeros_like(z)
    for rows in groups:
        idx, m = torch.tensor(rows, device=z.device), max(t[r] for r in rows)
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
                         synth.prosody, pt, synth.prosody_durations, sv.buckets, sv.prosody_dtype)


def _masked_vocos(model, x: Tensor, m: Tensor) -> Tensor:
    """Vocos backbone + ISTFT head (BigVGAN-style framing, as ``Vocoder._vocos_bigvgan``) on ``x`` ``[B, C, T]``
    with the valid-frame mask ``m`` ``[B, 1, T]``: ``[B, T * hop]``, each row as vocoded alone."""
    bb, head = model.backbone, model.head
    h = bb.embed(x * m)  # each convolution sees zeros after the row's end, as its own padding
    h = bb.norm(h.transpose(1, 2)).transpose(1, 2)
    for block in bb.convnext:
        h = block(h * m)
    h = head.out(bb.final_layer_norm(h.transpose(1, 2))).transpose(1, 2)
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
def vocode_masked(vocoder, mel: Tensor, lengths: Tensor) -> Tensor:
    """Unnormalised log-mels ``[B, n_mels, T]`` whose rows have ``lengths`` frames -> waveforms ``[B, T * hop]``:
    row ``b`` equals ``vocoder(mel[b:b+1, :, :lengths[b]])`` on its first ``lengths[b] * hop`` samples (zeros after).
    One batch for a Vocos with BigVGAN-style mels (vocos-ft, vocos-v2); other vocoders run row by row."""
    B, _, T = mel.shape
    model = vocoder.model
    if (vocoder.kind == "vocos" and vocoder.mel == "bigvgan" and not vocoder.noise
            and getattr(getattr(model, "head", None), "istft", None) is not None
            and model.head.istft.padding != "center"):
        m = sequence_mask(lengths.to(mel.device), T)[:, None].float()
        wav = _masked_vocos(model, mel.to(vocoder.device).float(), m)
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


def _encode_fresh(synth, reqs: list[dict], fresh: list[int], speaker, length_scale: float,
                  prosody_temperature: float | None, buckets: int,
                  prosody_dtype: str | None = None) -> tuple[list[int], list[list[int]], list[tuple]]:
    """The first round's text side: the requests in ``buckets`` groups of similar raw first-sentence length (an
    estimate: only the speed depends on it); each group's text frontend runs, then its text pass (:func:`_encode`) is
    queued on the GPU before the next group's frontend, so the two overlap. Returns the requests that have a
    sentence (the batch's rows, group by group), the groups (row indices) and their text passes."""
    if not batchable(synth):
        _batch_args(synth, [], speaker, prosody_temperature)  # raises
    _, tempo = synth._speaker(speaker)
    spk = synth.speaker_id(speaker)
    pt = synth.prosody_temperature if prosody_temperature is None else prosody_temperature
    est = [len(_RAW_SENTENCE_END.split(reqs[i]["text"].strip(), maxsplit=1)[0]) for i in fresh]
    need, groups, encoded = [], [], []
    for rows in length_groups(est, buckets):
        kept = []
        for j in rows:
            r = reqs[fresh[j]]
            r["sentences"] = split_sentences(normalize(r["text"]))
            if r["sentences"]:
                kept.append(fresh[j])
        if not kept:
            continue
        groups.append(list(range(len(need), len(need) + len(kept))))
        need += kept
        ids = [text_to_ids(reqs[i]["sentences"][0], normalized=True) for i in kept]
        encoded.append(_encode(synth.model, ids, spk, length_scale * tempo, [reqs[i]["g"] for i in kept],
                               synth.prosody, 1.0 if pt is None else pt, synth.prosody_durations,
                               prosody_dtype))
    return need, groups, encoded


class _WholeMels:
    """Whole-sentence mels of one acoustic batch, in the interface of :class:`~drifting_tts.chunked.WindowedMel`."""

    def __init__(self, mel: Tensor, lengths: list[int]):
        self.mel, self.committed = mel, lengths

    def step(self, rows: list[int]) -> None:
        raise RuntimeError("whole-sentence mels are complete")


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

    ``serving``: speed options of the batched passes (:class:`Serving`; default: none)."""
    from .chunked import WindowedMel
    from .synthesize import DEFAULT_VOICE, silence

    chunking = Chunking() if chunked is True else chunked or None
    if chunking is not None:
        chunk = chunking.chunk
    speaker = DEFAULT_VOICE if speaker is None else speaker
    voc, dev = synth.vocoder, synth.device
    pause = synth._pause(pause, speaker)
    seeds = list(range(len(texts))) if seeds is None else seeds
    reqs = [{"text": t, "sentences": None, "k": 0, "windows": deque(), "rng": random.Random(s),
             "g": torch.Generator(device=dev).manual_seed(s)} for t, s in zip(texts, seeds)]
    batches: list = []  # one per acoustic batch: _WholeMels or WindowedMel (normalised mels)
    sv = serving or Serving()
    generate = batched_generator(synth.model, sv.dit_dtype, sv.compile)
    head = first + (voc.context or 0)
    while True:
        fresh = [i for i, r in enumerate(reqs) if r["sentences"] is None]
        if fresh:  # the first round: the text frontend, group by group, overlapped with the GPU
            need, groups, encoded = _encode_fresh(synth, reqs, fresh, speaker, length_scale, prosody_temperature,
                                                  sv.buckets, sv.prosody_dtype)
            if need:
                inputs = _finish(synth.model, groups, encoded, synth.speaker_id(speaker), cfg_scale, temperature,
                                 [reqs[i]["g"] for i in need])
        else:
            need = [i for i, r in enumerate(reqs) if not r["windows"] and r["k"] < len(r["sentences"])]
            if need:
                inputs = synth_prepare(synth, [reqs[i]["sentences"][reqs[i]["k"]] for i in need], speaker, cfg_scale,
                                       temperature, [reqs[i]["g"] for i in need], length_scale, prosody_temperature, sv)
        if need:
            z, cond, spk, alpha, labels, y_len = inputs
            lens = y_len.tolist()
            if chunking is None:
                batches.append(_WholeMels(_dit_whole(generate, z, cond, spk, alpha, labels, y_len, sv.buckets),
                                          lens))
            else:
                batches.append(WindowedMel(generate, z, cond, spk, alpha, labels, lens, chunking, head))
            for row, (i, t) in enumerate(zip(need, lens)):
                r = reqs[i]
                gap = silence(pause, r["sentences"][r["k"] - 1], r["rng"]) if r["k"] else None
                r["windows"].extend((len(batches) - 1, row, *w, k == 0)
                                    for k, w in enumerate(stream_windows(t, first, chunk, voc.context)))
                r["gap"], r["k"] = gap, r["k"] + 1
        active = [i for i, r in enumerate(reqs) if r["windows"]]
        if not active:
            return
        ws = [reqs[i]["windows"].popleft() for i in active]
        while True:  # the DiT windows these vocoder windows still need (one per request and round when aligned)
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
            rows = torch.tensor([ws[j][1] for j in sel], device=dev)
            starts = torch.tensor([ws[j][2] for j in sel], device=dev)
            parts.append(_take(batches[src].mel[rows], starts, width))
            order += sel
        x = synth.stats.denormalize(parts[0] if len(parts) == 1 else torch.cat(parts))
        lengths = torch.tensor([ws[j][3] - ws[j][2] for j in order], device=dev)
        wav = vocode_masked(voc, x, lengths)
        keep = [ws[j][5] * HOP_LENGTH for j in order]
        pieces = _take(wav, torch.tensor([ws[j][4] * HOP_LENGTH for j in order], device=dev), max(keep)).cpu()
        out = []
        for row, j in enumerate(order):
            i = active[j]
            gap = reqs[i].pop("gap", None) if ws[j][6] else None  # before a sentence's first piece
            if gap is not None:
                out.append((i, gap))
            out.append((i, pieces[row, : keep[row]]))
        yield out
