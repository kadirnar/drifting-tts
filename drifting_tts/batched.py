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

import random
from collections import deque
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from .alignment import sequence_mask
from .audio import HOP_LENGTH, N_FFT
from .models.text_encoder import durations_to_alignment, frames_to_alignment
from .text import normalize, split_sentences, text_to_ids


@torch.no_grad()
def acoustic_batch(model, ids: list[list[int]], spk: int, cfg_scale: float, temperature: float,
                   length_scale: float, generators: list[torch.Generator], prosody=None,
                   prosody_temperature: float = 1.0, prosody_durations: str = "sampled") -> tuple[Tensor, Tensor]:
    """Token ids of ``B`` sentences (one voice) -> normalised mels ``[B, n_mels, T]`` (zero after each row's length)
    and the lengths ``[B]``. ``generators``: one per row. ``prosody``: a one-pass predictor (``drift`` / ``mse``,
    :func:`drifting_tts.fast.graphable`) whose token pitch (and, with ``prosody_durations="sampled"``, durations)
    replace the regressors', as ``ProsodyPredictor.predict`` followed by ``DriftingTTS.synthesize``."""
    dev = next(model.parameters()).device
    lens = [len(x) for x in ids]
    B, n = len(ids), max(lens)
    text = torch.zeros(B, n, dtype=torch.long)
    for b, x in enumerate(ids):
        text[b, : lens[b]] = torch.as_tensor(x)
    text, text_len = text.to(dev), torch.tensor(lens, device=dev)
    spk = torch.full((B,), int(spk), dtype=torch.long, device=dev)
    drift = prosody is not None and prosody.kind == "drift"
    if drift:  # ProsodyPredictor._draw: token noise, then global noise, from each request's generator
        net = prosody.net
        z_tok = torch.zeros(B, net.noise_tok, n, device=dev)
        z_glob = torch.empty(B, net.noise_glob, device=dev)
        for b, g in enumerate(generators):
            z_tok[b, :, : lens[b]] = torch.randn(net.noise_tok, lens[b], device=dev, generator=g)
            z_glob[b] = torch.randn(net.noise_glob, device=dev, generator=g)
    h, mu, logw, x_mask = model.encoder(text, text_len, spk)
    frames = None
    if prosody is None:
        if model.pitch_enabled:
            h, _ = model.pitch_condition(h, x_mask, spk)
    else:
        cond, base = prosody.condition(model, h, x_mask, spk, logw, prosody.stats)
        out = (prosody.net(cond, x_mask, z_tok * prosody_temperature, z_glob * prosody_temperature) if drift
               else prosody.net(cond, x_mask))
        frames, pitch = prosody.frames_and_pitch((base + out[:, :2]) * x_mask, out[:, 2], x_mask, length_scale)
        h, _ = model.pitch_condition(h, x_mask, spk, pitch)
    if frames is None or prosody_durations == "regressor":
        attn, y_len = durations_to_alignment(logw, x_mask, length_scale)
    else:
        attn, y_len = frames_to_alignment(frames, x_mask)
    cond = model.frame_condition(h, mu, attn)
    T, t = cond.shape[-1], y_len.tolist()
    gen = model.generator
    # DriftingTTS.synthesize: the mel noise, then the style codes (rollout); zero after each row's length, as the
    # single-request path pads a partial patch with zeros
    z = torch.zeros(B, model.n_mels, T, device=dev)
    labels = torch.empty(B, max(1, gen.noise_coords), dtype=torch.long, device=dev)
    for b, g in enumerate(generators):
        z[b, :, : t[b]] = torch.randn(model.n_mels, t[b], device=dev, generator=g)
        labels[b] = torch.randint(0, gen.noise_classes, (max(1, gen.noise_coords),), device=dev, generator=g)
    mask = sequence_mask(y_len, T)
    alpha = torch.full((B,), float(cfg_scale), device=dev)
    mel = model.rollout(z * temperature, cond, spk, alpha, gen.num_steps, mask=mask, noise_labels=labels)
    return mel * mask[:, None], y_len


def batchable(synth) -> bool:
    """Whether :func:`synth_mels` supports this Synthesizer's prosody source: none, or a one-pass predictor without
    a duration row of its own (no separate duration temperature, no borrowed rhythm) and no second pitch predictor."""
    from .fast import graphable

    return (graphable(synth.prosody, synth.prosody_spread) and not synth._duration_row()
            and getattr(synth, "prosody_pitch", None) is None)


@torch.no_grad()
def synth_mels(synth, sentences: list[str], speaker, cfg_scale: float, temperature: float,
               generators: list[torch.Generator], length_scale: float = 1.0,
               prosody_temperature: float | None = None) -> tuple[Tensor, Tensor]:
    """:func:`acoustic_batch` with a :class:`~drifting_tts.synthesize.Synthesizer`'s model, prosody source and the
    voice's duration factor: normalised ``sentences`` -> normalised mels ``[B, n_mels, T]`` and lengths ``[B]``."""
    if not batchable(synth):
        raise NotImplementedError("batched synthesis supports no prosody predictor or a one-pass one (drift / mse) "
                                  "without a duration temperature, rhythm or pitch predictor of its own")
    _, tempo = synth._speaker(speaker)
    pt = synth.prosody_temperature if prosody_temperature is None else prosody_temperature
    return acoustic_batch(synth.model, [text_to_ids(s, normalized=True) for s in sentences],
                          synth.speaker_id(speaker), cfg_scale, temperature, length_scale * tempo, generators,
                          synth.prosody, 1.0 if pt is None else pt, synth.prosody_durations)


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


def _take(x: Tensor, starts: Tensor, n: int) -> Tensor:
    """``x[b, ..., starts[b]: starts[b] + n]`` for every row (indices past the end repeat the last element)."""
    idx = (starts[:, None] + torch.arange(n, device=x.device)[None]).clamp_max(x.shape[-1] - 1)
    return x.gather(-1, idx.view(idx.shape[0], *[1] * (x.dim() - 2), n).expand(*x.shape[:-1], n))


@torch.no_grad()
def stream_batched(synth, texts: list[str], speaker=None, cfg_scale: float = 1.0, temperature: float = 1.0,
                   seeds: list[int] | None = None, length_scale: float = 1.0, pause=0.0, first: int = 32,
                   chunk: int = 256, prosody_temperature: float | None = None) -> Iterator[list[tuple[int, Tensor]]]:
    """:meth:`Synthesizer.stream` for many requests at once (one voice). Yields round by round the pieces now on the
    host, ``[(request index, waveform piece), ...]``; each request's pieces join into what ``synth.stream`` gives for
    its text and seed (default seed: its index), up to float rounding.

    A round runs the acoustic model on the next sentence of every request that has used up its mel (all requests in
    the first round), then one masked vocoder batch with the next streaming window of every unfinished request.
    ``pause`` (seconds, a callable or ``"punct"``) is the silence between a request's sentences."""
    from .synthesize import DEFAULT_VOICE, silence

    speaker = DEFAULT_VOICE if speaker is None else speaker
    voc, dev = synth.vocoder, synth.device
    pause = synth._pause(pause, speaker)
    seeds = list(range(len(texts))) if seeds is None else seeds
    reqs = [{"sentences": split_sentences(normalize(t)), "k": 0, "windows": deque(), "rng": random.Random(s),
             "g": torch.Generator(device=dev).manual_seed(s)} for t, s in zip(texts, seeds)]
    mels: list[Tensor] = []
    while True:
        need = [i for i, r in enumerate(reqs) if not r["windows"] and r["k"] < len(r["sentences"])]
        if need:
            mel, lens = synth_mels(synth, [reqs[i]["sentences"][reqs[i]["k"]] for i in need], speaker, cfg_scale,
                                   temperature, [reqs[i]["g"] for i in need], length_scale, prosody_temperature)
            mels.append(synth.stats.denormalize(mel))
            for row, (i, t) in enumerate(zip(need, lens.tolist())):
                r = reqs[i]
                gap = silence(pause, r["sentences"][r["k"] - 1], r["rng"]) if r["k"] else None
                r["windows"].extend((len(mels) - 1, row, *w, k == 0)
                                    for k, w in enumerate(stream_windows(t, first, chunk, voc.context)))
                r["gap"], r["k"] = gap, r["k"] + 1
        active = [i for i, r in enumerate(reqs) if r["windows"]]
        if not active:
            return
        ws = [reqs[i]["windows"].popleft() for i in active]
        width = max(w[3] - w[2] for w in ws)
        parts, order = [], []
        for src in sorted({w[0] for w in ws}):  # one gather per acoustic batch the windows come from
            sel = [j for j, w in enumerate(ws) if w[0] == src]
            rows = torch.tensor([ws[j][1] for j in sel], device=dev)
            starts = torch.tensor([ws[j][2] for j in sel], device=dev)
            parts.append(_take(mels[src][rows], starts, width))
            order += sel
        x = parts[0] if len(parts) == 1 else torch.cat(parts)
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
