"""Prosody measurements (intonation, rhythm, pauses) and opt-in inference-time prosody controls.

Measurements (``drifting-tts prosody`` runs them on held-out recordings and the model's renditions of their texts):

* **audio level** (:func:`prosody_features`): F0 from WORLD harvest every 10 ms (60-500 Hz) in semitones relative to
  the utterance median: std, 5-95% range, skew, excess kurtosis, movement (mean ``|ΔF0|`` between voiced neighbours)
  and micro-variation (mean ``|F0 - 5-frame moving average|``), with the definitions of the Prosody-40 diagnosis
  (docs/EXPERIMENTS.md §5) so the numbers stay comparable; voiced %, level std, internal pauses (energy based) and the
  speaking rate (syllables per second of speech; in Turkish every vowel is one syllable);
* **paired** (:func:`paired_metrics`): against a recording of the same text, dynamic time warping on MFCCs, then the
  log-F0 correlation and RMSE along the path, and the ratio of the speech durations;
* **token level, no audio** (:func:`token_targets`, :func:`token_report`): the duration and pitch predictors against
  the targets of training (MAS durations under the model's own prior ``mu``, :func:`token_pitch` with its
  ``lf0_stats``): per-utterance Pearson r, the flatness ratio (std predicted / std ground truth) and the MAE;
* **seed diversity** (:func:`seed_diversity`): spread of time-normalised F0 contours and of durations over seeds.

Controls: :func:`scale_deviations` (a gain on the pitch or log-duration deviations from the utterance mean),
:func:`voiced_tokens` and :class:`PausePolicy` (silence between sentences by their final punctuation, a ``pause`` of
:class:`drifting_tts.synthesize.Synthesizer`). Inputs are 24 kHz mono waveforms (numpy or CPU tensors).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .audio import SAMPLE_RATE
from .text import BLANK_ID, PAD_ID, SYMBOLS

FRAME_MS = 10.0  # analysis frame period
F0_FLOOR, F0_CEIL = 60.0, 500.0  # harvest search range of the Prosody-40 diagnosis
SILENCE_DB = 35.0  # a frame is silent this far below the utterance's loud level (95th percentile of frame RMS)
MIN_PAUSE_S = 0.1  # shorter silences inside speech are stop closures, not pauses
VOWELS = frozenset("aeıioöuü")
VOICED_LETTERS = VOWELS | frozenset("bcdgğjlmnrvyzw")
# tokens with voiced frames in < 80% of the training targets (studio voice, dio F0): docs/PROSODY.md
UNVOICED_PRONE = frozenset("çpst.!?")
PUNCTUATION = ".,!?"
ST_PER_LN = 12 / math.log(2)  # semitones per natural-log unit


# --- audio level -------------------------------------------------------------------------------------------------


def f0_contour(wav, sr: int = SAMPLE_RATE, method: str = "harvest", frame_period: float = FRAME_MS) -> np.ndarray:
    """WORLD F0 in Hz every ``frame_period`` ms (0: unvoiced); ``harvest`` (default) or ``dio`` + stonemask."""
    import pyworld

    x = np.ascontiguousarray(np.asarray(wav, dtype=np.float64))
    if method == "harvest":
        f0, _ = pyworld.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=frame_period)
    elif method == "dio":
        f0, t = pyworld.dio(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=frame_period)
        f0 = pyworld.stonemask(x, f0, t, sr)
    else:
        raise ValueError(f"unknown F0 method {method!r}")
    return f0


def _frames(x: np.ndarray, sr: int, frame_ms: float = FRAME_MS) -> np.ndarray:
    hop = int(round(sr * frame_ms / 1000))
    n = len(x) // hop
    return x[: n * hop].reshape(n, hop)


def frame_level_db(wav, sr: int = SAMPLE_RATE, frame_ms: float = FRAME_MS) -> np.ndarray:
    """RMS level in dB of consecutive ``frame_ms`` frames."""
    return 10 * np.log10((_frames(np.asarray(wav, dtype=np.float64), sr, frame_ms) ** 2).mean(1) + 1e-12)


def silent_frames(level_db: np.ndarray, below_db: float = SILENCE_DB) -> np.ndarray:
    """Frames ``below_db`` under the loud level (95th percentile), with speech islands of 1-2 frames (clicks, breath
    noise) inside silence closed."""
    if len(level_db) == 0:
        return np.zeros(0, dtype=bool)
    s = level_db < np.percentile(level_db, 95) - below_db
    for k in (1, 2):  # an island of k non-silent frames between silent ones
        for i in range(1, len(s) - k):
            if s[i - 1] and s[i + k] and not s[i: i + k].any():
                s[i: i + k] = True
    return s


def pauses(silent: np.ndarray, min_frames: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Internal pauses: runs ``[start, end)`` of at least ``min_frames`` silent frames between the first and the last
    non-silent frame. Also returns that speech span ``(first, last + 1)`` (``(0, 0)`` without speech)."""
    speech = np.nonzero(~silent)[0]
    if len(speech) == 0:
        return [], (0, 0)
    a, b = int(speech[0]), int(speech[-1]) + 1
    runs, start = [], None
    for i in range(a, b + 1):
        if i < b and silent[i]:
            start = i if start is None else start
        elif start is not None:
            if i - start >= min_frames:
                runs.append((start, i))
            start = None
    return runs, (a, b)


def syllables(text: str) -> int:
    """Syllables of normalised Turkish text: one per vowel."""
    return sum(c in VOWELS for c in text)


def voiced_runs(f0: np.ndarray, min_frames: int = 1) -> list[tuple[int, int]]:
    """Runs ``[start, end)`` of consecutive voiced frames, at least ``min_frames`` long."""
    v = np.concatenate([[False], f0 > 0, [False]])
    edges = np.nonzero(v[1:] != v[:-1])[0]
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_frames]


def pitch_reversals(f0: np.ndarray, smooth: int = 5, dead_zone: float = 0.05, frame_ms: float = FRAME_MS) -> float:
    """Local pitch reversals per voiced second: sign changes of the slope of the F0 contour (semitones, moving average
    over ``smooth`` frames within each voiced run), slopes below ``dead_zone`` semitones per frame keeping the
    previous sign."""
    changes, voiced = 0, 0
    for a, b in voiced_runs(f0, smooth + 2):
        st = np.convolve(12 * np.log2(f0[a:b]), np.ones(smooth) / smooth, mode="valid")
        d = np.diff(st)
        signs = np.sign(d[np.abs(d) >= dead_zone])
        changes += int((signs[1:] != signs[:-1]).sum())
        voiced += b - a
    return changes / max(voiced * frame_ms / 1000, 1e-6) if voiced else float("nan")


def f0_statistics(f0: np.ndarray) -> dict:
    """Intonation statistics of one F0 track (Hz, 0 = unvoiced), in semitones relative to its median; ``f0_cv`` is
    the coefficient of variation in Hz, ``f0_reversals`` :func:`pitch_reversals`."""
    v = f0 > 0
    if v.sum() < 5:
        return {}
    st = 12 * np.log2(f0[v] / np.median(f0[v]))
    move = np.abs(np.diff(12 * np.log2(np.where(v, f0, 1.0))))[v[1:] & v[:-1]]
    smooth = np.convolve(st, np.ones(5) / 5, mode="same")
    c = st - st.mean()
    m2 = max((c**2).mean(), 1e-12)
    return {"f0_median_hz": float(np.median(f0[v])), "f0_std": float(st.std()),
            "f0_range": float(np.percentile(st, 95) - np.percentile(st, 5)),
            "f0_skew": float((c**3).mean() / m2**1.5), "f0_kurt": float((c**4).mean() / m2**2 - 3),
            "f0_move": float(move.mean()) if len(move) else float("nan"),
            "f0_micro": float(np.abs(st - smooth)[2:-2].mean()),
            "f0_cv": float(f0[v].std() / f0[v].mean()), "f0_reversals": pitch_reversals(f0)}


def token_f0(f0: np.ndarray, frames: np.ndarray, pitch_st: np.ndarray | None = None, mask: np.ndarray | None = None,
             hop_s: float = 256 / SAMPLE_RATE, frame_ms: float = FRAME_MS) -> dict:
    """F0 of a waveform measured per token of a known alignment (``frames``: mel frames per token, ``hop_s`` seconds
    each; ``f0`` every ``frame_ms`` ms).

    ``f0_tok_std``: mean std (semitones) of F0 inside the tokens with >= 3 voiced frames; ``f0_within``: the share of
    the utterance's F0 variance that lies inside tokens (1 - between-token share). With ``pitch_st`` (the token pitch
    the audio was conditioned on, semitones, on the tokens of ``mask``): ``render_r``, ``render_flat`` (std of the
    realised / of the conditioning token pitch) and ``render_bias``: how faithfully the acoustic model and vocoder
    render the token pitch they are given."""
    edges = np.concatenate([[0], np.cumsum(frames)]) * hop_s
    t = np.arange(len(f0)) * frame_ms / 1000
    tok = np.searchsorted(edges, t, side="right") - 1  # token of every F0 frame
    v = (f0 > 0) & (tok >= 0) & (tok < len(frames))
    if v.sum() < 10:
        return {}
    st, tok = 12 * np.log2(f0[v] / np.median(f0[v])), tok[v]
    n = np.bincount(tok, minlength=len(frames))
    mean = np.bincount(tok, st, minlength=len(frames)) / np.maximum(n, 1)
    within = ((st - mean[tok]) ** 2).sum() / max(((st - st.mean()) ** 2).sum(), 1e-12)
    sq = np.bincount(tok, (st - mean[tok]) ** 2, minlength=len(frames))
    many = n >= 3
    out = {"f0_within": float(within), "f0_tok_std": float(np.sqrt(sq[many] / n[many]).mean()) if many.any()
           else float("nan")}
    if pitch_st is not None:
        sel = (n >= 2) & (np.ones(len(frames), bool) if mask is None else np.asarray(mask, bool))
        if sel.sum() >= 3:
            ref = 12 * np.log2(np.median(f0[f0 > 0]))
            realised = mean[sel] + ref  # back to absolute semitones (re 1 Hz)
            given = np.asarray(pitch_st)[sel]
            out.update({f"render_{k}": v for k, v in _agreement(realised - realised.mean(),
                                                                  given - given.mean()).items() if k != "mae"})
            out["render_bias"] = float((realised - given).mean())
    return out


def prosody_features(wav, sr: int = SAMPLE_RATE, f0: np.ndarray | None = None, text: str | None = None,
                     min_pause: float = MIN_PAUSE_S) -> dict:
    """Audio-level prosody of one utterance (see the module docstring); ``f0`` as :func:`f0_contour` (computed if
    omitted), ``text`` (normalised) adds the speaking rate. ``pause_lengths`` lists the internal pauses in seconds."""
    x = np.asarray(wav, dtype=np.float64)
    f0 = f0_contour(x, sr) if f0 is None else np.asarray(f0, dtype=np.float64)
    out = f0_statistics(f0)
    out["voiced_pct"] = float(100 * (f0 > 0).mean())
    # the level measures of the Prosody-40 diagnosis: std of |x| per frame in dB, 40 dB below the loudest frame
    e = 20 * np.log10(np.abs(_frames(x, sr)).std(1) + 1e-6)
    out["silence_pct"] = float(100 * (e < e.max() - 40).mean())
    out["energy_std"] = float(e[e > e.max() - 40].std())
    runs, (a, b) = pauses(silent_frames(frame_level_db(x, sr)), int(round(min_pause * 1000 / FRAME_MS)))
    lengths = [(end - start) * FRAME_MS / 1000 for start, end in runs]
    speech = (b - a) * FRAME_MS / 1000
    out.update(duration_s=len(x) / sr, speech_s=speech, pauses=len(lengths), pause_s=float(sum(lengths)),
               pause_lengths=lengths)
    if text and speech > 0:
        n = syllables(text)
        out["rate_sps"] = n / speech
        out["artic_sps"] = n / max(speech - sum(lengths), 1e-3)
    return out


# --- paired: against a recording of the same text ----------------------------------------------------------------


def mfcc(wav, sr: int = SAMPLE_RATE, n: int = 20, frame_ms: float = FRAME_MS) -> np.ndarray:
    """MFCCs 1..n (no energy term), mean-normalised per utterance, every ``frame_ms`` ms: ``[n, frames]``. Frame
    ``k`` is centred on ``k * frame_ms`` like the F0 of :func:`f0_contour`."""
    import librosa

    hop = int(round(sr * frame_ms / 1000))
    m = librosa.feature.mfcc(y=np.asarray(wav, dtype=np.float32), sr=sr, n_mfcc=n + 1, n_fft=1024, hop_length=hop,
                             n_mels=40)[1:]
    return m - m.mean(1, keepdims=True)


def dtw_path(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dynamic-time-warping path ``[L, 2]`` (frame of ``a``, frame of ``b``) between feature sequences ``[d, T]``."""
    import librosa

    _, wp = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    return wp[::-1].copy()


def speech_span(wav, sr: int = SAMPLE_RATE) -> float:
    """Seconds from the first to the last non-silent frame."""
    _, (a, b) = pauses(silent_frames(frame_level_db(wav, sr)), 1)
    return (b - a) * FRAME_MS / 1000


def paired_metrics(ref, gen, sr: int = SAMPLE_RATE, ref_f0: np.ndarray | None = None,
                   gen_f0: np.ndarray | None = None) -> dict:
    """Prosody of ``gen`` against the recording ``ref`` of the same text: MFCC-DTW, then along the path the Pearson
    correlation of log-F0 (``f0_corr``) and its RMSE in semitones with each utterance relative to its own median
    (``f0_rmse``: contour shape, not register), the voicing agreement, the mean MFCC distance, and the speech
    duration ratio gen / ref (``dur_ratio``)."""
    ref, gen = np.asarray(ref, dtype=np.float64), np.asarray(gen, dtype=np.float64)
    ref_f0 = f0_contour(ref, sr) if ref_f0 is None else np.asarray(ref_f0, dtype=np.float64)
    gen_f0 = f0_contour(gen, sr) if gen_f0 is None else np.asarray(gen_f0, dtype=np.float64)
    a, b = mfcc(ref, sr), mfcc(gen, sr)
    path = dtw_path(a, b)
    i, j = path[:, 0], path[:, 1]
    out = {"mfcc_dist": float(np.linalg.norm(a[:, i] - b[:, j], axis=0).mean())}
    fr, fg = ref_f0[np.minimum(i, len(ref_f0) - 1)], gen_f0[np.minimum(j, len(gen_f0) - 1)]
    out["vuv_agree"] = float(((fr > 0) == (fg > 0)).mean())
    both = (fr > 0) & (fg > 0)
    if both.sum() >= 10 and (ref_f0 > 0).any() and (gen_f0 > 0).any():
        lr = 12 * np.log2(fr[both] / np.median(ref_f0[ref_f0 > 0]))
        lg = 12 * np.log2(fg[both] / np.median(gen_f0[gen_f0 > 0]))
        out["f0_corr"] = float(np.corrcoef(lr, lg)[0, 1])
        out["f0_rmse"] = float(np.sqrt(((lr - lg) ** 2).mean()))
    rs = speech_span(ref, sr)
    if rs > 0:
        out["dur_ratio"] = speech_span(gen, sr) / rs
    return out


# --- token level: the prosody predictors against the targets of training ----------------------------------------


@torch.no_grad()
def token_targets(model, ids: Tensor, spk: Tensor, mel: Tensor | None = None, f0: Tensor | None = None) -> dict:
    """Predicted and (with ``mel``) ground-truth prosody of one utterance, computed as in training.

    Args:
        model: a :class:`drifting_tts.models.tts.DriftingTTS` in eval mode.
        ids: token ids ``[1, N]`` (with blanks); spk: ``[1]``.
        mel: the recording's normalised mel ``[1, n_mels, T]``: MAS against the model's prior ``mu`` gives the
            ground-truth durations (``durations_gt``, frames summing to ``T``).
        f0: the recording's F0 on the mel frames ``[1, T]`` (Hz, ``f0.bin``): ground-truth token pitch
            (``pitch_gt``, normalised log-F0 with the model's ``lf0_stats``, 0 on tokens without voiced frames) and
            ``voiced_gt`` (tokens with voiced frames).
    Returns:
        1-D tensors on the model's device: ``logw`` (predicted log-durations, before any length scale), ``pitch``
        (predicted token pitch) and the ground truth as available.
    """
    from .models.text_encoder import align, token_pitch

    h, mu, logw, x_mask = model.encoder(ids, torch.tensor([ids.shape[1]], device=ids.device), spk)
    out = {"ids": ids[0], "logw": logw[0, 0].float()}
    if model.pitch_enabled:
        out["pitch"] = model.pitch_condition(h, x_mask, spk)[1][0, 0].float()
    if mel is None:
        return out
    y_mask = torch.ones(1, 1, mel.shape[-1], device=mel.device)
    attn, logw_gt = align(mu.float(), x_mask, mel.float(), y_mask)
    out["logw_gt"], out["durations_gt"] = logw_gt[0, 0], attn.sum(-1)[0].round().long()
    if f0 is not None and model.pitch_enabled:
        lf0_mean, lf0_std = model.lf0_stats
        out["pitch_gt"] = token_pitch(f0.float(), attn, lf0_mean, lf0_std)[0, 0]
        out["voiced_gt"] = torch.bmm(attn, (f0 > 0).float()[:, :, None])[0, :, 0] > 0
    return out


def frames_from_logw(logw: Tensor, scale: float = 1.0) -> Tensor:
    """Frames per token as synthesis uses them: ``ceil(exp(logw) * scale)`` (see ``durations_to_alignment``)."""
    return torch.ceil(torch.exp(logw) * scale).clamp_min(0)


def char_durations(ids: Tensor, frames: Tensor) -> tuple[list[str], np.ndarray]:
    """Per character of the text: its token plus the blank after it (the leading blank is dropped)."""
    ids, frames = ids.tolist(), frames.float().cpu().numpy()
    chars, dur = [], []
    for k in range(1, len(ids), 2):
        chars.append(SYMBOLS[ids[k]])
        dur.append(frames[k] + (frames[k + 1] if k + 1 < len(ids) else 0.0))
    return chars, np.asarray(dur)


def _agreement(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Pearson r, flatness (std ratio pred / gt) and MAE of one utterance's paired values."""
    out = {"mae": float(np.abs(pred - gt).mean()) if len(gt) else float("nan")}
    if len(gt) >= 3 and gt.std() > 0:
        out["flat"] = float(pred.std() / gt.std())
        out["r"] = float(np.corrcoef(pred, gt)[0, 1]) if pred.std() > 0 else float("nan")
    return out


def token_report(t: dict, frame_rate: float, frames: Tensor | None = None, pitch: Tensor | None = None,
                 lf0_std: float = 1.0, scale: float = 1.0) -> dict:
    """Predictors against the ground truth of :func:`token_targets` for one utterance.

    ``frames`` (default: the predicted durations as synthesised, ``ceil(exp(logw) * scale)``) and ``pitch`` (default:
    the predicted token pitch) may be modified versions, e.g. after a gain. Durations are compared per letter
    (token + following blank) in log-frames: ``dur_r``, ``dur_flat``, ``dur_mae``, their coefficients of variation
    ``dur_cv`` / ``dur_cv_gt`` and the total length ratio ``len_ratio``. Pitch is compared on the tokens with voiced
    frames in semitones (``lf0_std``: the model's log-F0 std): ``pitch_r``, ``pitch_flat``, ``pitch_mae``,
    ``pitch_bias``. ``punct`` maps each punctuation mark to ``[predicted, ground truth]`` seconds of the mark plus
    the space after it (the pause at that position)."""
    out: dict = {}
    if "durations_gt" in t:
        frames = frames_from_logw(t["logw"], scale) if frames is None else frames
        chars, dp = char_durations(t["ids"], frames)
        _, dg = char_durations(t["ids"], t["durations_gt"])
        letter = np.array([c not in PUNCTUATION and c != " " for c in chars])
        lp, lg = np.log(np.maximum(dp[letter], 1)), np.log(np.maximum(dg[letter], 1))
        out.update({f"dur_{k}": v for k, v in _agreement(lp, lg).items()})
        out["dur_cv"] = float(dp[letter].std() / max(dp[letter].mean(), 1e-6))
        out["dur_cv_gt"] = float(dg[letter].std() / max(dg[letter].mean(), 1e-6))
        out["len_ratio"] = float(frames.sum() / t["durations_gt"].sum())
        punct: dict[str, list] = {}
        for k, c in enumerate(chars[:-1]):  # internal marks only: the last one ends the utterance
            if c in PUNCTUATION:
                n = 2 if chars[k + 1] == " " else 1
                punct.setdefault(c, []).append([float(dp[k: k + n].sum() / frame_rate),
                                                float(dg[k: k + n].sum() / frame_rate)])
        out["punct"] = punct
    if "pitch_gt" in t:
        pitch = t["pitch"] if pitch is None else pitch
        v = t["voiced_gt"]
        pp = (pitch[v].float().cpu().numpy() * lf0_std * ST_PER_LN)
        pg = (t["pitch_gt"][v].float().cpu().numpy() * lf0_std * ST_PER_LN)
        out.update({f"pitch_{k}": val for k, val in _agreement(pp, pg).items()})
        out["pitch_bias"] = float((pp - pg).mean()) if len(pg) else float("nan")
    return out


# --- seed diversity ----------------------------------------------------------------------------------------------


def seed_diversity(f0s: list[np.ndarray], lengths: list[float] | None = None,
                   durations: list[Tensor] | None = None, points: int = 200) -> dict:
    """Spread of K renditions of one text (one per seed).

    ``f0_spread``: mean over time of the std across seeds of the F0 contours (semitones relative to the pooled
    median, unvoiced gaps interpolated, time-normalised between the first and last voiced frame to ``points``
    points); ``f0_std_cv``: coefficient of variation of the per-rendition F0 std; ``len_cv``: of ``lengths``
    (e.g. seconds); ``token_dur_std``: mean per-token std of log-frames across seeds (``durations``: ``[N]`` each)."""
    ref = np.median(np.concatenate([f[f > 0] for f in f0s]))
    curves, stds = [], []
    for f in f0s:
        v = np.nonzero(f > 0)[0]
        st = 12 * np.log2(f[v] / ref)
        curves.append(np.interp(np.linspace(v[0], v[-1], points), v, st))
        stds.append(st.std())
    out = {"f0_spread": float(np.stack(curves).std(0).mean()), "f0_std_cv": float(np.std(stds) / np.mean(stds))}
    if lengths is not None:
        out["len_cv"] = float(np.std(lengths) / np.mean(lengths))
    if durations is not None:
        d = torch.stack([x.float().cpu() for x in durations]).clamp_min(1).log()
        out["token_dur_std"] = float(d.std(0, unbiased=False).mean())
    return out


# --- controls ----------------------------------------------------------------------------------------------------


def voiced_tokens(ids: Tensor) -> Tensor:
    """Tokens whose training pitch target is almost always voiced (``[..., N]`` bool): all but the voiceless
    consonants ç p s t, sentence-final marks and the blanks between two tokens that are not voiced letters. Unvoiced
    targets are 0 in training (the global mean), so a pitch gain leaves the excluded tokens alone."""
    voiced = torch.tensor([s in VOICED_LETTERS for s in SYMBOLS], device=ids.device)[ids]
    prone = torch.tensor([s in UNVOICED_PRONE for s in SYMBOLS], device=ids.device)[ids]
    left = torch.nn.functional.pad(voiced, (1, 0))[..., :-1]
    right = torch.nn.functional.pad(voiced, (0, 1))[..., 1:]
    blank = ids == BLANK_ID
    return ~prone & ~(blank & ~left & ~right) & (ids != PAD_ID)


def letter_tokens(ids: Tensor) -> Tensor:
    """Tokens of letters (not blanks, spaces or punctuation): ``[..., N]`` bool."""
    table = torch.tensor([len(s) == 1 and s.isalpha() for s in SYMBOLS], device=ids.device)
    return table[ids]


def scale_deviations(x: Tensor, gain: float, mask: Tensor, keep_total: bool = False) -> Tensor:
    """``mean + gain * (x - mean)`` on the ``mask`` ed entries (mean over them, per row); others unchanged.

    ``keep_total`` (for log-durations): then shift the masked entries so that ``sum(exp(x))`` over them is
    unchanged, i.e. the gain changes the rhythm but not the length."""
    m = mask.to(x.dtype)
    mean = (x * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1)
    y = torch.where(mask, mean + gain * (x - mean), x)
    if keep_total:
        shift = torch.log((x.exp() * m).sum(-1, keepdim=True) / (y.exp() * m).sum(-1, keepdim=True).clamp_min(1e-8))
        y = torch.where(mask, y + shift, y)
    return y


# Pauses at internal marks in the training data (energy-based silence at the mark: mean, std in seconds) and the
# silence the model already leaves at the edges of a generated sentence (``edge``: leading + trailing, T 0.3, α 2,
# vocos-ft). Measured with scripts/pause_stats.py (1000 utterances per group); see docs/PROSODY.md.
# Marks seen fewer than 20 times are left out (they fall back to ".").
PAUSES: dict[int | str, dict] = {
    722: {"edge": 0.161, ".": (0.125, 0.093), "?": (0.103, 0.064), ",": (0.052, 0.053)},  # studio
    389: {"edge": 0.176, ".": (0.848, 0.348), ",": (0.473, 0.208)},  # male
    323: {"edge": 0.070, ".": (0.223, 0.328), ",": (0.126, 0.174)},  # female
    "base": {"edge": 0.123, ".": (0.234, 0.329), "?": (0.238, 0.326), ",": (0.159, 0.190)},  # corpus without 722
}


@dataclass
class PausePolicy:
    """Silence between sentences by the first one's final mark: a ``pause`` of ``Synthesizer``.

    The gap between two generated sentences is the trailing silence of the first, the inserted silence and the
    leading silence of the second. The policy aims at a gap like the training data's pause at that mark (``gaps``:
    mark -> ``(mean, std)`` seconds; ``jitter`` 0 gives the mean, ``j > 0`` draws ``mean + j * std * N(0, 1)``) and
    inserts ``max(0, gap - edge)``, ``edge`` being the sentences' own edge silence. A sentence cut without a mark (a
    long one split at a space) uses the ``,`` entry; a mark missing from ``gaps`` uses ``.``."""

    gaps: dict[str, tuple[float, float]] = field(default_factory=lambda: {".": (0.15, 0.0)})
    edge: float = 0.0
    jitter: float = 0.0

    @classmethod
    def for_voice(cls, speaker: int | str | None = None, jitter: float = 0.0) -> PausePolicy:
        """The measured policy of a speaker ID (else of the multi-speaker corpus, ``base``)."""
        table = dict(PAUSES.get(int(speaker) if str(speaker).isdigit() else speaker, PAUSES["base"]))
        return cls(table, table.pop("edge"), jitter)

    def __call__(self, sentence: str, rng: random.Random | None = None) -> float:
        mark = sentence.rstrip()[-1:]
        mean, std = self.gaps.get(mark if mark in PUNCTUATION else ",", self.gaps["."])
        gap = mean + self.jitter * std * rng.gauss(0.0, 1.0) if self.jitter and rng is not None else mean
        return float(max(0.0, gap - self.edge))
