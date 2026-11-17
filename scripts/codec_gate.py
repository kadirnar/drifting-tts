"""Codec gate (#91): round trip, tolerance to latent error and phonetic structure of every audio representation.

The three questions a representation must pass before a TTS model is trained on it, answered by one script for the
mel + vocoder pairs and the audio VAEs of :mod:`drifting_tts.latents`.

1. **Round trip** (encode -> decode of held-out recordings), from the 24 kHz recordings (``audio.bin``) and, for the
   backends that take or return more than 24 kHz, from the hi-res recordings (``audio_hr.bin`` of
   ``prepare --save-audio --hires-rate 48000``; skipped with a note when the root has none). Per round trip: WER / CER
   (Whisper, the data root's language), UTMOSv2, speaker similarity with the recording, PESQ (wide band) and STOI
   (when ``pesq`` / ``pystoi`` are installed), the log-mel L1 distance at 24 kHz and the log-spectral distance in the
   bands 8-12 kHz and 12-20 kHz (:func:`drifting_tts.metrics.band_lsd`; the judges work at 16 kHz and see neither).
2. **Tolerance to latent error**: the recordings' own frames are perturbed in normalised units (per-channel mean and
   standard deviation over the selected utterances) and decoded: Gaussian noise of 0.05-0.5 standard deviations,
   shrinking towards the mean by 0.7-0.9, temporal smoothing (1/4 1/2 1/4). UTMOSv2 and WER for each, and one number
   per backend: the noise level that costs 0.2 UTMOSv2 (linear interpolation along the curve). This is the table of
   docs/MIMI.md (0.2 standard deviations cost Mimi 0.67 UTMOSv2, smoothing 1.3), for every backend.
3. **Phonetic structure of the L2 distance**, which the drifting kernel relies on. Character segments come from MMS
   forced alignment (``torchaudio.pipelines.MMS_FA``); a segment is the mean of its normalised frames. **ABX error**:
   triplets A, X of one character and B of another, all of one speaker; an error is ``d(A, X) >= d(B, X)`` under L2
   (ties count half), averaged over (speaker, character pair) cells. **Linear probe**: multinomial logistic
   regression from one frame to its character, trained and tested on different utterances, next to the majority-class
   baseline. Both are character-level and context-free, on this repository's own recordings: they compare the rows
   of this table with each other, and are **not** ZeroSpeech ABX numbers (triphone contexts, DTW over frames) nor
   comparable with the ~8% Kyutai quote for Mimi. Characters are a fair phoneme proxy for Turkish and a poor one
   for English. A 12.5 Hz frame (Mimi) is longer than most characters, so its segments also hold their neighbours.
4. **Repeated latents** (``--model``, a DiT or an ARDrift model trained on latents; for ARDrift pass ``--cfg 1
   --temperature 0.5``): the model's generated frames are split into the groups of ``latent_repeat`` copies that
   ``LatentVocoder`` averages; reported per position in the group: the mean squared deviation from the group mean,
   relative to the per-channel variance. The recordings' stored latents give exactly 0. This is the measurement #94
   (crops off the frame grid) asks for. With the judges, the generated latents are also decoded twice, from the mean
   of the copies (what ``LatentVocoder`` does) and from the first copy alone (#92): if one copy sounds better than
   their mean, the averaging hides a head that cannot make four values agree.

``results.json`` and ``results.md`` (one row per backend) go to ``--out``. ``--max-roundtrip-loss`` and
``--max-noise-loss`` add a pass / fail column: UTMOSv2 below the recording after a 24 kHz round trip, and UTMOSv2 lost
to noise of 0.2 standard deviations. Without them there is no verdict (#91 fixes the thresholds after the first table).

    python scripts/codec_gate.py --data data/train --out outputs/codec_gate
    python scripts/codec_gate.py --data data/train --backends voxcpm2 --skip tolerance phonetic \
        --model runs/tts_voxcpm2/model_ema.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor

from drifting_tts.audio import FRAME_RATE, HOP_LENGTH, N_MELS, SAMPLE_RATE, BigVGANLogMel, make_logmel, resample
from drifting_tts.evaluate import _plain, summarize
from drifting_tts.latents import VAE_BACKENDS, AudioBackend, LatentStats, load_backend
from drifting_tts.metrics import band_lsd, bootstrap_ci, error_counts

BACKENDS = ("vocos", "bigvgan", "dacvae", "voxcpm2", "voxcpm1.5", "mimi", "semantic-vae")
MEL_VOCODERS = {"vocos": "vocos-v2", "bigvgan": "bigvgan-v2-ft"}  # the vocoders of releases v3.2 and v3.1
PARTS = ("native", "tolerance", "phonetic", "repeat")
NOISE, SCALE = (0.05, 0.1, 0.2, 0.3, 0.5), (0.7, 0.8, 0.9)
BANDS = {"lsd_8_12": (8000.0, 12000.0), "lsd_12_20": (12000.0, 20000.0)}
GATE_NOISE, GATE_LOSS = 0.2, 0.2  # the gate's noise level (standard deviations) and the loss the tolerance is read at
TR = str.maketrans({"ç": "c", "ş": "s", "ğ": "g", "ı": "i", "ö": "o", "ü": "u", "â": "a", "î": "i", "û": "u"})
Spans = list[tuple[str, float, float]]  # (character, start, end) in seconds


# ---------------------------------------------------------------------------------------------- recordings, backends


class Recordings:
    """The first ``num`` utterances of ``split`` (from ``offset``) with audio: 24 kHz waveforms, the hi-res ones
    when the root has ``audio_hr.bin``, the normalised texts and the speakers."""

    def __init__(self, root: str | Path, split: str = "val", offset: int = 0, num: int = 100):
        root = Path(root)
        stats = json.loads((root / "stats.json").read_text())
        self.language = stats.get("language", "tr")
        if not (root / "audio.bin").exists():
            raise FileNotFoundError(f"{root / 'audio.bin'} missing: run `drifting-tts prepare --save-audio`")
        entries = [e for e in map(json.loads, open(root / "index.jsonl"))]
        self.items = [e for e in entries if e["split"] == split and "audio_offset" in e][offset: offset + num]
        if not self.items:
            raise SystemExit(f"no utterances with audio in the '{split}' split of {root} from offset {offset}")
        audio = np.memmap(root / "audio.bin", dtype=np.float16, mode="r")
        self.wavs = [self._read(audio, e["audio_offset"], e["audio_samples"]) for e in self.items]
        self.hr_rate, self.hires = None, None
        if (root / "audio_hr.bin").exists() and stats.get("hr_rate") and all("hr_offset" in e for e in self.items):
            hr = np.memmap(root / "audio_hr.bin", dtype=np.float16, mode="r")
            self.hr_rate = int(stats["hr_rate"])
            self.hires = [self._read(hr, e["hr_offset"], e["hr_samples"]) for e in self.items]
        self.texts = [e["norm_text"] for e in self.items]
        self.refs = [_plain(t, self.language) for t in self.texts]
        self.speakers = [e.get("spk_id", 0) for e in self.items]
        self.seconds = sum(len(w) for w in self.wavs) / SAMPLE_RATE

    @staticmethod
    def _read(audio: np.ndarray, offset: int, samples: int) -> Tensor:
        return torch.from_numpy(np.array(audio[offset: offset + samples], dtype=np.float32))

    def __len__(self) -> int:
        return len(self.items)


class MelVocoder(AudioBackend):
    """A log-mel front end and its vocoder as an audio backend: ``encode`` is the mel the vocoder expects (BigVGAN's
    or Vocos's own), ``decode`` the vocoder (a registry name of :data:`drifting_tts.vocoder.VOCODERS` or a fine-tuned
    checkpoint). 24 kHz vocoders only. The vocoder makes its own phase, so the output is not shifted to the input."""

    input_rate = output_rate = SAMPLE_RATE
    frame_rate, dim = FRAME_RATE, N_MELS
    waveform_matching = False

    def __init__(self, spec: str, device: str = "cuda", name: str | None = None):
        from drifting_tts.vocoder import load_vocoder

        self.vocoder = load_vocoder(spec, device)
        self.name, self.spec, self.device = name or spec, spec, device
        self.mel = make_logmel(self.vocoder.mel).to(device)

    def num_frames(self, samples: int, sr: int) -> int:
        return -(-samples * SAMPLE_RATE // sr) // HOP_LENGTH + (self.vocoder.mel == "vocos")  # Vocos: centred STFT

    def encode(self, wav: Tensor, sr: int) -> Tensor:
        return self.mel(self._input(wav, sr)[:, 0])

    def decode(self, latent: Tensor) -> Tensor:
        return self.vocoder(latent)


def build_backend(name: str, args) -> AudioBackend:
    """``vocos`` / ``bigvgan`` (the release vocoders, or ``--vocos`` / ``--vocoder``), ``mel:<vocoder>`` or an audio
    VAE of :mod:`drifting_tts.latents`."""
    if name.startswith("mel:"):
        return MelVocoder(name[4:], args.device, name)
    if name in MEL_VOCODERS:
        spec = (args.vocos if name == "vocos" else args.vocoder) or MEL_VOCODERS[name]
        return MelVocoder(spec, args.device, name)
    if name in VAE_BACKENDS:
        return load_backend(name, args.device)
    raise SystemExit(f"unknown backend {name!r}: choose from {', '.join(BACKENDS)} or mel:<vocoder>")


# ------------------------------------------------------------------------------------------------------ round trip


def lag(ref: Tensor, out: Tensor, max_lag: int) -> int:
    """Lag (samples) that maximises the cross-correlation of ``out`` with ``ref``; positive: ``out`` is late."""
    n = min(len(ref), len(out))
    f = torch.fft.rfft(ref[:n].double(), 2 * n).conj() * torch.fft.rfft(out[:n].double(), 2 * n)
    cc = torch.fft.irfft(f, 2 * n)
    return int(torch.cat([cc[-max_lag:], cc[: max_lag + 1]]).argmax().item()) - max_lag


def fit(wav: Tensor, n: int, shift: int = 0) -> Tensor:
    """``wav`` moved ``shift`` samples earlier (negative: later), then cut or zero-padded to ``n`` samples."""
    wav = wav[shift:] if shift >= 0 else F.pad(wav, (-shift, 0))
    return F.pad(wav, (0, max(0, n - len(wav))))[:n]


@torch.no_grad()
def decode(backend: AudioBackend, z: Tensor, n: int, shift: int = 0) -> Tensor:
    """Frames ``[dim, T]`` -> ``n`` samples on the CPU, moved ``shift`` samples earlier."""
    return fit(backend.decode(z[None].to(backend.device))[0].float().cpu(), n, shift)


@torch.no_grad()
def roundtrip(backend: AudioBackend, wavs: Sequence[Tensor], sr: int) -> tuple[list[Tensor], list[Tensor], list[int]]:
    """Encode and decode every recording (``[samples]`` at ``sr``): the outputs at ``backend.output_rate``, shifted
    by their measured lag (within 10 ms; not for a vocoder, which makes its own phase) and cut to the input's
    duration, the frames ``[dim, T]`` and the lags."""
    rate, match = backend.output_rate, getattr(backend, "waveform_matching", True)
    outs, latents, lags = [], [], []
    for w in wavs:
        z = backend.encode(w, sr)[0].float().cpu()
        y = backend.decode(z[None].to(backend.device))[0].float().cpu()
        k = lag(resample(w, sr, rate), y, max(1, rate // 100)) if match else 0
        outs.append(fit(y, -(-len(w) * rate // sr), k))
        latents.append(z)
        lags.append(k)
    return outs, latents, lags


def to16k(wav, sr: int) -> np.ndarray:
    """Judge input: 16 kHz with torchaudio's default filter, as ``evaluate`` and ``benchmark`` feed the judges (the
    sharp :func:`drifting_tts.audio.resample` is for the audio paths)."""
    import torchaudio

    x = torch.as_tensor(wav).float()
    return (x if sr == 16_000 else torchaudio.functional.resample(x, sr, 16_000)).numpy()


def optional_metrics() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """PESQ (wide band) and STOI on 16 kHz audio, those whose package is installed."""
    out = {}
    try:
        from pesq import pesq

        out["pesq"] = lambda r, e: float(pesq(16_000, r, e, "wb"))
    except ImportError:
        print("[codec_gate] no PESQ column (pip install pesq)", flush=True)
    try:
        from pystoi import stoi

        out["stoi"] = lambda r, e: float(stoi(r, e, 16_000, extended=False))
    except ImportError:
        print("[codec_gate] no STOI column (pip install pystoi)", flush=True)
    return out


def sample_metrics(ref: Tensor, ref_sr: int, est: Tensor, est_sr: int, logmel: torch.nn.Module,
                   extra: dict[str, Callable] | None = None) -> dict[str, float | None]:
    """Signal-level distances of one output to its recording: ``mel_l1`` (mean absolute log-mel difference at
    24 kHz, in nepers), the band log-spectral distances of :data:`BANDS` at the lower of the two rates (``None`` for a
    band above its Nyquist frequency), and the ``extra`` metrics (:func:`optional_metrics`) at 16 kHz (``None`` when
    one fails, e.g. PESQ on a clip without speech)."""
    a, b = logmel(resample(ref, ref_sr, SAMPLE_RATE)), logmel(resample(est, est_sr, SAMPLE_RATE))
    t = min(a.shape[-1], b.shape[-1])
    out: dict[str, float | None] = {"mel_l1": float((a[..., :t] - b[..., :t]).abs().mean())}
    sr = min(ref_sr, est_sr)
    r, e = resample(ref, ref_sr, sr), resample(est, est_sr, sr)
    for key, (lo, hi) in BANDS.items():
        out[key] = band_lsd(r, e, sr, lo, hi) if hi <= sr / 2 else None
    if extra:
        r16, e16 = to16k(ref, ref_sr), to16k(est, est_sr)
        n = min(len(r16), len(e16))
        for key, fn in extra.items():
            try:
                out[key] = fn(r16[:n], e16[:n])
            except Exception:  # noqa: BLE001 (pesq raises its own errors on clips it cannot score)
                out[key] = None
    return out


def mean_metrics(rows: list[dict]) -> dict[str, float | None]:
    """The mean of every key over the rows that have it."""
    out = {}
    for key in dict.fromkeys(k for r in rows for k in r):
        v = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(v)) if v else None
    return out


# -------------------------------------------------------------------------------------------- tolerance to latent error


def active_channels(stats: LatentStats) -> Tensor:
    """Channels that carry signal. Some VAE channels are unused: constant (std ~ 0), or, for a codec with a
    posterior, as wide as the prior: their spread over the data is below the posterior's own standard deviation
    (``stats.posterior``, :func:`posterior_spread`), so they hold its noise and nothing of the audio."""
    active = stats.std > 0.01 * stats.std.median()
    post = getattr(stats, "posterior", None)
    return active if post is None else active & (stats.std > post)


@torch.no_grad()
def posterior_spread(backend: AudioBackend, wavs: Sequence[Tensor], sr: int) -> Tensor | None:
    """The posterior's mean standard deviation per channel ``[dim]`` over the recordings, or ``None`` for a backend
    without a posterior."""
    if not hasattr(backend, "posterior"):
        return None
    total, frames = None, 0
    for w in wavs:
        std = backend.posterior(w.to(backend.device), sr)[1][0].double().cpu()
        total, frames = (std.sum(1) if total is None else total + std.sum(1)), frames + std.shape[1]
    return (total / frames).float()


def normalize(z: Tensor, stats: LatentStats) -> Tensor:
    """``[dim, T]`` frames in per-channel standard deviations; unused channels are 0."""
    act = active_channels(stats)
    return torch.where(act[:, None], (z - stats.mean[:, None]) / stats.std.clamp_min(1e-12)[:, None], 0.0)


def perturb(z: Tensor, stats: LatentStats, kind: str, value: float = 0.0, eps: Tensor | None = None) -> Tensor:
    """A perturbation of frames ``[dim, T]``, in normalised units: ``noise`` adds ``value`` standard deviations of
    Gaussian noise (``eps``: the standard-normal draw, so that every level shares it), ``scale`` shrinks the frames
    towards the channel means by ``value``, ``smooth`` filters them in time with (1/4, 1/2, 1/4)."""
    mean, std = stats.mean[:, None], stats.std[:, None]
    if kind == "noise":
        return z + value * std * (torch.randn_like(z) if eps is None else eps)
    if kind == "scale":
        return mean + value * (z - mean)
    if kind == "smooth":
        x = F.pad(z[None], (1, 1), mode="replicate")[0]
        return 0.25 * x[:, :-2] + 0.5 * x[:, 1:-1] + 0.25 * x[:, 2:]
    raise ValueError(f"unknown perturbation {kind!r}")


def perturbations(noise: Sequence[float] = NOISE, scale: Sequence[float] = SCALE) -> list[tuple[str, str, float]]:
    """``(condition name, kind, value)`` of the tolerance curve."""
    return ([(f"noise{v:g}", "noise", v) for v in noise] + [(f"scale{v:g}", "scale", v) for v in scale]
            + [("smooth", "smooth", 0.0)])


def loss_curve(levels: Sequence[float], scores: Sequence[float], clean: float) -> tuple[list[float], list[float]]:
    """The points ``(level, clean - score)`` in increasing level, from ``(0, 0)``."""
    pts = sorted(zip(levels, scores))
    return [0.0] + [x for x, _ in pts], [0.0] + [clean - s for _, s in pts]


def level_for_loss(levels: Sequence[float], scores: Sequence[float], clean: float, loss: float = GATE_LOSS
                   ) -> float | None:
    """The smallest perturbation level at which the score is ``loss`` below ``clean``, by linear interpolation
    between the measured levels; ``None`` if the largest level costs less."""
    xs, ys = loss_curve(levels, scores, clean)
    for (x0, y0), (x1, y1) in itertools.pairwise(zip(xs, ys)):
        if y0 < loss <= y1:
            return x0 + (loss - y0) * (x1 - x0) / (y1 - y0)
    return None


def loss_at(levels: Sequence[float], scores: Sequence[float], clean: float, level: float = GATE_NOISE) -> float:
    """The score lost at ``level`` (linear interpolation between the measured levels, held past the largest)."""
    xs, ys = loss_curve(levels, scores, clean)
    return float(np.interp(level, xs, ys))


# ----------------------------------------------------------------------------- phonetic structure of the L2 distance


class MmsAligner:
    """Character spans by MMS forced alignment (``torchaudio.pipelines.MMS_FA``): the letters of the normalised text,
    romanised for the aligner (ç -> c, ...), labelled with the original character."""

    def __init__(self, device: str = "cuda", min_score: float = 0.0):
        import torchaudio

        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False).to(device).eval()
        self.dictionary = {c: i for i, c in enumerate(bundle.get_labels(star=None))}
        self.device, self.min_score = device, min_score

    @torch.no_grad()
    def __call__(self, wav16: Tensor, text: str) -> Spans | None:
        import torchaudio

        chars = [(c, c.translate(TR)) for c in unicodedata.normalize("NFC", text.lower()) if c.isalpha()]
        chars = [(c, r) for c, r in chars if r in self.dictionary]
        if not chars:
            return None
        emission, _ = self.model(wav16[None].to(self.device))
        tokens = torch.tensor([[self.dictionary[r] for _, r in chars]], device=self.device)
        try:
            path, scores = torchaudio.functional.forced_align(emission, tokens, blank=0)
        except RuntimeError:  # fewer frames than the text needs
            return None
        spans = torchaudio.functional.merge_tokens(path[0], scores[0].exp())
        sec = wav16.shape[0] / emission.shape[1] / 16_000
        return [(c, s.start * sec, s.end * sec) for (c, _), s in zip(chars, spans) if s.score >= self.min_score]


def span_frames(spans: Spans, frames: int, rate: float) -> list[tuple[str, int, int]]:
    """``(character, first frame, end frame)`` of every span: the frames whose centre ``(t + 1/2) / rate`` lies in the
    span, or the one frame holding the span's midpoint when it is shorter than a frame."""
    out = []
    for c, start, end in spans:
        a, b = math.ceil(start * rate - 0.5), math.ceil(end * rate - 0.5)
        if b <= a:
            a = int((start + end) / 2 * rate)
            b = a + 1
        a, b = max(a, 0), min(b, frames)
        if b > a:
            out.append((c, a, b))
    return out


def segment_embeddings(z: Tensor, spans: Spans, rate: float) -> tuple[Tensor, list[str]]:
    """Normalised frames ``[dim, T]`` -> one embedding per character span (the mean of its frames) ``[n, dim]`` and
    the characters."""
    seg = span_frames(spans, z.shape[-1], rate)
    emb = torch.stack([z[:, a:b].mean(-1) for _, a, b in seg]) if seg else z.new_zeros(0, z.shape[0])
    return emb, [c for c, _, _ in seg]


def frame_labels(z: Tensor, spans: Spans, rate: float) -> tuple[Tensor, list[str]]:
    """The frames inside a character span ``[n, dim]`` and their characters (frames between spans are left out)."""
    x, y = [], []
    for c, a, b in span_frames(spans, z.shape[-1], rate):
        x.append(z[:, a:b].T)
        y += [c] * (b - a)
    return (torch.cat(x) if x else z.new_zeros(0, z.shape[0])), y


def abx_error(emb: Tensor, labels: Sequence, groups: Sequence, per_pair: int = 50, seed: int = 0) -> dict | None:
    """ABX error of embeddings ``[n, dim]`` under L2. For every group (speaker) and ordered pair of labels ``(a, b)``
    with at least two segments of ``a`` and one of ``b``, ``per_pair`` random triplets A, X of ``a`` (two different
    segments) and B of ``b``; an error is ``d(A, X) > d(B, X)``, a tie half of one. Returns the mean over the
    (group, a, b) cells, its 95% bootstrap interval over cells, and the counts; ``None`` without any cell."""
    rng = np.random.default_rng(seed)
    emb = emb.double()
    by_group: dict = defaultdict(lambda: defaultdict(list))
    for i, (lab, g) in enumerate(zip(labels, groups)):
        by_group[g][lab].append(i)
    cells = []
    for g in sorted(by_group, key=str):
        idx = {lab: torch.tensor(v) for lab, v in sorted(by_group[g].items(), key=lambda kv: str(kv[0]))}
        for a, ia in idx.items():
            if len(ia) < 2:
                continue
            for b, ib in idx.items():
                if a == b:
                    continue
                ai = rng.integers(0, len(ia), per_pair)
                xi = (ai + rng.integers(1, len(ia), per_pair)) % len(ia)  # another segment of a
                bi = rng.integers(0, len(ib), per_pair)
                x = emb[ia[xi]]
                dax, dbx = (emb[ia[ai]] - x).norm(dim=1), (emb[ib[bi]] - x).norm(dim=1)
                cells.append(float(((dax > dbx).double() + 0.5 * (dax == dbx).double()).mean()))
    if not cells:
        return None
    return {"error": float(np.mean(cells)), "ci": list(bootstrap_ci(cells)), "cells": len(cells),
            "triplets": len(cells) * per_pair}


def linear_probe(x_train: Tensor, y_train: Sequence, x_test: Tensor, y_test: Sequence, steps: int = 200,
                 weight_decay: float = 1e-4, seed: int = 0) -> dict | None:
    """Multinomial logistic regression from one frame ``[n, dim]`` to its label (L-BFGS on the training frames,
    features standardised by them): accuracy on the test frames next to the majority-class baseline (the most
    frequent training label). ``None`` when either side is empty."""
    if not len(y_train) or not len(y_test):
        return None
    classes = sorted(set(y_train), key=str)
    index = {c: i for i, c in enumerate(classes)}
    known = [i for i, c in enumerate(y_test) if c in index]  # a test label never seen in training counts as an error
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    xt, xs = ((x.double() - mean) / std for x in (x_train, x_test))
    yt = torch.tensor([index[c] for c in y_train])
    torch.manual_seed(seed)
    w = torch.zeros(xt.shape[1], len(classes), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(len(classes), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xt @ w + b, yt) + weight_decay * w.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    pred = (xs @ w + b).argmax(1).tolist()
    hits = sum(pred[i] == index[y_test[i]] for i in known)
    major = Counter(y_train).most_common(1)[0][0]
    return {"accuracy": hits / len(y_test), "baseline": sum(c == major for c in y_test) / len(y_test),
            "classes": len(classes), "train_frames": len(y_train), "test_frames": len(y_test)}


def phonetic(latents: Sequence[Tensor], spans: Sequence[Spans | None], speakers: Sequence, rate: float,
             per_pair: int = 50, steps: int = 200, seed: int = 0) -> dict:
    """ABX error and linear probe of normalised frames (``[dim, T]`` per utterance) on the character spans; every
    fifth utterance is the probe's test set."""
    emb, labels, groups = [], [], []
    train, test = ([], []), ([], [])
    for k, (z, sp, spk) in enumerate(zip(latents, spans, speakers)):
        if not sp:
            continue
        e, lab = segment_embeddings(z, sp, rate)
        emb.append(e)
        labels += lab
        groups += [spk] * len(lab)
        x, y = frame_labels(z, sp, rate)
        side = test if k % 5 == 4 else train
        side[0].append(x)
        side[1].extend(y)
    if not emb:
        return {"abx": None, "probe": None, "segments": 0}
    probe = linear_probe(torch.cat(train[0]), train[1], torch.cat(test[0]), test[1], steps, seed=seed) \
        if train[0] and test[0] else None
    return {"abx": abx_error(torch.cat(emb), labels, groups, per_pair, seed), "probe": probe, "segments": len(labels)}


# ----------------------------------------------------------------------------------------------- repeated latents


def repeat_deviation(latents: Sequence[Tensor], repeat: int) -> dict:
    """How far the ``repeat`` copies of one codec frame differ. ``latents``: ``[dim, T]`` per utterance, in the
    data's normalised units. Per position in the group: the mean squared deviation from the group mean, relative to
    the channel's variance over all frames, averaged over channels; ``share``: the same over all positions (the part
    of the variance that averaging the copies throws away). Exactly 0 for stored latents (``extract-latents``)."""
    num = total = sq = None
    groups = frames = 0
    for x in latents:
        x = x.double()
        n = x.shape[-1] // repeat
        g = x[:, : n * repeat].reshape(x.shape[0], n, repeat)
        dev = (g - g.mean(-1, keepdim=True)).pow(2).sum(1)  # [dim, repeat]
        num = dev if num is None else num + dev
        total = x.sum(-1) if total is None else total + x.sum(-1)
        sq = x.pow(2).sum(-1) if sq is None else sq + x.pow(2).sum(-1)
        groups, frames = groups + n, frames + x.shape[-1]
    if not groups:
        return {"positions": [], "share": None, "groups": 0}
    var = (sq / frames - (total / frames) ** 2).clamp_min(1e-12)
    rel = num / groups / var[:, None]
    return {"positions": rel.mean(0).tolist(), "share": float(rel.mean()), "groups": groups}


@torch.no_grad()
def generated_latents(args) -> tuple[list[Tensor], list[Tensor], int, dict]:
    """The latents a model trained on latents generates for held-out texts (as ``scripts/latent_smoothness.py``:
    predicted durations with the voice's duration factor, seed = index), the recordings' stored latents, the
    data's ``latent_repeat`` and what decoding them needs (the data's backend and statistics, the texts)."""
    from drifting_tts.data import MelDataset
    from drifting_tts.train import load_tts

    model, cfg, _ = load_tts(args.model, args.device)
    model.eval()
    ds = MelDataset(args.model_data or cfg.data.root, args.split, min_frames=1, max_frames=10**9)
    items = [i for i in range(len(ds)) if args.speaker is None or ds.items[i]["spk_id"] == args.speaker]
    items = items[: args.model_num]
    if not items:
        raise SystemExit(f"--model: no {args.split} utterances" + ("" if args.speaker is None
                                                                    else f" of speaker {args.speaker}"))
    gen, rec = [], []
    for k, i in enumerate(items):
        it = ds[i]
        spk = int(it["spk"])
        text = it["text"][None].to(args.device)
        g = torch.Generator(device=args.device).manual_seed(k)
        x, n = model.synthesize(text, torch.tensor([text.shape[1]], device=args.device),
                                torch.tensor([spk], device=args.device), cfg_scale=args.cfg,
                                temperature=args.temperature, generator=g,
                                length_scale=model.duration_scales.get(spk, model.duration_scale))
        gen.append(x[0, :, : int(n[0])].float().cpu())
        rec.append(it["mel"])
    info = {"backend": ds.backend, "stats": ds.stats, "language": ds.language,
            "refs": [_plain(ds.items[i]["norm_text"], ds.language) for i in items]}
    return gen, rec, ds.latent_repeat, info


@torch.no_grad()
def decode_copies(gen: Sequence[Tensor], repeat: int, info: dict, device: str) -> dict[str, list[np.ndarray]]:
    """The generated latents (normalised, ``[dim, T]``) decoded from the mean of every group of ``repeat`` copies
    and from its first copy alone: 16 kHz audio under ``mean`` and ``copy0``."""
    from drifting_tts.latents.vocoder import LatentVocoder, native_frames

    voc = LatentVocoder(info["backend"], device, repeat=1, sample_rate=16_000)
    out: dict[str, list[np.ndarray]] = {"mean": [], "copy0": []}
    for x in gen:
        z = info["stats"].denormalize(x[None].float())
        out["mean"].append(voc(native_frames(z, repeat))[0].cpu().numpy())
        out["copy0"].append(voc(z[..., ::repeat])[0].cpu().numpy())
    return out


# ------------------------------------------------------------------------------------------------------- judges


class JudgeFactory:
    """The judges, built one at a time (:func:`judge` frees each before the next). A judge whose package is missing
    is skipped with a message."""

    def __init__(self, device: str = "cuda", asr: str = "large-v3", asr_batch: int = 16):
        self.device, self.asr_model, self.asr_batch = device, asr, asr_batch

    def _build(self, what: str, factory: Callable):
        try:
            return factory()
        except ImportError as e:
            print(f"[codec_gate] no {what} ({e})", flush=True)
            return None

    def asr(self, language: str) -> Callable[[list[np.ndarray]], list[str]] | None:
        def build():
            from drifting_tts.score import AsrScorer

            return AsrScorer(self.asr_model, self.device, beam_size=5, batch_size=self.asr_batch,
                             language=language).transcribe

        return self._build("WER (Whisper)", build)

    def sv(self) -> Callable[[np.ndarray], Tensor] | None:
        def build():
            from drifting_tts.judges import SpeakerEmbedder

            return SpeakerEmbedder("wavlm-large-ecapa", self.device)

        return self._build("speaker similarity", build)

    def mos(self) -> Callable[[np.ndarray], float] | None:
        def build():
            from drifting_tts.judges import UTMOSv2

            return UTMOSv2(self.device)

        return self._build("UTMOSv2", build)


def mean_ci(v: Sequence[float]) -> dict:
    return {"mean": float(np.mean(v)), "ci": list(bootstrap_ci(v))}


def judge(audio: dict[str, list[np.ndarray]], refs: Sequence[str], language: str, factory,
          sim_keys: Sequence[str] = ()) -> dict[str, dict]:
    """Every condition of ``audio`` (16 kHz utterances, in the order of ``refs``) through the judges of ``factory``
    (``asr(language)``, ``sv()``, ``mos()`` return a callable or ``None``): ``wer`` / ``cer`` with bootstrap
    intervals, ``utmosv2``, and for ``sim_keys`` the ``speaker_sim`` with ``audio["recording"]``."""
    res: dict[str, dict] = {k: {} for k in audio}

    def free():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    asr = factory.asr(language)
    if asr is not None:
        for key, wavs in audio.items():
            hyps = [_plain(h, language) for h in asr(list(wavs))]
            rows = [{"hyp": h, **error_counts(r, h)} for r, h in zip(refs, hyps)]
            res[key].update({k: list(v) if isinstance(v, tuple) else v for k, v in summarize(rows).items()})
            print(key, f"WER {100 * res[key]['wer']:.2f}%", flush=True)
    del asr
    free()
    sv = factory.sv() if sim_keys else None
    if sv is not None:
        orig = [sv(w) for w in audio["recording"]]
        for key in sim_keys:
            res[key]["speaker_sim"] = mean_ci([float(sv(w) @ e) for w, e in zip(audio[key], orig)])
    del sv
    free()
    mos = factory.mos()
    if mos is not None:
        for key, wavs in audio.items():
            res[key]["utmosv2"] = mean_ci([mos(w) for w in wavs])
            print(key, f"UTMOSv2 {res[key]['utmosv2']['mean']:.3f}", flush=True)
    del mos
    free()
    return res


# ----------------------------------------------------------------------------------------------- the gate and tables


def tolerance_summary(conds: dict[str, dict], clean: float | None, noise: Sequence[float]) -> dict:
    """``noise_for_loss`` (the noise level that costs :data:`GATE_LOSS` UTMOSv2; ``None``: more than the largest
    level) and ``loss_at_noise`` (UTMOSv2 lost at :data:`GATE_NOISE`), from the judged conditions of one backend."""
    scores = [conds.get(f"noise{v:g}", {}).get("utmosv2", {}).get("mean") for v in noise]
    if clean is None or not noise or any(s is None for s in scores):
        return {}
    return {"noise_for_loss": level_for_loss(noise, scores, clean), "loss_at_noise": loss_at(noise, scores, clean),
            "max_noise": max(noise)}


def verdict(row: dict, recording: dict, max_roundtrip_loss: float | None, max_noise_loss: float | None) -> str | None:
    """``pass`` / ``fail`` against the thresholds given (``None``: no thresholds, or a number they need is missing).
    Round trip: UTMOSv2 of the recordings minus that of the 24 kHz round trip; noise: UTMOSv2 lost at 0.2 std."""
    checks = []
    if max_roundtrip_loss is not None:
        rec, rt = (d.get("utmosv2", {}).get("mean") for d in (recording, row["roundtrip"]["24k"]))
        checks.append(None if rec is None or rt is None else rec - rt <= max_roundtrip_loss)
    if max_noise_loss is not None:
        loss = row.get("tolerance", {}).get("loss_at_noise")
        checks.append(None if loss is None else loss <= max_noise_loss)
    if not checks or any(c is None for c in checks):
        return None
    return "pass" if all(checks) else "fail"


def _num(v, fmt: str = ".3f", scale: float = 1.0, suffix: str = "") -> str:
    """``scale * v`` in the fixed-point format ``fmt`` (``–`` for ``None``; a rounded zero has no minus sign)."""
    return "–" if v is None else f"{round(scale * v, int(fmt[-2])) + 0.0:{fmt}}{suffix}"


def _get(d: dict | None, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def tables(results: dict) -> str:
    """``results.md``: the gate table (one row per backend), the tolerance curves and the repeated-latents check."""
    rec, rows = results["recording"], results["backends"]
    rec_mos = _get(rec, "utmosv2", "mean")
    gate = any(r.get("gate") for r in rows.values())
    head = ["backend", "in → out", "frames", "WER", "UTMOSv2", "Δ recording", "SIM", "PESQ", "STOI", "mel L1",
            "LSD 8–12 kHz", "native: WER", "UTMOSv2", "LSD 8–12 kHz", "LSD 12–20 kHz",
            f"noise for −{GATE_LOSS:g} UTMOSv2", f"Δ at {GATE_NOISE:g} std", "Δ smoothing", "ABX error",
            "probe (majority)"]
    head += ["gate"] if gate else []
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    cells = ["recording", "–", "–", _num(rec.get("wer"), ".2f", 100, "%"), _num(rec_mos)] + ["–"] * (len(head) - 5)
    out.append("| " + " | ".join(cells) + " |")
    for name, r in rows.items():
        rt, nat, tol = r["roundtrip"]["24k"], r["roundtrip"].get("native") or {}, r.get("tolerance", {})
        mos = clean = _get(rt, "utmosv2", "mean")
        smooth = _get(tol, "conditions", "smooth", "utmosv2", "mean")
        noise = tol.get("noise_for_loss")
        noise_cell = _num(noise, ".2f", suffix=" std") if noise is not None else \
            (f"> {tol['max_noise']:g} std" if "max_noise" in tol else "–")
        probe = r.get("probe")
        cells = [name, f"{r['input_rate'] / 1000:g} → {r['output_rate'] / 1000:g} kHz", f"{r['frame_rate']:g} Hz",
                 _num(rt.get("wer"), ".2f", 100, "%"), _num(mos),
                 _num(None if mos is None or rec_mos is None else mos - rec_mos, "+.3f"),
                 _num(_get(rt, "speaker_sim", "mean")), _num(rt.get("pesq"), ".2f"), _num(rt.get("stoi")),
                 _num(rt.get("mel_l1")), _num(rt.get("lsd_8_12"), ".2f", suffix=" dB"),
                 _num(nat.get("wer"), ".2f", 100, "%"), _num(_get(nat, "utmosv2", "mean")),
                 _num(nat.get("lsd_8_12"), ".2f", suffix=" dB"), _num(nat.get("lsd_12_20"), ".2f", suffix=" dB"),
                 noise_cell, _num(None if tol.get("loss_at_noise") is None else -tol["loss_at_noise"], "+.3f"),
                 _num(None if smooth is None or clean is None else smooth - clean, "+.3f"),
                 _num(_get(r, "abx", "error"), ".1f", 100, "%"),
                 "–" if not probe else f"{100 * probe['accuracy']:.1f}% ({100 * probe['baseline']:.1f}%)"]
        cells += [r.get("gate") or "–"] if gate else []
        out.append("| " + " | ".join(cells) + " |")
    names = list(dict.fromkeys(c for r in rows.values() for c in r.get("tolerance", {}).get("conditions", {})))
    if names:
        out += ["", "UTMOSv2 (WER) of the perturbed frames, decoded:", "",
                "| backend | clean | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 2)]
        for name, r in rows.items():
            conds = r.get("tolerance", {}).get("conditions", {})
            cell = lambda d: f"{_num(_get(d, 'utmosv2', 'mean'))} ({_num(_get(d, 'wer'), '.2f', 100, '%')})"
            out.append(f"| {name} | {cell(r['roundtrip']['24k'])} | " + " | ".join(cell(conds.get(c)) for c in names)
                       + " |")
    rep = results.get("repeated_latents")
    if rep:
        r = rep["repeat"]
        out += ["", f"Copies of one codec frame ({rep['model']}): squared deviation from the group mean / channel "
                "variance.", "", "| latents | " + " | ".join(f"position {p}" for p in range(r)) + " | all |",
                "|" + "---|" * (r + 2)]
        for side in ("generated", "recorded"):
            d = rep[side]
            out.append(f"| {side} | " + " | ".join(_num(v, ".4f") for v in d["positions"])
                       + f" | {_num(d['share'], '.4f')} |")
        if rep.get("decoded"):
            out += ["", "The generated latents decoded from the mean of the copies and from the first copy alone:", "",
                    "| decoded from | WER | UTMOSv2 |", "|---|---|---|"]
            for key, label in (("mean", "the mean of the copies"), ("copy0", "the first copy")):
                d = rep["decoded"].get(key, {})
                out.append(f"| {label} | {_num(d.get('wer'), '.2f', 100, '%')} | "
                           f"{_num(_get(d, 'utmosv2', 'mean'))} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="prepared dataset with audio.bin (prepare --save-audio); with "
                   "audio_hr.bin (--hires-rate) the 44.1 / 48 kHz codecs are also tested at their own rate")
    p.add_argument("--split", default="val")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--backends", nargs="+", default=list(BACKENDS),
                   help=f"{', '.join(BACKENDS)} or mel:<vocoder> (a registry name or a checkpoint)")
    p.add_argument("--vocoder", default=None,
                   help=f"the vocoder of the bigvgan row (default {MEL_VOCODERS['bigvgan']})")
    p.add_argument("--vocos", default=None, help=f"the vocoder of the vocos row (default {MEL_VOCODERS['vocos']})")
    p.add_argument("--skip", nargs="+", default=[], choices=PARTS,
                   help="native: the hi-res round trip; tolerance; phonetic: ABX and probe; repeat: --model's check")
    p.add_argument("--no-judges", action="store_true", help="no WER / UTMOSv2 / SIM (signal metrics, ABX, probe only)")
    p.add_argument("--noise", type=float, nargs="*", default=list(NOISE), help="noise levels (standard deviations)")
    p.add_argument("--scale", type=float, nargs="*", default=list(SCALE), help="shrink factors")
    p.add_argument("--abx-per-pair", type=int, default=50, help="triplets per (speaker, character pair)")
    p.add_argument("--probe-steps", type=int, default=200, help="L-BFGS iterations of the linear probe")
    p.add_argument("--min-align-score", type=float, default=0.0, help="drop character spans aligned below this score")
    p.add_argument("--model", default=None,
                   help="a DiT or an ARDrift model trained on latents (model_ema.pt): the repeated-latents check")
    p.add_argument("--model-data", default=None, help="its latent data root (default: the checkpoint's data.root)")
    p.add_argument("--model-num", type=int, default=100)
    p.add_argument("--speaker", type=int, default=None, help="--model: only this speaker's utterances")
    p.add_argument("--temperature", type=float, default=0.3, help="--model: noise temperature")
    p.add_argument("--cfg", type=float, default=2.0, help="--model: guidance")
    p.add_argument("--max-roundtrip-loss", type=float, default=None,
                   help="gate: UTMOSv2 a 24 kHz round trip may lose against the recording")
    p.add_argument("--max-noise-loss", type=float, default=None,
                   help=f"gate: UTMOSv2 that noise of {GATE_NOISE:g} standard deviations may cost")
    p.add_argument("--asr", default="large-v3")
    p.add_argument("--asr-batch", type=int, default=16)
    p.add_argument("--save-wavs", type=int, default=0, help="write the first N round trips per backend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/codec_gate")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def gate_backend(name: str, backend: AudioBackend, rec: Recordings, spans: Sequence[Spans | None] | None, args,
                 audio: dict[str, list[np.ndarray]], extra: dict[str, Callable], out: Path) -> dict:
    """The round trips, the perturbed decodes (their 16 kHz audio goes to ``audio`` under ``<name>/<condition>``) and
    the phonetic measurements of one backend."""
    logmel = BigVGANLogMel()
    rate = backend.output_rate
    row: dict = {"input_rate": backend.input_rate, "output_rate": rate, "frame_rate": backend.frame_rate,
                 "dim": backend.dim, "roundtrip": {}}
    outs, latents, lags = roundtrip(backend, rec.wavs, SAMPLE_RATE)
    row["roundtrip"]["24k"] = mean_metrics([sample_metrics(w, SAMPLE_RATE, y, rate, logmel, extra)
                                            for w, y in zip(rec.wavs, outs)])
    row["lag_samples"] = sorted(set(lags))
    audio[f"{name}/24k"] = [to16k(y, rate) for y in outs]
    for k, y in enumerate(outs[: args.save_wavs]):
        (out / "wav").mkdir(exist_ok=True)
        sf.write(out / "wav" / f"{name.replace(':', '_').replace('/', '_')}_{k:04d}.wav", y.numpy(), rate)
    if "native" not in args.skip and max(backend.input_rate, rate) > SAMPLE_RATE:
        if rec.hires is None:
            print(f"[codec_gate] {name}: no native-rate round trip ({args.data} has no audio_hr.bin: "
                  "prepare --save-audio --hires-rate 48000)", flush=True)
        else:
            nat, _, _ = roundtrip(backend, rec.hires, rec.hr_rate)
            row["roundtrip"]["native"] = mean_metrics([sample_metrics(w, rec.hr_rate, y, rate, logmel, extra)
                                                       for w, y in zip(rec.hires, nat)])
            row["roundtrip"]["native"]["source_rate"] = rec.hr_rate
            audio[f"{name}/native"] = [to16k(y, rate) for y in nat]
    stats = LatentStats.from_latents(latents)
    stats.posterior = posterior_spread(backend, rec.wavs, SAMPLE_RATE)
    row["active_channels"] = int(active_channels(stats).sum())
    if stats.posterior is not None:  # what a posterior sample adds, in standard deviations of an active channel
        act = active_channels(stats)
        row["posterior_rel"] = float((stats.posterior / stats.std)[act].median()) if act.any() else None
    if "tolerance" not in args.skip:
        conds = perturbations(args.noise, args.scale)
        for c, _, _ in conds:
            audio[f"{name}/{c}"] = []
        for k, (w, z) in enumerate(zip(rec.wavs, latents)):
            eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(args.seed + k))
            n = -(-len(w) * rate // SAMPLE_RATE)
            for c, kind, value in conds:
                y = decode(backend, perturb(z, stats, kind, value, eps), n, lags[k])
                audio[f"{name}/{c}"].append(to16k(y, rate))
        row["tolerance"] = {"conditions": {c: {} for c, _, _ in conds}}
    if spans is not None:
        row.update(phonetic([normalize(z, stats) for z in latents], spans, rec.speakers, backend.frame_rate,
                            args.abx_per_pair, args.probe_steps, args.seed))
    return row


def run(args, judges=None, aligner: Callable[[Tensor, str], Spans | None] | None = None,
        backends: dict[str, AudioBackend] | None = None, generate: Callable | None = None) -> dict:
    """The gate. ``judges``: a :class:`JudgeFactory` (``None``: signal metrics only); ``aligner``: ``(16 kHz
    waveform, normalised text) -> character spans`` (default :class:`MmsAligner`); ``backends``: built backends by
    name, in place of :func:`build_backend`; ``generate``: in place of :func:`generated_latents`."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recordings(args.data, args.split, args.offset, args.num)
    audio: dict[str, list[np.ndarray]] = {"recording": [to16k(w, SAMPLE_RATE) for w in rec.wavs]}
    spans = None
    if "phonetic" not in args.skip:
        aligner = aligner or MmsAligner(args.device, args.min_align_score)
        spans = [aligner(torch.from_numpy(w), t) for w, t in zip(audio["recording"], rec.texts)]
        print(f"[codec_gate] aligned {sum(s is not None for s in spans)}/{len(spans)} utterances, "
              f"{sum(len(s) for s in spans if s)} characters", flush=True)
        del aligner
    extra = optional_metrics()
    rows: dict[str, dict] = {}
    for name in args.backends:
        backend = backends[name] if backends is not None else build_backend(name, args)
        rows[name] = gate_backend(name, backend, rec, spans, args, audio, extra, out)
        print(name, json.dumps({k: v for k, v in rows[name].items() if k != "tolerance"}), flush=True)
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    recording: dict = {}
    if judges is not None:
        sim = [k for k in audio if k.endswith(("/24k", "/native"))]
        scores = judge(audio, rec.refs, rec.language, judges, sim)
        recording = scores["recording"]
        for name, row in rows.items():
            for mode in row["roundtrip"]:
                row["roundtrip"][mode].update(scores[f"{name}/{mode}"])
            if "tolerance" in row:
                tol = row["tolerance"]
                tol["conditions"] = {c: scores[f"{name}/{c}"] for c in tol["conditions"]}
                tol.update(tolerance_summary(tol["conditions"], _get(row, "roundtrip", "24k", "utmosv2", "mean"),
                                             args.noise))
            row["gate"] = verdict(row, recording, args.max_roundtrip_loss, args.max_noise_loss)
    results = {"data": str(args.data), "split": args.split, "utterances": len(rec), "seconds": rec.seconds,
               "language": rec.language, "hr_rate": rec.hr_rate, "seed": args.seed,
               "thresholds": {"max_roundtrip_loss": args.max_roundtrip_loss, "max_noise_loss": args.max_noise_loss},
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "torch": torch.__version__,
               "recording": recording, "backends": rows}
    if args.model and "repeat" not in args.skip:
        gen, stored, repeat, *info = (generate or generated_latents)(args)
        results["repeated_latents"] = {"model": str(args.model), "repeat": repeat, "utterances": len(gen),
                                       "temperature": args.temperature, "cfg": args.cfg,
                                       "generated": repeat_deviation(gen, repeat),
                                       "recorded": repeat_deviation(stored, repeat)}
        if judges is not None and info and repeat > 1:  # one copy against the mean of the copies (#92)
            info = info[0]
            results["repeated_latents"]["decoded"] = judge(decode_copies(gen, repeat, info, args.device),
                                                           info["refs"], info["language"], judges)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.md").write_text(tables(results) + "\n")
    print(tables(results))
    print(f"-> {out / 'results.json'}")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args, judges=None if args.no_judges else JudgeFactory(args.device, args.asr, args.asr_batch))


if __name__ == "__main__":
    main()
