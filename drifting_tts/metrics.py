"""Model-free metrics: edit-distance error rates with bootstrap intervals, and spectral statistics that compare
generated log-mels with the recordings on the same frames (ground-truth alignment).

**Harmonic contrast** is the mean of ``|x[k+1] - 2 x[k] + x[k-1]|`` along mel frequency ``k`` on voiced frames.
It measures the depth of the harmonic peaks and valleys, and a muffled, over-smoothed spectrum scores low.
**Global variance** is the per-bin variance over time within an utterance (temporal dynamics). **Level** is the
mean log-mel difference in dB. Each is reported per band (low / mid ≈ 1.2–4.5 kHz / high) as generated vs.
recorded: the first two as ratios, so 1.0 means "like the recordings".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from .audio import SAMPLE_RATE, make_logmel

BANDS_HZ = {"low": (0.0, 1200.0), "mid": (1200.0, 4500.0), "high": (4500.0, SAMPLE_RATE / 2 + 1)}
DB_PER_NEPER = 20 / math.log(10)  # log-mels are natural logs of magnitudes


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_counts(ref: str, hyp: str) -> dict[str, int]:
    """CER / WER numerators and denominators of one utterance (characters include spaces, as in ``jiwer``)."""
    return {"char_errors": edit_distance(ref, hyp), "chars": len(ref),
            "word_errors": edit_distance(ref.split(), hyp.split()), "words": len(ref.split())}


def bootstrap_ci(num: Sequence[float], den: Sequence[float] | None = None, n: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap interval of ``sum(num) / sum(den)`` (a mean if ``den`` is None), resampling
    utterances."""
    num = np.asarray(num, dtype=np.float64)
    den = np.ones_like(num) if den is None else np.asarray(den, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def mel_center_freqs(backend: str = "vocos") -> Tensor:
    """Centre frequency in Hz (peak of the triangular filter) of every mel bin of the backend's front end."""
    m = make_logmel(backend)
    fb = m.mel.mel_scale.fb if backend == "vocos" else m.fb.T  # [n_freqs, n_mels]
    return torch.linspace(0, SAMPLE_RATE / 2, fb.shape[0])[fb.argmax(0)]


def second_difference(mel: Tensor) -> Tensor:
    """``|x[k+1] - 2 x[k] + x[k-1]|`` along frequency: ``[n_mels, T] -> [n_mels - 2, T]`` (bins 1 .. n-2)."""
    return (mel[2:] - 2 * mel[1:-1] + mel[:-2]).abs()


def loud_frames(mel: Tensor) -> Tensor:
    """Voicing proxy when no F0 is available: frames louder than the utterance's median frame."""
    e = mel.mean(0)
    return e > e.median()


class SpectralComparison:
    """Accumulates per-band statistics of generated vs. recorded log-mels that share their frames."""

    def __init__(self, center_hz: Tensor):
        self.bands = {b: (center_hz >= lo) & (center_hz < hi) for b, (lo, hi) in BANDS_HZ.items()}
        self.sums: dict[str, float] = {}
        self.voiced = self.frames = self.utterances = 0

    def _add(self, key: str, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value

    def add(self, gen: Tensor, rec: Tensor, voiced: Tensor | None = None) -> None:
        """``gen``, ``rec``: ``[n_mels, T]`` unnormalised log-mels; ``voiced``: ``[T]`` bool (default: loud frames)."""
        gen, rec = gen.float().cpu(), rec.float().cpu()
        voiced = loud_frames(rec) if voiced is None else voiced.bool().cpu()
        for name, x in (("gen", gen), ("rec", rec)):
            d2, var, mean = second_difference(x)[:, voiced], x.var(1), x.mean(1)
            for b, m in self.bands.items():
                self._add(f"{name}_hc_{b}", float(d2[m[1:-1]].sum()))
                self._add(f"{name}_gv_{b}", float(var[m].mean()))
                self._add(f"{name}_level_{b}", float(mean[m].mean()))
        self.voiced += int(voiced.sum())
        self.frames += gen.shape[1]
        self.utterances += 1

    def summary(self) -> dict[str, float]:
        """``hc_*``: harmonic contrast ratio; ``gv_*``: global-variance ratio; ``level_db_*``: level difference."""
        s, out = self.sums, {}
        for b in self.bands:
            out[f"hc_{b}"] = s[f"gen_hc_{b}"] / max(s[f"rec_hc_{b}"], 1e-12)
            out[f"gv_{b}"] = s[f"gen_gv_{b}"] / max(s[f"rec_gv_{b}"], 1e-12)
            out[f"level_db_{b}"] = DB_PER_NEPER * (s[f"gen_level_{b}"] - s[f"rec_level_{b}"]) / self.utterances
        for b in self.bands:  # absolute recording contrast (per voiced frame and bin), to compare data sets
            n = max(self.voiced * int(self.bands[b][1:-1].sum()), 1)
            out[f"rec_hc_{b}"] = s[f"rec_hc_{b}"] / n
        out.update(utterances=self.utterances, frames=self.frames, voiced_frames=self.voiced)
        return out
