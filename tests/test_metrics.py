import pytest
import torch
import torch.nn.functional as F

from drifting_tts.metrics import (
    BANDS_HZ,
    SpectralComparison,
    bootstrap_ci,
    edit_distance,
    error_counts,
    mel_center_freqs,
)


def test_edit_distance_and_error_counts():
    assert edit_distance("kitap", "kitab") == 1
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "") == 3
    assert edit_distance("bu bir test".split(), "bu test".split()) == 1
    assert error_counts("bu bir test", "bu test") == {"char_errors": 4, "chars": 11, "word_errors": 1, "words": 3}


def test_bootstrap_ci_brackets_the_estimate():
    num, den = [1, 0, 3, 2, 0, 1], [10, 12, 9, 11, 10, 8]
    lo, hi = bootstrap_ci(num, den)
    assert lo <= sum(num) / sum(den) <= hi and lo < hi
    assert bootstrap_ci([0.5] * 5) == (0.5, 0.5)


def test_mel_center_freqs():
    for backend in ("vocos", "bigvgan"):
        c = mel_center_freqs(backend)
        assert c.shape == (100,) and (c[1:] >= c[:-1]).all() and c[-1] <= 12_000
        mid = (c >= 1200) & (c < 4500)
        assert 30 <= int(mid.sum()) <= 40  # ≈ bins 35-69 of 100


def _harmonic_mels(frames: int = 40):
    """Log-mels of a harmonic comb (F0 gliding 110 -> 220 Hz) sampled at the mel centre frequencies."""
    centers = mel_center_freqs("vocos")
    f0 = torch.linspace(110, 220, frames)
    harmonics = torch.arange(1, 110)[:, None, None] * f0[None, None, :]
    mag = torch.exp(-0.5 * ((centers[None, :, None] - harmonics) / 25) ** 2).sum(0)
    return torch.log(mag * torch.exp(-centers / 4000)[:, None] + 1e-3), centers


def test_harmonic_contrast_of_a_smoothed_spectrum_is_below_one():
    rec, centers = _harmonic_mels()
    smooth = F.avg_pool1d(rec.T[:, None], 5, 1, 2, count_include_pad=False)[:, 0].T  # blur along frequency
    acc = SpectralComparison(centers)
    acc.add(smooth, rec, voiced=torch.ones(rec.shape[1], dtype=torch.bool))
    s = acc.summary()
    assert all(s[f"hc_{b}"] < 0.8 for b in BANDS_HZ), s
    assert s["voiced_frames"] == rec.shape[1] and s["utterances"] == 1

    same = SpectralComparison(centers)
    same.add(rec, rec)  # default voicing: the louder half of the frames
    s = same.summary()
    for b in BANDS_HZ:
        assert s[f"hc_{b}"] == pytest.approx(1.0) and s[f"gv_{b}"] == pytest.approx(1.0)
        assert s[f"level_db_{b}"] == pytest.approx(0.0, abs=1e-6)


def test_global_variance_and_level():
    rec, centers = _harmonic_mels()
    mean = rec.mean(1, keepdim=True)
    acc = SpectralComparison(centers)
    acc.add(mean + 0.5 * (rec - mean) + 1.0, rec)  # half the temporal deviation, +1 neper (8.69 dB)
    s = acc.summary()
    for b in BANDS_HZ:
        assert s[f"gv_{b}"] == pytest.approx(0.25, rel=1e-4)
        assert s[f"level_db_{b}"] == pytest.approx(8.686, abs=1e-3)
