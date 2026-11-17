"""scripts/prefix_eval.py on the CPU: what it computes from audio and transcripts (the models, Whisper and UTMOSv2
are not exercised here)."""

import numpy as np
import pytest

from drifting_tts.metrics import error_counts
from scripts import prefix_eval as pe


def _row(name: str, ref: str, hyp: str) -> dict:
    return {"name": name, "ref": ref, "hyp": hyp, **error_counts(ref, hyp)}


def test_quiet_share_counts_the_frames_far_under_the_loudest():
    sr = 24_000
    rng = np.random.default_rng(0)
    speech = 0.3 * rng.standard_normal(sr)
    hiss = 1e-4 * rng.standard_normal(sr)  # 70 dB under
    assert pe.quiet_share(np.concatenate([speech, hiss]), sr) == pytest.approx(0.5)
    assert pe.quiet_share(speech, sr) == 0.0
    assert pe.quiet_share(np.concatenate([speech, 0.5 * speech]), sr) == 0.0  # 6 dB under is not quiet
    assert pe.quiet_share(np.zeros(10), sr) == 1.0  # shorter than a frame


def test_sentence_stats_tell_falling_silent_from_misreading():
    text = "so there is this guy on the platform"
    silent = pe.sentence_stats([_row("0.wav", text, "so there is this")])  # the beginning and nothing else
    assert silent["said"] == pytest.approx(0.5) and silent["right_of_said"] == 1.0
    assert silent["beginning_only"] == 1 and silent["first"] == 4 and silent["wer"] == pytest.approx(0.5)
    wrong = pe.sentence_stats([_row("0.wav", text, "so there is this cat in a hat")])  # as many words, half wrong
    assert wrong["said"] == 1.0 and wrong["right_of_said"] == pytest.approx(0.5)
    assert wrong["beginning_only"] == 0 and wrong["first"] == 4 and wrong["wer"] == pytest.approx(0.5)
    right = pe.sentence_stats([_row("0.wav", text, text), _row("1.wav", text, "so there is this guy on the plate")])
    assert (right["entire"], right["under20"], right["under50"]) == (1, 2, 2)
    assert right["beginning_only"] == 0  # a complete transcript is not "the beginning only"


def test_paired_difference_is_over_the_sentences_both_have():
    text = "one two three four"
    a = [_row("0.wav", text, text), _row("1.wav", text, "one two"), _row("2.wav", text, "")]
    b = [_row("0.wav", text, "one"), _row("1.wav", text, "one two"), _row("9.wav", text, text)]
    d, lo, hi = pe.paired_difference(a, b)
    assert d == pytest.approx((0 + 2 - 3 - 2) / 8)  # 2.wav and 9.wav have no partner
    assert lo <= d <= hi
    assert pe.paired_difference(a, a)[:3] == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        pe.paired_difference(a, [_row("7.wav", text, text)])
