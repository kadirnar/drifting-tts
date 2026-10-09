import math
import random

import numpy as np
import pytest
import torch

from drifting_tts.prosody import (
    PausePolicy,
    f0_statistics,
    frames_from_logw,
    letter_tokens,
    paired_metrics,
    pitch_reversals,
    prosody_features,
    scale_deviations,
    seed_diversity,
    token_f0,
    token_report,
    token_targets,
    voiced_tokens,
)
from drifting_tts.text import text_to_ids

SR = 24_000


def tone(f0_hz: np.ndarray, sr: int = SR, frame_s: float = 0.01) -> np.ndarray:
    """A harmonic signal following a per-frame F0 contour (0 = silence)."""
    f = np.repeat(f0_hz, int(sr * frame_s))
    phase = 2 * np.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.3 * x * (f > 0)).astype(np.float64)


def test_f0_statistics_in_semitones():
    st = np.concatenate([np.zeros(50), np.full(50, 4.0)])  # two equal halves 4 semitones apart
    f0 = 100 * 2 ** (st / 12)
    s = f0_statistics(np.concatenate([[0.0], f0, [0.0]]))
    assert s["f0_std"] == pytest.approx(2.0, abs=1e-6)
    assert s["f0_range"] == pytest.approx(4.0, abs=1e-6)
    assert s["f0_skew"] == pytest.approx(0.0, abs=1e-6)
    assert s["f0_kurt"] == pytest.approx(-2.0, abs=1e-6)  # two-point distribution
    assert s["f0_move"] == pytest.approx(4.0 / 99, abs=1e-6)  # one 4-semitone step over 99 voiced pairs
    assert f0_statistics(np.zeros(10)) == {}


def test_pitch_reversals_count_slope_sign_changes_per_voiced_second():
    t = np.arange(200) * 0.01  # 2 s
    wavy = 120 * 2 ** (3 * np.sin(2 * np.pi * 2 * t) / 12)  # 2 Hz: 4 turning points per second
    assert pitch_reversals(wavy) == pytest.approx(4.0, abs=0.6)
    assert pitch_reversals(np.linspace(100, 130, 200)) == 0.0
    assert math.isnan(pitch_reversals(np.zeros(50)))


def test_token_f0_of_a_step_contour_rendered_exactly():
    frames = np.array([10, 10, 10])  # mel frames of 256 / 24000 s
    hz = np.array([100.0, 120.0, 110.0])
    t = np.arange(32) * 0.01
    f0 = hz[np.minimum((t / (10 * 256 / SR)).astype(int), 2)]
    r = token_f0(f0, frames, pitch_st=12 * np.log2(hz), mask=np.ones(3, bool))
    assert r["f0_within"] == pytest.approx(0.0, abs=1e-9) and r["f0_tok_std"] == pytest.approx(0.0, abs=1e-9)
    assert r["render_r"] == pytest.approx(1.0) and r["render_flat"] == pytest.approx(1.0)
    assert r["render_bias"] == pytest.approx(0.0, abs=1e-6)
    half = token_f0(f0, frames, pitch_st=2 * 12 * np.log2(hz), mask=np.ones(3, bool))
    assert half["render_flat"] == pytest.approx(0.5)  # the audio moves half as much as its conditioning


def test_features_find_the_pause_and_the_pitch_movement():
    contour = np.concatenate([np.linspace(100, 140, 60), np.zeros(30), np.linspace(140, 100, 60)])
    f = prosody_features(tone(contour), text="aaa aaa")
    assert f["pauses"] == 1 and f["pause_lengths"][0] == pytest.approx(0.3, abs=0.03)
    assert f["voiced_pct"] == pytest.approx(80, abs=8)
    assert f["f0_std"] == pytest.approx(5.83 / 12**0.5, abs=0.15)  # 100-140 Hz: 5.8 semitones, ~uniform
    assert f["rate_sps"] == pytest.approx(6 / f["speech_s"])


def test_paired_metrics_of_a_signal_with_itself_and_a_flattened_copy():
    contour = 120 * 2 ** (3 * np.sin(np.linspace(0, 3 * np.pi, 150)) / 12)
    x = tone(contour)
    same = paired_metrics(x, x)
    assert same["f0_corr"] > 0.999 and same["f0_rmse"] < 0.05 and same["dur_ratio"] == pytest.approx(1.0)
    flat = paired_metrics(x, tone(120 * 2 ** (np.log2(contour / 120) * 0.5)))
    assert flat["f0_corr"] > 0.95 and flat["f0_rmse"] > 0.5  # same shape, half the excursion


def _targets(pred_frames, gt_frames, pitch, pitch_gt, text="ab, c"):
    ids = torch.tensor(text_to_ids(text, normalized=True))
    n = len(ids)
    gt = torch.tensor(gt_frames, dtype=torch.long)[:n]
    logw = torch.tensor(pred_frames, dtype=torch.float).log()[:n] - 1e-4  # ceil(exp(.)) gives the frames back
    return {"ids": ids, "logw": logw, "durations_gt": gt,
            "pitch": torch.tensor(pitch)[:n], "pitch_gt": torch.tensor(pitch_gt)[:n],
            "voiced_gt": torch.ones(n, dtype=torch.bool)}


def test_token_report_perfect_and_flat_predictors():
    gt = [3, 5, 1, 9, 2, 4, 1, 7, 2, 6, 3]
    pitch_gt = [0.1, 0.5, -0.2, 0.8, 0.0, -0.4, 0.3, 0.2, -0.1, 0.6, 0.1]
    t = _targets(gt, gt, pitch_gt, pitch_gt)
    r = token_report(t, frame_rate=100.0, scale=1.0)
    assert r["dur_r"] == pytest.approx(1.0) and r["dur_flat"] == pytest.approx(1.0) and r["dur_mae"] == 0
    assert r["pitch_r"] == pytest.approx(1.0) and r["pitch_mae"] == pytest.approx(0.0, abs=1e-6)
    assert r["len_ratio"] == pytest.approx(1.0)
    assert r["punct"][","][0] == pytest.approx([(4 + 1 + 7 + 2) / 100] * 2)  # "," + blank + " " + blank
    mean = float(np.mean(pitch_gt))
    flat = token_report(_targets(gt, gt, [mean] * 11, pitch_gt), frame_rate=100.0)
    assert flat["pitch_flat"] == pytest.approx(0.0, abs=1e-6)


def test_token_masks():
    ids = torch.tensor(text_to_ids("as da.", normalized=True))  # _ a _ s _ ' ' _ d _ a _ . _
    v = voiced_tokens(ids).tolist()
    assert v[1] and not v[3] and v[7] and not v[11]  # a voiced, s and '.' excluded, d voiced
    assert not v[4]  # blank between 's' and ' ': neither is a voiced letter
    assert letter_tokens(ids).tolist() == [False, True, False, True, False, False, False, True, False, True,
                                           False, False, False]


def test_scale_deviations():
    x = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
    mask = torch.tensor([[True, True, True, False]])
    assert torch.equal(scale_deviations(x, 1.0, mask), x)
    y = scale_deviations(x, 2.0, mask)
    assert torch.allclose(y, torch.tensor([[0.0, 2.0, 4.0, 10.0]]))
    z = scale_deviations(x, 1.5, mask, keep_total=True)
    assert torch.allclose(z[mask].exp().sum(), x[mask].exp().sum()) and z[0, 3] == 10.0


def test_seed_diversity():
    f = 100 * 2 ** (np.sin(np.linspace(0, 6, 300)) / 12)
    same = seed_diversity([f, f.copy(), f.copy()], lengths=[3.0, 3.0, 3.0])
    assert same["f0_spread"] == pytest.approx(0.0, abs=1e-9) and same["len_cv"] == 0.0
    moved = seed_diversity([f, f * 2 ** (1 / 12)], durations=[torch.tensor([2, 4]), torch.tensor([2, 8])])
    assert moved["f0_spread"] == pytest.approx(0.5, abs=1e-6)  # one semitone apart: std 0.5
    assert moved["token_dur_std"] == pytest.approx(math.log(2) / 4, abs=1e-6)


def test_pause_policy():
    p = PausePolicy({".": (0.5, 0.2), ",": (0.1, 0.05)}, edge=0.15)
    assert p("bir cümle.") == pytest.approx(0.35)
    assert p("bir cümle?") == pytest.approx(0.35)  # unknown mark: the '.' entry
    assert p("uzun bir cümle") == 0.0  # cut without a mark: ',' entry, below the edge silence
    j = PausePolicy({".": (0.5, 0.2)}, edge=0.15, jitter=1.0)
    draws = [j("a.", random.Random(s)) for s in range(200)]
    assert min(draws) >= 0.0 and np.std(draws) > 0.1 and draws[3] == j("a.", random.Random(3))
    assert PausePolicy.for_voice("12345").edge >= 0  # an unknown speaker gets the corpus policy
    studio = PausePolicy.for_voice(722)
    assert studio("a.") == 0.0  # the studio voice pauses less than a generated sentence's own edge silence
    assert PausePolicy.for_voice("389")("a.") > 0.5


def test_synthesizer_pause_accepts_a_policy():
    from drifting_tts.synthesize import silence

    assert silence(0.15, "a.", random.Random(0)).numel() == int(0.15 * SR)
    assert silence(lambda s, rng: 0.5 if s.endswith(".") else 0.0, "a.", random.Random(0)).numel() == SR // 2


def _tiny_pitch_tts():
    from drifting_tts.config import Config
    from drifting_tts.models.tts import DriftingTTS

    cfg = Config({"text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
                  "gen": {"hidden": 32, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 2.0, "n_registers": 2,
                          "noise_classes": 4, "noise_coords": 2, "num_steps": 1},
                  "pitch": {"enabled": True}})
    torch.manual_seed(0)
    model = DriftingTTS(cfg, num_speakers=2).eval()
    for p in model.generator.final.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return model


def test_prosody_overrides_reproduce_the_default_path_and_take_oracle_durations():
    model = _tiny_pitch_tts()
    ids = torch.tensor([text_to_ids("ab ca.", normalized=True)])
    n, spk = torch.tensor([ids.shape[1]]), torch.tensor([1])

    def synth(**kw):
        g = torch.Generator().manual_seed(7)
        return model.synthesize(ids, n, spk, temperature=0.5, length_scale=1.3, generator=g, **kw)[0]

    t = token_targets(model, ids, spk)
    same = synth(durations=frames_from_logw(t["logw"], 1.3)[None], pitch=t["pitch"][None, None])
    assert torch.equal(same, synth())
    assert not torch.equal(synth(pitch=t["pitch"][None, None] + 1.0), synth())
    mel = torch.randn(1, 100, 40)
    f0 = torch.where(torch.rand(1, 40) > 0.3, 100 + 50 * torch.rand(1, 40), torch.zeros(1, 40))
    gt = token_targets(model, ids, spk, mel, f0)
    assert int(gt["durations_gt"].sum()) == 40 and (gt["durations_gt"] >= 1).all()
    assert synth(durations=gt["durations_gt"][None], pitch=gt["pitch_gt"][None, None]).shape[-1] == 40
    assert torch.equal(gt["pitch_gt"] != 0, gt["voiced_gt"])


def test_parse_system():
    from drifting_tts.prosody_eval import parse_system

    s = parse_system("oracle-dur+pitch-gain-1.4")
    assert s.kind == "onepass" and s.dur == ("gt",) and s.pitch == ("gain", 1.4) and s.needs_recording
    assert parse_system("predicted").kind == "split" and parse_system("pause-punct-j1").jitter == 1.0
    assert parse_system("dur-mix-0.5").dur == ("mix", 0.5) and not parse_system("pitch-gain-1.2").needs_recording
    w = parse_system("onepass@cfg1.5@win64@t0.6")
    assert (w.cfg, w.attn_window, w.temperature) == (1.5, 64, 0.6) and parse_system("predicted@cfg1").cfg == 1.0
    for bad in ("oracle-everything", "onepass@beam4", "recording@cfg1"):
        with pytest.raises(ValueError):
            parse_system(bad)
