"""The head diagnostics (#97, #85) on the CPU: mel texture, the classifier two-sample test and the calibration script
on a tiny model. No trained weights, no vocoder."""

import json
import sys

import numpy as np
import pytest
import torch

from drifting_tts.config import Config
from drifting_tts.models.ar import ARDriftTTS
from scripts import c2st_mel as c2st
from scripts import drift_calibration as calib
from scripts import gta_mismatch as gm

CFG = {"data": {"max_frames": 1500},
       "drift": {"crop_frames": 8},
       "model": {"kind": "ar",
                 "text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
                 "ar": {"patch": 4, "lookahead": 2, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32,
                        "head_depth": 2}}}


def _smooth_mel(n: int, t: int, seed: int = 0) -> torch.Tensor:
    """Mel-like ``[n, 100, t]``: a random walk along time, smooth along frequency."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 100, t, generator=g).cumsum(-1) * 0.1
    return torch.nn.functional.avg_pool1d(x.transpose(1, 2), 5, 1, 2).transpose(1, 2)


def _data_root(tmp_path, n: int = 6, frames: int = 64):
    """A prepared-data root with ``n`` utterances of random mels (the layout of ``drifting-tts prepare``)."""
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(0)
    mels = (rng.standard_normal((n * frames, 100)) - 6).astype(np.float16)
    mels.tofile(root / "mels.bin")
    with open(root / "index.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"text": "merhaba dünya", "norm_text": "merhaba dünya", "speaker": f"s{i % 2}",
                                "spk_id": i % 2, "split": "val", "frames": frames, "offset": i * frames,
                                "audio_seconds": frames / 93.75, "quality_score": None}) + "\n")
    (root / "speakers.json").write_text(json.dumps({"s0": 0, "s1": 1}))
    (root / "stats.json").write_text(json.dumps({"backend": "bigvgan", "language": "tr", "mean": -6.0, "std": 1.0}))
    return root


def _checkpoint(tmp_path) -> str:
    torch.manual_seed(0)
    m = ARDriftTTS(Config(CFG["model"]), num_speakers=2, n_mels=100)
    with torch.no_grad():
        for p in m.parameters():  # zero-initialised output layers: make the head's noise visible
            p.add_(torch.randn_like(p) * 0.05)
    path = tmp_path / "model.pt"
    torch.save({"ema": m.state_dict(), "config": CFG, "num_speakers": 2, "n_mels": 100, "step": 1,
                "stats": {"mean": -6.0, "std": 1.0, "backend": "bigvgan"}}, path)
    return str(path)


# ----------------------------------------------------------------------------------------------------- texture


def test_texture_sees_smoothing_and_ignores_the_padding():
    mel = _smooth_mel(2, 64) + 0.2 * torch.randn(2, 100, 64, generator=torch.Generator().manual_seed(1))
    length = torch.tensor([64, 64])
    sharp = calib.texture(mel, length, 4)
    blurred = calib.texture(torch.nn.functional.avg_pool1d(mel, 3, 1, 1), length, 4)
    for name in ("varl", "detail_rms", "delta_rms"):
        assert (blurred[name] <= sharp[name]).all(), name
    assert (blurred["varl"] < 0.7 * sharp["varl"]).all()
    padded = torch.cat([mel, 5 * torch.randn(2, 100, 16)], -1)  # frames past the length: not looked at
    again = calib.texture(padded, length, 4)
    for name, v in sharp.items():
        torch.testing.assert_close(again[name], v, rtol=1e-4, atol=1e-6)


def test_boundary_ratio_is_one_for_a_recording_and_rises_when_patches_do_not_join():
    mel = _smooth_mel(4, 128)
    length = torch.full((4,), 128)
    assert calib.texture(mel, length, 4)["boundary_ratio"].mean() == pytest.approx(1.0, abs=0.1)
    offsets = 0.5 * torch.randn(4, 100, 32).repeat_interleave(4, -1)  # every patch shifted on its own
    assert calib.texture(mel + offsets, length, 4)["boundary_ratio"].mean() > 2.0


def test_temporal_cosine_transform_separates_the_mean_from_the_alternation():
    from drifting_tts.utils import dct_matrix

    m = dct_matrix(4)
    torch.testing.assert_close(m @ m.T, torch.eye(4), atol=1e-6, rtol=0)
    still, flip = torch.ones(4), torch.tensor([1.0, -1.0, 1.0, -1.0])
    torch.testing.assert_close(m @ still, torch.tensor([2.0, 0, 0, 0]), atol=1e-6, rtol=0)  # all in coefficient 0
    assert (m @ flip)[0].abs() < 1e-6 and (m @ flip)[3].abs() > (m @ flip)[1].abs()  # mostly in the fastest
    x = torch.randn(3, 5, 4, 100)
    torch.testing.assert_close(calib.frame_detail(x)[1, 2], calib.spectral_detail(x[1, 2].T[None])[0].T)


def test_patch_scalars():
    x = torch.randn(3, 5, 400)
    s = calib.patch_scalars(x, 4, 100)
    assert {k: tuple(v.shape) for k, v in s.items()} == {"level": (3, 5), "depth": (3, 5), "motion": (3, 5)}
    torch.testing.assert_close(s["level"], x.mean(-1))
    still = torch.randn(3, 5, 1, 100).expand(-1, -1, 4, -1).reshape(3, 5, 400)  # four equal frames: no motion
    assert calib.patch_scalars(still, 4, 100)["motion"].abs().max() < 1e-6


# -------------------------------------------------------------------------------------------------------- C2ST


def test_views_are_what_they_say():
    x = torch.randn(4, 100, 8)
    assert c2st.view(x, "full").shape == (4, 1, 100, 8)
    parts = [c2st.view(x, n)[:, 0] for n in ("low", "mid", "high")]
    torch.testing.assert_close(torch.cat(parts, 1), x)
    torch.testing.assert_close(c2st.view(x, "envelope") + c2st.view(x, "detail"), x[:, None], atol=1e-6, rtol=0)
    torch.testing.assert_close(c2st.view(x, "delta")[:, 0], x[:, :, 1:] - x[:, :, :-1])
    torch.testing.assert_close(c2st.view(x, "contour")[:, 0, 0], x.mean(1))
    shuffled = c2st.view(x, "shuffled")[:, 0]
    torch.testing.assert_close(shuffled.sort(-1).values, x.sort(-1).values)  # the same frames, another order
    assert not torch.equal(shuffled, x)


def test_pairs_draw_the_same_window_of_both_inside_one_utterance():
    real = [torch.full((100, 10), float(i)) + torch.arange(10.0) / 100 for i in range(5)]  # utterance i: i.00 .. i.09
    fake = [m + 1000 for m in real]
    pairs = c2st.Pairs(real, fake, 4, "cpu")
    a, b = pairs.sample(200, torch.Generator().manual_seed(0))
    assert a.shape == b.shape == (200, 100, 4)
    torch.testing.assert_close(b - a, torch.full_like(a, 1000.0))
    assert (a[:, 0].floor() == a[:, 0, :1].floor()).all()  # never across two utterances
    assert len(pairs.starts) == 5 * 7
    whole = c2st.Pairs(real, fake, 4, "cpu", stride=4)  # whole patches: starts at frames 0 and 4 of each utterance
    assert len(whole.starts) == 5 * 2
    a, _ = whole.sample(100, torch.Generator().manual_seed(0))
    frame = ((a[:, 0, 0] % 1) * 100).round()
    assert set(frame.tolist()) == {0.0, 4.0}


def test_pairs_keep_speech_windows_clamp_the_floor_and_dither_both_sides():
    quiet, loud = torch.full((100, 8), -1.7), torch.zeros(100, 8)
    real = [torch.cat([quiet, loud], 1)]
    fake = [real[0] - 1.0]  # below the recording's floor
    pairs = c2st.Pairs(real, fake, 4, "cpu", active=-0.5)
    assert pairs.starts.tolist() == [8, 9, 10, 11, 12]  # windows of loud frames only
    assert float(pairs.fake.min()) == pytest.approx(float(pairs.real.min()))
    g = torch.Generator().manual_seed(0)
    a, b = c2st.Pairs(real, real, 4, "cpu", dither=0.1).sample(64, g)
    assert 0.05 < float((a - b).std()) < 0.3  # independent noise on each side


def test_c2st_is_at_chance_for_one_distribution_and_finds_a_smoothed_one():
    torch.manual_seed(0)
    real = [torch.randn(100, 80) for _ in range(20)]
    same = [torch.randn(100, 80) for _ in range(20)]
    null = c2st.c2st(real, same, 8, "full", "cpu", steps=60, batch=32, seed=0)
    assert abs(null["accuracy"] - 0.5) < 0.06
    blurred = [torch.nn.functional.avg_pool1d(m, 3, 1, 1) for m in real]
    found = c2st.c2st(real, blurred, 8, "full", "cpu", steps=60, batch=32, seed=0)
    assert found["accuracy"] > 0.9 and found["paired_accuracy"] > 0.9


# ----------------------------------------------------------------------------------------- calibration script


def test_calibration_script_on_a_tiny_model(tmp_path, monkeypatch, capsys):
    root, model = _data_root(tmp_path), _checkpoint(tmp_path)
    out = tmp_path / "calibration.json"
    monkeypatch.setattr(sys, "argv", ["drift_calibration.py", "--model", model, "--data", str(root), "--num", "4",
                                      "--batch", "2", "--samples", "8", "--temperatures", "1.0", "0.5", "--max-rows",
                                      "16", "--out", str(out)])
    calib.main()
    r = json.loads(out.read_text())
    assert r["utterances"] == 4 and set(r["temperatures"]) == {"T1", "T0.5"}
    t1, t05 = r["temperatures"]["T1"], r["temperatures"]["T0.5"]
    for key in ("patch", "frame", "detail"):  # half the noise: less spread for the same error scale
        assert 0 < t05["spread_over_error"][key] < t1["spread_over_error"][key]
    assert set(t1["rank"]) == {"level", "depth", "motion"} and 0 <= t1["rank"]["level"]["top"] <= 1
    for name in ("mel", "detail"):  # one ratio per temporal frequency of the 4-frame patch
        assert len(t1["spread_skill_time"][name]) == 4 and sum(t1["error_share_time"][name]) == pytest.approx(1.0)
        assert all(a > b > 0 for a, b in zip(t1["spread_skill_time"][name], t05["spread_skill_time"][name]))
    assert set(r["texture"]) == {"recording", "T1/teacher_forced", "T1/free_running", "T0.5/teacher_forced",
                                 "T0.5/free_running"}
    assert r["texture"]["recording"]["boundary_ratio"] == pytest.approx(1.0, abs=0.2)
    assert "spread / error" in capsys.readouterr().out


# ------------------------------------------------------------------------------------ GTA mismatch (#90)


def test_gta_mismatch_measures_distance_pause_energy_and_patch_joins():
    rec = [m for m in _smooth_mel(3, 64)]
    assert gm.distance_db(rec, rec, 2.0)["mean"] == 0.0
    d = gm.distance_db(rec, [m + 0.1 for m in rec], 2.0)
    assert d["mean"] == pytest.approx(0.1 * 2.0 * gm.DB, rel=1e-4) and len(d["parts"]) == 4
    assert d["parts"] == pytest.approx([d["mean"]] * 4, rel=1e-4)
    late = [m + 0.1 * (torch.arange(64) >= 48) for m in rec]  # the error is in the last quarter only
    assert gm.distance_db(rec, late, 2.0)["parts"][:3] == [0.0, 0.0, 0.0]
    # a murmur in the quietest frames only: +6 dB of power there is 0.5 * ln(4) in log-magnitude
    loud = []
    for m in rec:
        quiet = (2 * m).exp().sum(0).argsort()[:6]  # 10% of 64 frames
        x = m.clone()
        x[:, quiet] += 0.5 * np.log(4.0)
        loud.append(x)
    assert gm.pause_excess_db(rec, loud, 0.0, 1.0)["mean"] == pytest.approx(10 * np.log10(4.0), abs=1e-3)
    assert gm.pause_excess_db(rec, rec, 0.0, 1.0)["mean"] == pytest.approx(0.0, abs=1e-4)
    # a recording has no patch grid; independent offsets per patch do not join
    assert gm.frame_change_db(rec, 1.0, 4)["ratio"] == pytest.approx(1.0, abs=0.1)
    steps = [m + 0.3 * torch.randn(16, generator=torch.Generator().manual_seed(0)).repeat_interleave(4) for m in rec]
    c = gm.frame_change_db(steps, 1.0, 4)
    assert c["ratio"] > 1.5 and c["inside"] == pytest.approx(gm.frame_change_db(rec, 1.0, 4)["inside"], rel=1e-4)


def test_gta_mismatch_renditions_share_the_recordings_frames(tmp_path):
    from drifting_tts.data import MelDataset
    from drifting_tts.train import load_tts

    model, _, _ = load_tts(_checkpoint(tmp_path), "cpu")
    ds = MelDataset(_data_root(tmp_path), "val", min_frames=1)
    idx = [4, 0, 2]
    mels = gm.renditions(model, ds, idx, "cpu", 0.5, batch=2)
    assert set(mels) == set(gm.KINDS) and all(len(v) == 3 for v in mels.values())
    for j, i in enumerate(idx):  # in the order asked for, frame for frame with the recording
        torch.testing.assert_close(mels["rec"][j], ds[i]["mel"].float(), rtol=1e-4, atol=1e-5)
        assert all(mels[k][j].shape == (100, 64) for k in gm.KINDS)
    assert not torch.allclose(mels["tf"][0], mels["tf2"][0])  # two draws of the head's noise
    cold = gm.renditions(model, ds, idx, "cpu", 0.0, batch=2)
    torch.testing.assert_close(cold["tf"][0], cold["tf2"][0])  # no noise: one teacher-forced mel
    torch.testing.assert_close(cold["free"][0][:, :4], cold["tf"][0][:, :4], rtol=1e-4, atol=1e-5)  # the first patch
    assert not torch.allclose(cold["free"][0][:, 4:], cold["tf"][0][:, 4:], atol=1e-3)
