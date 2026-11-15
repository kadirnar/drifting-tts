"""The head-only fine-tune (#97) on the CPU: its kernel maps, the state cache and a few steps on a tiny model."""

import json
import sys

import pytest
import torch

from drifting_tts.models.ar import ar_features, patchify
from drifting_tts.models.mae2d import MelMAE2d
from scripts import head_finetune as hf
from tests.test_head_diagnostics import _checkpoint, _data_root


def test_kernel_features_shapes_and_the_join_with_the_history():
    torch.manual_seed(0)
    x, prev = torch.randn(3, 5, 400, requires_grad=True), torch.randn(3, 400)
    maps = ("patch", "frame", "detail", "delta", "delta_detail", "tile", "tile_detail", "delta2", "ctile2",
            "ctile25_detail")
    f = hf.kernel_features(x, prev, 4, 100, maps)
    assert {k: tuple(v.shape) for k, v in f.items()} == {
        "patch": (3, 5, 1, 400), "frame": (3, 5, 4, 100), "detail": (3, 5, 4, 100), "delta": (3, 5, 4, 100),
        "delta_detail": (3, 5, 4, 100), "tile": (3, 5, 19, 40), "tile_detail": (3, 5, 19, 40),
        "delta2": (3, 5, 3, 100), "ctile2": (3, 5, 99, 10), "ctile25_detail": (3, 5, 7, 125)}
    base = ar_features(x, 4, 100)
    for name in ("patch", "frame", "detail"):  # the training's maps, unchanged
        torch.testing.assert_close(f[name], base[name])
    fr, last = x.view(3, 5, 4, 100), prev.view(3, 4, 100)[:, -1]
    seq = torch.cat([last[:, None, None].expand(-1, 5, -1, -1), fr], 2)  # the last recorded frame, then the patch
    torch.testing.assert_close(f["delta"], seq[:, :, 1:] - seq[:, :, :-1])  # location 0: the join with the recording
    torch.testing.assert_close(f["delta2"][:, :, 0], seq[:, :, 2] - 2 * seq[:, :, 1] + seq[:, :, 0])
    torch.testing.assert_close(f["tile"][:, :, 1].view(3, 5, 4, 10), fr[..., 5:15])  # frames x bins 5..14
    torch.testing.assert_close(f["ctile2"][:, :, 3].view(3, 5, 5, 2), seq[..., 3:5])  # 5 frames x bins 3..4
    assert set(hf.kernel_features(x, prev, 4, 100, ("patch", "delta"))) == {"patch", "delta"}
    f["tile_detail"].sum().backward()
    assert x.grad.abs().sum() > 0


def test_whitened_cosine_maps():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 400)
    coef = hf.dct2(x.view(3, 5, 4, 100))
    torch.testing.assert_close(coef.pow(2).sum((-1, -2)), x.pow(2).sum(-1), rtol=1e-4, atol=0)  # energy is kept
    torch.testing.assert_close(coef[..., 0, :], hf.dct2(x.view(3, 5, 4, 100).mean(-2, keepdim=True))[..., 0, :] * 2,
                               atol=1e-4, rtol=1e-4)  # temporal frequency 0: the mean of the frames
    scale = torch.rand(4, 100) + 0.5
    f = hf.kernel_features(x, torch.randn(3, 400), 4, 100, ("wdct", "wpatch"), scale=scale)
    assert {k: tuple(v.shape) for k, v in f.items()} == {"wdct0": (3, 5, 1, 100), "wdct1": (3, 5, 1, 100),
                                                         "wdct2": (3, 5, 1, 100), "wdct3": (3, 5, 1, 100),
                                                         "wpatch": (3, 5, 1, 400)}
    torch.testing.assert_close(f["wdct3"][:, :, 0], coef[:, :, 3] / scale[3])
    torch.testing.assert_close(f["wpatch"][:, :, 0], (coef / scale).reshape(3, 5, 400))


def test_residual_scale_is_the_spread_of_the_recordings_around_the_head(tmp_path):
    from drifting_tts.data import MelDataset
    from drifting_tts.train import load_tts

    root, path = _data_root(tmp_path, n=3, frames=40), _checkpoint(tmp_path)
    model, _, _ = load_tts(path, "cpu")
    rows = hf.cache_states(model, MelDataset(root, "val", min_frames=1), "cpu", noise_max=0.0, batch=2, seed=0)
    scale = hf.residual_scale(model.head, rows, 4, 100, samples=8, dev="cpu")
    assert scale.shape == (4, 100) and (scale > 0).all()
    # white-noise recordings around a near-constant prior: every coefficient has about the mel's unit spread
    assert 0.6 < float(scale.pow(2).mean().sqrt()) < 1.4


def test_map_names():
    assert hf.map_name("delta2") == "delta2" and hf.map_name("ctile10_detail") == "ctile10_detail"
    with pytest.raises(Exception, match="unknown kernel map"):
        hf.map_name("ctile")


def test_mae_maps_look_at_the_patch_in_its_recorded_context():
    torch.manual_seed(0)
    mae = MelMAE2d(base_channels=8, layers=(1, 1, 1, 1)).eval().requires_grad_(False)
    x, prev = torch.randn(2, 3, 400, requires_grad=True), torch.randn(2, 1200)  # three recorded patches of history
    f = hf.kernel_features(x, prev, 4, 100, ("patch", "delta", "mae"), mae=mae)
    assert {k: tuple(v.shape) for k, v in f.items() if k.startswith("mae")} == {
        "mae_conv1": (2, 3, 100, 8), "mae_layer1": (2, 3, 100, 8), "mae_layer2": (2, 3, 25, 16),
        "mae_layer3": (2, 3, 13, 32), "mae_layer4": (2, 3, 7, 64)}  # the columns of the last 4 of 16 frames
    torch.testing.assert_close(f["delta"][:, :, 0], x.view(2, 3, 4, 100)[:, :, 0] - prev.view(2, 12, 100)[:, None, -1])
    f["mae_layer1"].sum().backward()
    assert x.grad.abs().sum() > 0
    other = hf.kernel_features(x.detach(), torch.randn(2, 1200), 4, 100, ("mae",), mae=mae)
    assert not torch.allclose(other["mae_layer2"], f["mae_layer2"])  # the history is part of the window


def test_cache_states_holds_whole_patches_after_the_context(tmp_path):
    from drifting_tts.data import MelDataset
    from drifting_tts.train import load_tts

    root, path = _data_root(tmp_path, n=3, frames=30), _checkpoint(tmp_path)  # 30 frames: 7 whole patches
    model, _, _ = load_tts(path, "cpu")
    ds = MelDataset(root, "val", min_frames=1)
    rows = hf.cache_states(model, ds, "cpu", noise_max=0.0, batch=2, seed=0)
    assert {k: tuple(v.shape) for k, v in rows.items()} == {"state": (18, 32), "target": (18, 400),
                                                            "base": (18, 400), "prev": (18, 400)}
    y = patchify(model.pad_frames(ds[0]["mel"][None]), 4)[0]  # utterance 0: its steps 1..6 are rows 0..5
    torch.testing.assert_close(rows["target"][:6].float(), y[1:7], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(rows["prev"][:6].float(), y[:6], atol=2e-3, rtol=2e-3)
    three = hf.cache_states(model, ds, "cpu", noise_max=0.0, batch=2, seed=0, context=3)  # steps 3..6
    assert three["prev"].shape == (12, 1200) and three["target"].shape == (12, 400)
    torch.testing.assert_close(three["target"][:4].float(), y[3:7], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(three["prev"][0].float(), y[0:3].flatten(), atol=2e-3, rtol=2e-3)  # oldest first


def test_head_finetune_trains_only_the_head_and_its_head_loads_into_the_diagnostics(tmp_path, monkeypatch, capsys):
    from scripts import drift_calibration as calib

    root, model = _data_root(tmp_path, n=4, frames=40), _checkpoint(tmp_path)
    out = tmp_path / "ft"
    monkeypatch.setattr(sys, "argv", ["head_finetune.py", "--model", model, "--data", str(root), "--split", "val",
                                      "--out", str(out), "--maps", "patch", "frame", "detail", "delta", "ctile2",
                                      "--steps", "100", "--rows", "8", "--accum", "2", "--samples", "4", "--warmup",
                                      "1", "--cache", str(tmp_path / "rows.pt")])
    hf.main()
    saved = torch.load(out / "head.pt")
    before = torch.load(model, weights_only=False)["ema"]
    assert saved["maps"] == ["patch", "frame", "detail", "delta", "ctile2"] and set(saved["taus"]) == set(saved["maps"])
    assert any(not torch.equal(v, before["head." + k]) for k, v in saved["head"].items())
    log = json.loads((out / "log.json").read_text())
    assert log[-1]["step"] == 100 and 0 < log[-1]["p_data/ctile2"] <= 1
    monkeypatch.setattr(sys, "argv", ["head_finetune.py", "--model", model, "--data", str(root), "--split", "val",
                                      "--out", str(tmp_path / "energy"), "--maps", "patch", "wdct", "--loss", "energy",
                                      "--steps", "100", "--rows", "8", "--accum", "1", "--samples", "4", "--warmup",
                                      "1", "--cache", str(tmp_path / "rows.pt")])  # the same rows, the energy score
    hf.main()
    log = json.loads((tmp_path / "energy" / "log.json").read_text())
    assert {"spread/patch", "spread/wdct0", "spread/wdct3", "energy"} <= set(log[-1]) and "tau/patch" not in log[-1]
    assert torch.load(tmp_path / "energy" / "head.pt")["maps"] == ["patch", "wdct0", "wdct1", "wdct2", "wdct3"]
    monkeypatch.setattr(sys, "argv", ["drift_calibration.py", "--model", model, "--head", str(out / "head.pt"),
                                      "--data", str(root), "--num", "2", "--batch", "2", "--samples", "4",
                                      "--temperatures", "1.0", "--max-rows", "8", "--no-free"])
    calib.main()
    assert "spread / error" in capsys.readouterr().out
