"""Kernel maps over a window (#97): the recorded patches before a step and the step's patch through a frozen Mel-MAE,
in ``train-ar``; and the energy score as the training's loss. Tiny models on the CPU."""

import pytest
import torch

from drifting_tts.config import Config, load_config
from drifting_tts.models.ar import ARDriftTTS, history_patches, window_features
from drifting_tts.models.mae2d import MelMAE2d
from drifting_tts.train_ar import kernel_loss, training_step
from tests.test_prompt import _batch

CFG = {"kind": "ar", "text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
       "ar": {"patch": 4, "lookahead": 2, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32, "head_depth": 2}}
OVER = ["drift.gen_per_cond=4", "drift.max_rows=24"]
WINDOW = ["drift.window.mae=unused", "drift.window.context=3", "drift.window.rows=8", "drift.window.samples=3",
          "drift.window.max_locations=6"]


def _model() -> ARDriftTTS:
    torch.manual_seed(0)
    m = ARDriftTTS(Config(CFG), num_speakers=3, n_mels=100).train()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def _mae() -> MelMAE2d:
    torch.manual_seed(1)
    return MelMAE2d(base_channels=8, layers=(1, 1, 1, 1)).eval().requires_grad_(False)


def test_history_patches_are_the_patches_before_each_step():
    yp = torch.arange(5.0)[None, :, None].expand(2, 5, 3)  # patch t holds the value t
    h = history_patches(yp, 3)
    assert h.shape == (2, 5, 3, 3)
    assert h[0, :, :, 0].tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]]  # oldest first


def test_window_features_see_the_join_and_reach_only_the_generated_patch():
    mae = _mae()
    hist = torch.randn(2, 3, 400, requires_grad=True)
    x = torch.randn(2, 3, 400, requires_grad=True)
    f = window_features(mae, hist.detach(), x, 4, 100)
    assert {k: tuple(v.shape) for k, v in f.items()} == {
        "mae_conv1": (2, 3, 100, 8), "mae_layer1": (2, 3, 100, 8), "mae_layer2": (2, 3, 25, 16),
        "mae_layer3": (2, 3, 13, 32), "mae_layer4": (2, 3, 7, 64)}  # the columns of the last 4 of 16 frames
    f["mae_layer2"].sum().backward()
    assert x.grad.abs().sum() > 0 and hist.grad is None
    other = window_features(mae, torch.randn(2, 3, 400), x.detach(), 4, 100)
    assert not torch.allclose(other["mae_layer2"], f["mae_layer2"])  # the history is part of the window


def test_kernel_loss_is_the_field_or_the_energy_score():
    torch.manual_seed(0)
    gen, pos = {"a": torch.randn(5, 4, 2, 6, requires_grad=True)}, {"a": torch.randn(5, 1, 2, 6)}
    tau = torch.nn.Parameter(torch.tensor(1.0))
    field, info = kernel_loss(gen, pos, "field", tau)
    assert "p_data/a" in info and float(info["tau_loss"]) != 0.0
    energy, info = kernel_loss(gen, pos, "energy", tau)
    assert "spread/a" in info and float(info["tau_loss"]) == 0.0 and float(energy) != float(field)
    with pytest.raises(ValueError, match="field or energy"):
        kernel_loss(gen, pos, "other", tau)


@pytest.mark.parametrize("loss", ["field", "energy"])
@pytest.mark.parametrize("window_loss", ["field", "energy"])
def test_training_step_with_window_maps(loss, window_loss):
    m, mae = _model(), _mae()
    cfg = load_config("configs/ar_tts.yaml", OVER + WINDOW + [f"drift.loss={loss}",
                                                              f"drift.window.loss={window_loss}"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    torch.manual_seed(0)
    value, metrics = training_step(m, _batch(), cfg, "cpu", tau, mae=mae, detail=True)
    value.backward()
    assert torch.isfinite(value) and m.head.out.weight.grad.abs().sum() > 0
    assert "window" in metrics and ("p_data" in metrics) == ("field" in (loss, window_loss))
    assert ("spread" in metrics) == ("energy" in (loss, window_loss))
    assert (tau.grad is not None and tau.grad != 0) == ("field" in (loss, window_loss))
    assert any(k.startswith(("spread/mae_", "p_data/mae_")) for k in metrics)
    plain = load_config("configs/ar_tts.yaml", OVER + [f"drift.loss={loss}"])
    torch.manual_seed(0)
    a = training_step(m, _batch(), plain, "cpu", tau)
    assert "window" not in a[1] and float(a[0]) != float(value)
    with pytest.raises(ValueError, match="Mel-MAE"):
        training_step(m, _batch(), cfg, "cpu", tau)


def test_train_ar_with_window_maps(tmp_path):
    from drifting_tts.cli import main
    from tests.test_data import _fake_parquet
    from tests.test_loop_eval import TINY

    _fake_parquet(tmp_path / "x.parquet", n=10)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--backend", "bigvgan"])
    main(["train-mae", "--config", "configs/mae2d.yaml", "--workdir", str(tmp_path / "mae"), f"data.root={data}",
          "train.cpu=true", "train.steps=1", "train.batch_size=2", "train.crop_frames=16", "train.num_workers=0",
          "model.base_channels=8", "model.layers=[1,1,1,1]", "model.speaker_head=false", "model.mask_patch=[10,4]",
          "train.save_every=1", "train.log_every=1"])
    work = tmp_path / "run"
    main(["train-ar", "--workdir", str(work), f"data.root={data}", *TINY, "train.steps=2",
          f"drift.window.mae={tmp_path / 'mae' / 'mae_ema.pt'}", "drift.window.rows=8", "drift.window.samples=2",
          "drift.loss=energy"])
    assert (work / "model_ema.pt").exists()
