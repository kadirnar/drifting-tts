"""More than one positive per drift problem (#98): a frozen teacher draws positives beside the recording, with its
one-step head or with a multi-step flow-matching head trained on its states. Tiny models on the CPU."""

import json
import sys

import pytest
import torch

from drifting_tts.config import Config, load_config
from drifting_tts.data import MelDataset, collate
from drifting_tts.drift import energy_score
from drifting_tts.models.ar import FlowHead
from drifting_tts.train import load_tts
from drifting_tts.train_ar import Teacher, training_step
from scripts import head_finetune as hf
from tests.test_head_diagnostics import _checkpoint, _data_root

OVER = ["drift.gen_per_cond=4", "drift.max_rows=24", "model.ar.dim=32", "model.ar.depth=2", "model.ar.heads=2",
        "model.ar.head_dim=32", "model.ar.head_depth=2", "model.text.d=16", "model.text.layers=1",
        "model.text.heads=2", "model.text.ffn=32", "model.text.spk_dim=8", "model.text.dropout=0.0"]


def test_flow_head_is_a_velocity_field_sampled_in_steps():
    torch.manual_seed(0)
    head = FlowHead(cond=16, out=12, dim=32, depth=2)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    s, z = torch.randn(5, 16), torch.randn(5, 12)
    one = head.sample(s, z, steps=1)
    torch.testing.assert_close(one, z + head.velocity(s, z, torch.zeros(5)))
    assert head.sample(s, z, steps=4).shape == (5, 12) and not torch.allclose(head.sample(s, z, 4), one)
    a, b = head.velocity(s, z, torch.zeros(5)), head.velocity(s, z, torch.full((5,), 0.5))
    assert not torch.allclose(a, b)  # the time is read
    loss = head.loss(s, torch.randn(5, 12), torch.Generator().manual_seed(1))
    loss.backward()
    assert torch.isfinite(loss) and head.body.out.weight.grad.abs().sum() > 0
    assert head.time.net[0].weight.grad is not None


def test_flow_matching_learns_a_conditional_distribution():
    """Two states, two distributions: N(-2, 0.3) and N(+2, 1). A few hundred steps, then samples in 16 steps."""
    torch.manual_seed(0)
    head = FlowHead(cond=4, out=1, dim=32, depth=2)
    opt = torch.optim.Adam(head.parameters(), 3e-3)
    states = torch.eye(4)[:2] * 3
    mean, std = torch.tensor([-2.0, 2.0]), torch.tensor([0.3, 1.0])
    for _ in range(600):
        i = torch.randint(2, (256,))
        target = (mean[i] + std[i] * torch.randn(256))[:, None]
        loss = head.loss(states[i], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for i in range(2):
        x = head.sample(states[i].expand(2000, -1), torch.randn(2000, 1), steps=16)
        assert abs(float(x.mean()) - float(mean[i])) < 0.25 and abs(float(x.std()) - float(std[i])) < 0.25


def test_energy_score_averages_over_several_observations():
    torch.manual_seed(0)
    gen, pos = torch.randn(3, 5, 4), torch.randn(3, 2, 4)
    both, _ = energy_score(gen, pos)
    single = [energy_score(gen, pos[:, i: i + 1])[0] for i in range(2)]
    torch.testing.assert_close(both, (single[0] + single[1]) / 2)


def test_training_step_with_a_teachers_positives(tmp_path, monkeypatch):
    root, path = _data_root(tmp_path, n=4, frames=40), _checkpoint(tmp_path)
    model = load_tts(path, "cpu")[0].train()
    ds = MelDataset(root, "val", min_frames=1)
    batch = collate([ds[i] for i in range(4)])
    cfg = load_config("configs/ar_tts.yaml", OVER)
    tau = torch.nn.Parameter(torch.tensor(1.0))
    teacher = Teacher(Config({"path": path, "positives": 3, "temperature": 1.0}), "cpu")
    assert teacher.flow is None and not any(p.requires_grad for p in teacher.model.parameters())
    torch.manual_seed(0)
    loss, metrics = training_step(model, batch, cfg, "cpu", tau, teacher=teacher)
    loss.backward()
    assert torch.isfinite(loss) and float(metrics["teacher_std"]) > 0 and model.head.out.weight.grad.abs().sum() > 0
    # the teacher is the model itself here: its samples are as spread as the student's
    assert abs(float(metrics["teacher_std"]) / float(metrics["sample_std"]) - 1) < 0.5
    torch.manual_seed(0)
    plain = training_step(model, batch, cfg, "cpu", tau)
    assert "teacher_std" not in plain[1] and float(plain[0]) != float(loss)
    energy = load_config("configs/ar_tts.yaml", OVER + ["drift.loss=energy"])
    torch.manual_seed(0)
    assert torch.isfinite(training_step(model, batch, energy, "cpu", tau, teacher=teacher)[0])

    # teacher B: a flow head trained on the model's states, then the positives come from it
    out = tmp_path / "flow"
    monkeypatch.setattr(sys, "argv", ["head_finetune.py", "--model", path, "--data", str(root), "--split", "val",
                                      "--out", str(out), "--loss", "flow", "--steps", "100", "--rows", "8",
                                      "--accum", "2", "--samples", "2", "--warmup", "1"])
    hf.main()
    saved = torch.load(out / "head.pt")
    assert saved["kind"] == "flow" and "time.net.0.weight" in saved["head"]
    assert json.loads((out / "log.json").read_text())[-1]["step"] == 100
    flow = Teacher(Config({"path": path, "positives": 3, "head": str(out / "head.pt"), "steps": 2}), "cpu")
    assert flow.flow is not None
    torch.manual_seed(0)
    loss2, m2 = training_step(model, batch, cfg, "cpu", tau, teacher=flow)
    assert torch.isfinite(loss2) and float(m2["teacher_std"]) > 0 and float(loss2) != float(loss)
    with pytest.raises(ValueError, match="not a flow head"):
        torch.save({"head": {}, "kind": "drift"}, tmp_path / "bad.pt")
        Teacher(Config({"path": path, "head": str(tmp_path / "bad.pt")}), "cpu")
