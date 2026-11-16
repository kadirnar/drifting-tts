"""Guidance for ARDrift (#99): the null branch without the text, the voice or both, voice dropout in training, and
the guided state distilled into the backbone. Tiny models on the CPU."""

import pytest
import torch

from drifting_tts.config import Config, load_config
from drifting_tts.models.ar import ARDriftTTS, ARGenerator
from drifting_tts.train_ar import distill_step, training_step
from tests.test_prompt import PROMPT, _batch
from tests.test_reference import CFG, OVER, _ref_batch


def _model(seed: int = 0) -> ARDriftTTS:
    torch.manual_seed(seed)
    m = ARDriftTTS(Config(CFG), num_speakers=3, n_mels=100).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def test_the_null_branch_is_without_the_text_the_voice_or_both():
    m = _model()
    cp, spk = torch.randn(2, 5, 12), torch.tensor([0, 2])
    tokens, _ = m.reference_tokens(torch.randn(2, 100, 24))
    null = m.ref_enc.null_tokens(2)
    c, s, t = m.null_branch(cp, spk, tokens, "text")
    assert not c[2:].any() and torch.equal(c[:2], cp) and s.tolist() == [0, 2, 0, 2] and torch.equal(t[2:], tokens)
    c, s, t = m.null_branch(cp, spk, tokens, "voice")
    assert torch.equal(c[2:], cp) and s.tolist() == [0, 2, 3, 3] and torch.equal(t[2:], null)
    c, s, t = m.null_branch(cp, spk, tokens, "both")
    assert not c[2:].any() and s.tolist() == [0, 2, 3, 3] and torch.equal(t[2:], null) and torch.equal(t[:2], tokens)
    with pytest.raises(ValueError, match="text, voice or both"):
        m.null_branch(cp, spk, tokens, "pitch")
    cond = torch.randn(2, 116, 29)
    out = {g: m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), cfg_scale=1.5, tokens=tokens, guide=g)
           for g in ("text", "voice", "both")}
    assert not torch.allclose(out["text"], out["voice"], atol=1e-4)
    assert not torch.allclose(out["text"], out["both"], atol=1e-4)
    plain = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), tokens=tokens)
    for g in out:  # a scale of 1 is no guidance, whatever the null branch
        one = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), cfg_scale=1.0, tokens=tokens, guide=g)
        assert torch.equal(one, plain)
    fast = ARGenerator(m, max_steps=16, graphs=False)
    for g, ref in out.items():
        for _ in range(2):
            got = fast(cond, spk, 0.7, torch.Generator().manual_seed(5), cfg_scale=1.5, tokens=tokens, guide=g)
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_voice_dropout_trains_utterances_without_any_voice():
    m = _model().train()
    over = OVER + [f"drift.prompt.{k}={v}" for k, v in PROMPT.items()] + ["drift.reference.prob=1.0"]
    cfg = load_config("configs/ar_tts.yaml", over + ["drift.voice_dropout=1.0"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    torch.manual_seed(0)
    loss, metrics = training_step(m, _ref_batch(), cfg, "cpu", tau)
    loss.backward()
    assert metrics["no_voice"] == 1.0 and metrics["prompted"] == 0.0 and metrics["referenced"] == 0.0
    assert metrics["id_dropped"] == 1.0
    grad = m.spk.weight.grad
    assert grad[m.prompt_row].abs().sum() > 0 and not grad[:3].any()  # no speaker's row is read
    assert m.encoder.spk.weight.grad[m.prompt_row].abs().sum() > 0 and not m.encoder.spk.weight.grad[:3].any()
    assert m.ref_enc.null.grad.abs().sum() > 0 and not m.ref_enc.out.weight.grad.any()
    off = load_config("configs/ar_tts.yaml", over)
    torch.manual_seed(0)
    assert training_step(m, _ref_batch(), off, "cpu", tau)[1]["prompted"] == 1.0
    plain = ARDriftTTS(Config({**CFG, "ar": {k: v for k, v in CFG["ar"].items() if k not in ("prompt", "reference")}}),
                       num_speakers=3, n_mels=100)
    with pytest.raises(ValueError, match="voice_dropout"):
        training_step(plain, _batch(), load_config("configs/ar_tts.yaml", OVER[:2] + ["drift.voice_dropout=0.5"]),
                      "cpu", tau)


@pytest.mark.parametrize("guide", ["text", "both"])
def test_distillation_trains_the_backbone_towards_the_guided_state(guide):
    teacher = _model().requires_grad_(False)
    student = _model().train()
    cfg = load_config("configs/ar_tts.yaml", OVER + ["drift.distill.teacher=unused", "drift.distill.cfg_scale=1.0",
                                                     f"drift.distill.guide={guide}", "drift.input_noise=0.0"])
    loss, metrics = distill_step(student, teacher, _batch(), cfg, "cpu")
    assert float(loss) < 1e-10 and float(metrics["guided_gap"]) < 1e-10  # scale 1: the state the model already gives
    cfg = load_config("configs/ar_tts.yaml", OVER + ["drift.distill.teacher=unused", "drift.distill.cfg_scale=2.0",
                                                     f"drift.distill.guide={guide}", "drift.input_noise=0.0"])
    loss, metrics = distill_step(student, teacher, _batch(), cfg, "cpu")
    loss.backward()
    assert float(loss) > 0 and abs(float(loss) - float(metrics["guided_gap"])) < 1e-6 * max(float(loss), 1.0)
    assert student.x_in.weight.grad.abs().sum() > 0 and student.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert student.head.out.weight.grad is None and student.encoder.spk.weight.grad is None  # not in the graph
    opt = torch.optim.Adam([p for n, p in student.named_parameters()
                            if not n.startswith(("encoder.", "head."))], 1e-3)
    first = float(loss)
    for _ in range(30):
        torch.manual_seed(0)
        loss = distill_step(student, teacher, _batch(), cfg, "cpu")[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first


def test_train_ar_distills_guidance(tmp_path):
    from drifting_tts.cli import main
    from tests.test_data import _fake_parquet
    from tests.test_loop_eval import TINY

    _fake_parquet(tmp_path / "x.parquet", n=10)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--backend", "bigvgan"])
    base, work = tmp_path / "base", tmp_path / "distilled"
    main(["train-ar", "--workdir", str(base), f"data.root={data}", *TINY, "train.steps=2", "drift.cond_dropout=0.5"])
    main(["train-ar", "--workdir", str(work), f"data.root={data}", *TINY, "train.steps=3", "train.save_every=3",
          f"train.init_from={base / 'model_ema.pt'}", f"drift.distill.teacher={base / 'model_ema.pt'}",
          "drift.distill.cfg_scale=1.5", "train.ema_decay=0.0"])
    a = torch.load(base / "model_ema.pt", weights_only=False)["ema"]
    b = torch.load(work / "model_ema.pt", weights_only=False)["ema"]
    same = [k for k in a if torch.equal(a[k], b[k])]
    assert all(k in same for k in a if k.startswith(("encoder.", "head.")))  # the head is not touched
    assert not torch.equal(a["blocks.0.attn.qkv.weight"], b["blocks.0.attn.qkv.weight"])
