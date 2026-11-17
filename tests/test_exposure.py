"""Exposure bias (#101): noise on the model's own history at inference, the history's noise level as an input of the
backbone, and free-running rollouts as the history in training. Tiny models on the CPU."""

import torch

from drifting_tts.config import Config, load_config
from drifting_tts.models.ar import ARDriftTTS, ARGenerator, patchify, unpatchify
from drifting_tts.train_ar import training_step
from tests.test_prompt import _batch

CFG = {"kind": "ar", "text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
       "ar": {"patch": 4, "lookahead": 2, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32, "head_depth": 2}}
OVER = ["drift.gen_per_cond=4", "drift.max_rows=24"]


def _model(seed: int = 0, **ar) -> ARDriftTTS:
    torch.manual_seed(seed)
    m = ARDriftTTS(Config({**CFG, "ar": {**CFG["ar"], **ar}}), num_speakers=3, n_mels=100).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def test_history_noise_changes_what_the_backbone_reads_not_the_draws():
    m = _model()
    cond, spk = torch.randn(2, 116, 29), torch.tensor([0, 2])
    clean = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5))
    noisy = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), history_noise=0.15)
    torch.testing.assert_close(noisy[..., :4], clean[..., :4])  # the first patch has no history: the same draw
    assert not torch.allclose(noisy[..., 4:], clean[..., 4:], atol=1e-4)
    assert torch.equal(noisy, m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), history_noise=0.15))
    fast = ARGenerator(m, max_steps=16, graphs=False)
    for kw in ({}, {"cfg_scale": 1.5}):
        ref = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), history_noise=0.15, **kw)
        for _ in range(2):
            out = fast(cond, spk, 0.7, torch.Generator().manual_seed(5), history_noise=0.15, **kw)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fast(cond, spk, 0.7, torch.Generator().manual_seed(5)), clean, atol=1e-4, rtol=1e-4)
    tied = _model(tied=True)
    ref = tied.generate(cond, spk, 0.7, torch.Generator().manual_seed(5), history_noise=0.15)
    out = ARGenerator(tied, max_steps=16, graphs=False)(cond, spk, 0.7, torch.Generator().manual_seed(5),
                                                        history_noise=0.15)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_the_noise_level_is_an_input_that_starts_at_zero():
    torch.manual_seed(0)
    plain = ARDriftTTS(Config(CFG), num_speakers=3, n_mels=100).eval()
    lev = ARDriftTTS(Config({**CFG, "ar": {**CFG["ar"], "noise_level": True}}), num_speakers=3, n_mels=100).eval()
    missing, unexpected = lev.load_state_dict(plain.state_dict(), strict=False)
    assert all(k.startswith("level_in.") for k in missing) and not unexpected
    cond, y, spk = torch.randn(2, 116, 16), torch.randn(2, 100, 16), torch.tensor([0, 1])
    noise = torch.tensor([0.1, 0.3])
    torch.manual_seed(1)
    a = plain.teacher_forced(cond, y, spk, noise)
    torch.manual_seed(1)
    torch.testing.assert_close(lev.teacher_forced(cond, y, spk, noise), a)  # a fine-tune starts as the model was
    with torch.no_grad():
        lev.level_in.net[-1].weight.add_(torch.randn_like(lev.level_in.net[-1].weight) * 0.1)
    torch.manual_seed(1)
    b = lev.teacher_forced(cond, y, spk, noise)
    torch.manual_seed(1)
    c = lev.teacher_forced(cond, y, spk, torch.tensor([0.1, 0.1]))
    torch.manual_seed(1)
    d = lev.teacher_forced(cond, y, spk, torch.tensor([0.1, 0.3]) * 0 + noise)
    assert not torch.allclose(b, a, atol=1e-4) and torch.allclose(b[0], c[0], atol=1e-5) and torch.equal(b, d)
    # generation: level 0 without history noise, the given level with it; the fast path the same
    m = _model(noise_level=True)
    cond29 = torch.randn(2, 116, 29)
    fast = ARGenerator(m, max_steps=16, graphs=False)
    for hn in (0.0, 0.15):
        ref = m.generate(cond29, spk, 0.7, torch.Generator().manual_seed(5), history_noise=hn)
        out = fast(cond29, spk, 0.7, torch.Generator().manual_seed(5), history_noise=hn)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    padded = m.pad_frames(cond29)
    mel = m.generate(cond29, spk, temperature=0.0)
    s = m.teacher_forced(padded, m.pad_frames(mel), spk)  # no noise, level 0: what generation runs at
    again = patchify(padded[:, :100], 4) + m.head(s.flatten(0, 1), torch.zeros(16, 400)).view(2, 8, -1)
    torch.testing.assert_close(unpatchify(again, 4, 100)[..., :29], mel, atol=1e-4, rtol=1e-4)


def test_a_rollout_is_free_running_generation_from_the_recorded_history():
    m = _model()
    cond, y, spk = torch.randn(2, 116, 32), torch.randn(2, 100, 32), torch.tensor([0, 2])
    roll = m.rollout(cond, y, spk, 3, 4, 0.7, torch.Generator().manual_seed(2))
    assert roll.shape == (2, 4, 400)
    # the same as generation with the recorded patches before step 3 as a kept prefix
    ref = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(2), prefix=y[..., :12])
    torch.testing.assert_close(unpatchify(roll, 4, 100), ref[..., 12:28], atol=1e-4, rtol=1e-4)
    one = m.rollout(cond, y, spk, 3, 1, 0.0)  # one step: the patch drawn after the recording, as gta
    torch.testing.assert_close(unpatchify(one, 4, 100), m.gta(cond, y, spk, 0.0)[..., 12:16], atol=1e-4, rtol=1e-4)


def test_training_step_with_rollouts_and_the_noise_level():
    m = _model(noise_level=True).train()
    cfg = load_config("configs/ar_tts.yaml", OVER + ["drift.rollout.prob=1.0", "drift.rollout.steps=[2,3]",
                                                     "model.ar.noise_level=true"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    seen = []
    orig = m.teacher_forced
    m.teacher_forced = lambda cond, y, *a, **kw: seen.append(y) or orig(cond, y, *a, **kw)
    b = _batch()
    torch.manual_seed(0)
    loss, metrics = training_step(m, b, cfg, "cpu", tau)
    loss.backward()
    assert torch.isfinite(loss) and metrics["rollout"] in (2.0, 3.0)
    assert m.level_in.net[-1].weight.grad.abs().sum() > 0  # the level is read
    hist, n = seen[-1], int(metrics["rollout"])
    diff = (hist[..., :64] - b["mel"]).abs().sum(1) > 0  # [B, T]: where the history is not the recording
    assert diff[0].sum() == 4 * n and torch.equal(diff[0], diff[1])  # one window of n patches, the same steps
    t0 = int(diff[0].nonzero()[0])
    assert t0 % 4 == 0 and t0 >= 4 and t0 + 4 * n <= 48  # after the first patch, inside the shorter utterance
    off = load_config("configs/ar_tts.yaml", OVER + ["drift.rollout.prob=0.0", "model.ar.noise_level=true"])
    assert training_step(m, b, off, "cpu", tau)[1]["rollout"] == 0.0
    assert "rollout" not in training_step(m, b, load_config("configs/ar_tts.yaml", OVER), "cpu", tau)[1]
