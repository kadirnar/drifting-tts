"""ARDrift v2 (#103): the text as a prefix, the end head, the training step and ``train-ar`` on it. Tiny models on
the CPU."""

import json

import pytest
import torch

from drifting_tts.config import Config, load_config
from drifting_tts.models.ar import patchify
from drifting_tts.models.ar_prefix import ARPrefixTTS, PreNet, characters, eos_loss, front_pad
from drifting_tts.text import BLANK_ID, text_to_ids
from drifting_tts.train_ar_prefix import training_step
from tests.test_prompt import _batch

CFG = {"kind": "ar_prefix", "text": {},
       "ar": {"patch": 4, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32, "head_depth": 2, "prompt": True}}
OVER = ["model.kind=ar_prefix", "model.ar.prompt=true", "drift.gen_per_cond=4", "drift.max_rows=24"]


def _model(seed: int = 0, **ar) -> ARPrefixTTS:
    torch.manual_seed(seed)
    m = ARPrefixTTS(Config({**CFG, "ar": {**CFG["ar"], **ar}}), num_speakers=3, n_mels=100).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def test_characters_and_front_padding():
    ids = torch.tensor([text_to_ids("ab c", normalized=True), text_to_ids("d", normalized=True) + [0] * 6])
    chars, n = characters(ids, torch.tensor([9, 3]))
    assert chars.shape == (2, 4) and n.tolist() == [4, 1] and BLANK_ID not in chars[0].tolist()
    assert torch.equal(characters(chars, n)[0], chars)  # already characters: unchanged
    out, mask = front_pad([torch.ones(2, 3), torch.ones(4, 3) * 2, torch.zeros(0, 3)], 3)
    assert out.shape == (3, 4, 3) and mask.tolist() == [[False, False, True, True], [True] * 4, [False] * 4]
    assert out[0, :2].abs().sum() == 0 and out[0, 2:].eq(1).all()


def test_a_row_does_not_see_the_padding_or_the_future():
    m = _model()
    text, n = torch.randint(3, 20, (2, 7)), torch.tensor([7, 4])
    y, spk = torch.randn(2, 100, 16), torch.tensor([0, 1])
    s = m.teacher_forced(text, n, y, spk)
    assert s.shape == (2, 5, 32)  # four patches and the step after them
    alone = m.teacher_forced(text[1:, :4], n[1:], y[1:], spk[1:])
    torch.testing.assert_close(s[1], alone[0], atol=1e-5, rtol=1e-5)  # the shorter text, padded in front
    later = y.clone()
    later[..., 8:] += 1.0
    s2 = m.teacher_forced(text, n, later, spk)
    torch.testing.assert_close(s2[:, :3], s[:, :3], atol=1e-5, rtol=1e-5)  # causal in the audio
    assert not torch.allclose(s2[:, 3:], s[:, 3:], atol=1e-4)
    other = m.teacher_forced(text.flip(1), n, y, spk)
    assert not torch.allclose(other[0], s[0], atol=1e-4)  # the text is read
    prompt = torch.randn(2, 100, 12)
    with_voice = m.teacher_forced(text, n, y, spk, prompt=prompt, prompt_len=torch.tensor([12, 0]))
    assert not torch.allclose(with_voice[0], s[0], atol=1e-4)
    torch.testing.assert_close(with_voice[1], s[1], atol=1e-5, rtol=1e-5)  # no prompt: as without one
    dropped = m.teacher_forced(text, n, y, spk, drop=torch.tensor([True, False]))
    assert not torch.allclose(dropped[0], s[0], atol=1e-4)
    torch.testing.assert_close(dropped[1], s[1], atol=1e-5, rtol=1e-5)


def test_generation_matches_teacher_forcing_and_stops_at_the_end():
    m = _model()
    text, spk = torch.randint(3, 20, (1, 6)), torch.tensor([2])
    mel = m.generate(text, spk, 0.0, max_steps=5, min_steps=5)
    assert mel.shape == (1, 100, 20)
    s = m.teacher_forced(text, torch.tensor([6]), mel, spk)
    again = m.head(s[0, :5], torch.zeros(5, 400))
    torch.testing.assert_close(again, patchify(mel, 4)[0], atol=1e-4, rtol=1e-4)
    g = [torch.Generator().manual_seed(3) for _ in range(2)]
    assert torch.equal(m.generate(text, spk, 0.7, g[0], max_steps=5, min_steps=5),
                       m.generate(text, spk, 0.7, g[1], max_steps=5, min_steps=5))
    with torch.no_grad():
        m.eos.bias.fill_(20.0)  # the end head says "over" at once: generation stops after min_steps
    assert m.generate(text, spk, 0.5, max_steps=50, min_steps=3).shape[-1] == 12
    with torch.no_grad():
        m.eos.bias.fill_(-20.0)  # ... or never: the bound on the length holds
    assert m.generate(text, spk, 0.5, max_steps=7).shape[-1] == 28
    assert 0 < m.generate(text, spk, 0.5).shape[-1] <= int(6 * m.max_rate * 93.75 / 4) * 4
    with pytest.raises(ValueError, match="one utterance"):
        m.generate(torch.randint(3, 20, (2, 6)), torch.tensor([0, 1]))
    blanks = torch.tensor([text_to_ids("ab cd", normalized=True)])
    mel, n = m.synthesize(blanks, torch.tensor([blanks.shape[1]]), spk, temperature=0.5,
                          generator=torch.Generator().manual_seed(0))
    assert mel.shape[-1] == int(n) and torch.isfinite(mel).all()
    voiced, _ = m.synthesize(blanks, torch.tensor([blanks.shape[1]]), spk, temperature=0.5,
                             generator=torch.Generator().manual_seed(0), prompt=torch.randn(1, 100, 16))
    assert voiced.shape != mel.shape or not torch.allclose(voiced, mel, atol=1e-4)


def test_prenet_is_a_bottleneck_with_dropout_at_inference():
    m = _model(prenet={"dim": 16, "dropout": 0.5})
    assert isinstance(m.x_in, PreNet) and m.x_in.layers[0].out_features == 16 and not m.training
    text, n = torch.randint(3, 20, (1, 7)), torch.tensor([7])
    y, spk = torch.randn(1, 100, 16), torch.tensor([0])
    a, b = m.teacher_forced(text, n, y, spk), m.teacher_forced(text, n, y, spk)
    assert a.shape == (1, 5, 32) and not torch.allclose(a, b)  # the dropout is on in eval mode
    torch.manual_seed(0)
    c = m.teacher_forced(text, n, y, spk)
    torch.manual_seed(0)
    assert torch.allclose(c, m.teacher_forced(text, n, y, spk))
    m.x_in.dropout = 0.0
    assert torch.allclose(m.teacher_forced(text, n, y, spk), m.teacher_forced(text, n, y, spk))
    mel = m.generate(text, spk, max_steps=6, generator=torch.Generator().manual_seed(0))
    assert mel.shape[:2] == (1, 100) and mel.shape[-1] % 4 == 0


def test_the_clock_lays_the_text_over_the_audio():
    m = _model(clock={"rate": 1.5, "lead": 4.0})
    assert m.clock and m.rate == 1.5 and m.lead == 4.0 and m.fit == "utterance" and not _model().clock
    text, n = torch.randint(3, 20, (2, 7)), torch.tensor([7, 4])
    spk, rate = torch.tensor([0, 1]), torch.tensor([2.0, 1.0])
    prompt, plen = torch.randn(2, 100, 12), torch.tensor([12, 0])
    out, mask, pos = m.prefix(text, n, spk, prompt, plen, rate=rate)
    assert out.shape[:2] == (2, 10) and mask.sum(1).tolist() == [10, 4]
    # the voice: three patches before the gap; the characters: (j + 0.5) * rate - lead
    assert pos[0, :3].tolist() == [-19.0, -18.0, -17.0] and pos[0, 3:].tolist() == [j * 2.0 - 3.0 for j in range(7)]
    assert pos[1, 6:].tolist() == [j - 3.5 for j in range(4)]
    assert m.prefix(text, n, spk)[2][1, 3:].tolist() == [(j + 0.5) * 1.5 - 4.0 for j in range(4)]  # the default rate
    assert _model().prefix(text, n, spk)[2] is None
    # a row alone and in a batch: the same states; the rate changes them
    y = torch.randn(2, 100, 16)
    s = m.teacher_forced(text, n, y, spk, rate=rate)
    alone = m.teacher_forced(text[1:, :4], n[1:], y[1:], spk[1:], rate=rate[1:])
    torch.testing.assert_close(s[1], alone[0], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(m.teacher_forced(text, n, y, spk, rate=rate * 1.5)[0], s[0], atol=1e-4)
    later = y.clone()
    later[..., 8:] += 1.0
    torch.testing.assert_close(m.teacher_forced(text, n, later, spk, rate=rate)[:, :3], s[:, :3], atol=1e-5, rtol=1e-5)
    # generation with a cache stands on the same clock as teacher forcing
    one, spk1 = text[:1, :6], torch.tensor([2])
    for r in (None, 2.5):
        mel = m.generate(one, spk1, 0.0, max_steps=5, min_steps=5, rate=r)
        tf = m.teacher_forced(one, torch.tensor([6]), mel, spk1, rate=None if r is None else torch.tensor([r]))
        torch.testing.assert_close(m.head(tf[0, :5], torch.zeros(5, 400)), patchify(mel, 4)[0], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(m.generate(one, spk1, 0.0, max_steps=5, min_steps=5, rate=1.0), mel, atol=1e-4)  # the tempo
    voiced = m.generate(one, spk1, 0.0, max_steps=5, min_steps=5, prompt=torch.randn(1, 100, 16))
    assert voiced.shape == (1, 100, 20)
    with torch.no_grad():
        m.eos.bias.fill_(-20.0)  # the end head silent: the utterance is as long as its text at the tempo
    assert m.generate(one, spk1, 0.5).shape[-1] == 9 * 4 and m.generate(one, spk1, 0.5, rate=2.5).shape[-1] == 15 * 4
    # the training step lays every utterance out at its own rate
    cfg = load_config("configs/ar_tts.yaml", OVER + ["drift.cond_dropout=0.5"])
    seen = {}
    tf0 = m.teacher_forced

    def spy(*a, **kw):
        seen["rate"] = a[8] if len(a) > 8 else kw.get("rate")
        return tf0(*a, **kw)

    m.teacher_forced = spy
    b = _batch()
    loss, _ = training_step(m.train(), b, cfg, "cpu", torch.nn.Parameter(torch.tensor(1.0)))
    chars, cn = characters(b["text"], b["text_len"])
    torch.testing.assert_close(seen["rate"], -(-b["mel_len"] // 4).float() / cn)
    assert torch.isfinite(loss)
    m.fit = "global"
    seen.clear()
    training_step(m, _batch(), cfg, "cpu", torch.nn.Parameter(torch.tensor(1.0)))
    assert seen["rate"] is None


def test_eos_loss_marks_the_step_after_the_last_patch():
    logits = torch.full((2, 5), -10.0)
    logits[0, 4], logits[1, 2] = 10.0, 10.0
    loss, acc = eos_loss(logits, torch.tensor([4, 2]))
    assert float(loss) < 1e-3 and float(acc) == 1.0  # the steps after utterance 2's end do not count
    wrong, acc = eos_loss(-logits, torch.tensor([4, 2]))
    assert float(wrong) > 5 and float(acc) == 0.0


def test_training_step_of_the_prefix_model():
    m = _model().train()
    cfg = load_config("configs/ar_tts.yaml", OVER + ["drift.cond_dropout=0.5", "drift.reference.prob=1.0",
                                                     "drift.reference.id_dropout=1.0"])
    tau = torch.nn.Parameter(torch.tensor(1.0))
    b = _batch()
    b["ref"], b["ref_len"] = torch.randn(2, 100, 24), torch.tensor([24, 0])
    torch.manual_seed(0)
    loss, metrics = training_step(m, b, cfg, "cpu", tau, detail=True)
    loss.backward()
    assert torch.isfinite(loss) and {"eos", "eos_acc", "p_data", "loss/patch"} <= set(metrics)
    assert metrics["prompted"] == 0.5 and metrics["id_dropped"] == 0.5 and metrics["rows"] == 24
    # one of the two utterances lost its text at this seed: the gap of the errors is measured on this step
    assert metrics["null_steps"] in (0.0, 1.0) and (metrics["null_steps"] == 1.0) == (float(metrics["text_gap"]) != 0)
    none = load_config("configs/ar_tts.yaml", OVER + ["drift.cond_dropout=0.0"])
    assert training_step(m, _batch(), none, "cpu", tau)[1]["null_steps"] == 0.0
    for p in (m.tok.weight, m.head.out.weight, m.eos.weight, m.x_in.weight, m.seg):
        assert p.grad is not None and p.grad.abs().sum() > 0
    grad = m.spk.weight.grad  # the prompted utterance reads the prompted row, the other its own
    assert grad[m.prompt_row].abs().sum() > 0 and grad[1].abs().sum() > 0 and not grad[0].any()
    energy = load_config("configs/ar_tts.yaml", OVER + ["drift.loss=energy"])
    assert "spread" in training_step(m, _batch(), energy, "cpu", tau)[1]
    b = _batch()
    b["spk"] = torch.tensor([-1, 1])  # a voice without a row and without a prompt: no voice
    assert torch.isfinite(training_step(m, b, energy, "cpu", tau)[0])
    with pytest.raises(ValueError, match="prompts"):
        training_step(_model(prompt=False), b, energy, "cpu", tau)


def test_train_ar_trains_the_prefix_model(tmp_path):
    from drifting_tts.cli import main
    from drifting_tts.train import load_tts
    from tests.test_data import _fake_parquet
    from tests.test_loop_eval import TINY

    _fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    _fake_parquet(tmp_path / "held.parquet", n=14)  # a held-out file: its utterances of 2 s or more are val
    main(["prepare", "--parquet-glob", str(tmp_path / "x.parquet"), "--val-glob", str(tmp_path / "held.parquet"),
          "--out", str(data), "--workers", "1", "--val-size", "2", "--no-trim", "--backend", "bigvgan"])
    work = tmp_path / "run"
    quiet = ["train.vocoder=griffin-lim", "train.eval.num=2", "train.eval.asr=none", "train.eval.mos=none",
             "train.eval.f0=false", "train.eval.batches=1", "train.eval.batch_size=2"]
    main(["train-ar", "--workdir", str(work), f"data.root={data}", *TINY, *OVER[:2], *quiet, "train.steps=2",
          "train.eval_every=2", "drift.reference.prob=1.0", "drift.reference.min_seconds=0.2",
          "drift.reference.max_seconds=0.6"])
    model, cfg, _ = load_tts(work / "model_ema.pt")
    assert isinstance(model, ARPrefixTTS) and model.prompted and cfg.model.kind == "ar_prefix"
    history = json.loads((work / "eval_history.json").read_text())  # the objective and synthesised sentences
    assert history[-1]["step"] == 2 and {"eos", "eos_acc", "length_ratio"} <= set(history[-1])
    text = torch.tensor([text_to_ids("merhaba", normalized=True)])
    mel, n = model.synthesize(text, torch.tensor([text.shape[1]]), torch.tensor([0]), temperature=0.5)
    assert mel.shape[1] == 100 and mel.shape[-1] == int(n)
