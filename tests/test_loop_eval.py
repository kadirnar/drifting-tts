"""Evaluation inside the training loop and several EMAs (#102): the helpers, and a tiny ARDrift run on the CPU."""

import json

import pytest
import torch

from drifting_tts.config import Config
from drifting_tts.loop_eval import AudioEval, build_emas, fixed_rng, heldout_metrics, load_emas, other_emas, record

TINY = ["data.filters={}", "train.cpu=true", "train.batch_size=3", "train.num_workers=0", "train.log_every=1",
        "train.save_every=2", "train.sample_every=0", "train.warmup=1", "drift.gen_per_cond=3", "drift.max_rows=16",
        "model.ar.dim=32", "model.ar.depth=1", "model.ar.heads=2", "model.ar.head_dim=32", "model.ar.head_depth=1",
        "model.text.d=16", "model.text.layers=1", "model.text.ffn=32", "model.text.spk_dim=8"]


def test_fixed_rng_gives_the_streams_back_and_heldout_metrics_repeat():
    torch.manual_seed(0)
    a = torch.rand(3)
    torch.manual_seed(0)
    with fixed_rng(7):
        inside = torch.rand(3)
    assert torch.equal(torch.rand(3), a) and not torch.equal(inside, a)  # the block did not move the stream
    step = lambda b: {"x": torch.rand(()) + b["v"], "y": 2.0}  # noqa: E731
    batches = [{"v": 1.0}, {"v": 3.0}]
    first, again = heldout_metrics(step, batches), heldout_metrics(step, batches)
    assert first == again and first["y"] == 2.0 and 2.0 < first["x"] < 3.0  # the same noise at every evaluation


def test_emas_by_decay_and_their_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 2)
    emas = build_emas(model, Config({"ema_decay": 0.9, "ema_decays": [0.5, 0.9999]}))
    assert list(emas) == ["0.5", "0.9999"] and list(build_emas(model, Config({"ema_decay": 0.9}))) == ["0.9"]
    with pytest.raises(ValueError, match="twice"):
        build_emas(model, Config({"ema_decay": 0.9, "ema_decays": [0.9, 0.9]}))
    with torch.no_grad():
        model.weight.add_(1.0)
    for e in emas.values():
        e.update(model)
    fast, slow = (e.model.weight for e in emas.values())
    assert (fast - model.weight).abs().max() < (slow - model.weight).abs().max()  # the slow one lags
    ck = {"ema": emas["0.5"].model.state_dict(), "emas": other_emas(emas)}
    assert list(ck["emas"]) == ["0.9999"]
    fresh = build_emas(torch.nn.Linear(2, 2), Config({"ema_decay": 0.9, "ema_decays": [0.5, 0.9999, 0.99]}))
    load_emas(fresh, ck)
    assert torch.equal(fresh["0.9999"].model.weight, slow) and torch.equal(fresh["0.5"].model.weight, fast)
    assert torch.equal(fresh["0.99"].model.weight, fast)  # not in the checkpoint: it starts from the main one

    class Writer:
        tags: list = []

        def add_scalar(self, tag, value, step):
            self.tags.append(tag)

    record(Writer(), tmp_path, 10, "0.5", True, {"wer": 0.5, "loss/patch": 1.0})
    record(Writer(), tmp_path, 10, "0.9999", False, {"wer": 0.4})
    record(Writer(), tmp_path, 10, "0.5", True, {"wer": 0.3, "loss/patch": 1.0})  # a resumed run repeats the step
    history = json.loads((tmp_path / "eval_history.json").read_text())
    assert [(r["ema"], r["wer"]) for r in history] == [("0.9999", 0.4), ("0.5", 0.3)]
    assert {"eval/wer", "eval/loss/patch", "eval_0.9999/wer"} == set(Writer.tags)


def test_train_ar_evaluates_in_the_loop_and_keeps_two_emas(tmp_path):
    from drifting_tts.cli import main
    from drifting_tts.data import MelDataset
    from drifting_tts.judges import Judges
    from drifting_tts.train import load_tts
    from drifting_tts.vocoder import load_vocoder
    from tests.test_data import _fake_parquet

    _fake_parquet(tmp_path / "x.parquet", n=14)
    _fake_parquet(tmp_path / "held.parquet", n=14)  # a held-out file: its four utterances of 2 s or more are val
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "x.parquet"), "--val-glob", str(tmp_path / "held.parquet"),
          "--out", str(data), "--workers", "1", "--val-size", "4", "--no-trim", "--backend", "bigvgan"])
    plain, work = tmp_path / "plain", tmp_path / "eval"
    main(["train-ar", "--workdir", str(plain), f"data.root={data}", *TINY, "train.steps=2", "train.ema_decay=0.9"])
    quiet = ["train.ema_decays=[0.9,0.99]", "train.vocoder=griffin-lim", "train.eval.num=2", "train.eval.asr=none",
             "train.eval.mos=none", "train.eval.f0=false"]
    main(["train-ar", "--workdir", str(work), f"data.root={data}", *TINY, *quiet, "train.steps=2",
          "train.eval_every=1", "train.eval.batches=2", "train.eval.batch_size=2", "train.eval.at_start=true"])
    history = json.loads((work / "eval_history.json").read_text())
    assert [(r["step"], r["ema"]) for r in history] == [(s, e) for s in (0, 1, 2) for e in ("0.9", "0.99")]
    row = history[0]
    assert {"loss", "drift", "prior", "duration", "p_data", "sample_std", "pauses", "length_ratio"} <= set(row)
    assert {"loss/patch", "force/frame", "p_data/detail"} <= set(row) and "wer" not in row  # no judge was asked for
    assert history[0]["loss"] == history[1]["loss"]  # before the first step the two EMAs are the same model
    assert history[4]["loss"] != history[5]["loss"]
    # the evaluation does not move the training: the same weights as the run without it
    a, b = (torch.load(w / "last.pt", weights_only=False) for w in (plain, work))
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert all(torch.equal(a["ema"][k], b["ema"][k]) for k in a["ema"]) and list(b["emas"]) == ["0.99"]
    main_ema, slow = load_tts(work / "model_ema.pt")[0], load_tts(work / "model_ema_0.99.pt")[0]
    assert not torch.equal(main_ema.head.out.weight, slow.head.out.weight)

    main(["train-ar", "--workdir", str(work), f"data.root={data}", *TINY, *quiet, "train.steps=3",
          "train.eval_every=3", "train.eval.batches=1"])  # resumed: both EMAs go on
    history = json.loads((work / "eval_history.json").read_text())
    assert [(r["step"], r["ema"]) for r in history][-2:] == [(3, "0.9"), (3, "0.99")]
    after = torch.load(work / "last.pt", weights_only=False)
    assert not torch.equal(after["emas"]["0.99"]["head.out.weight"], b["emas"]["0.99"]["head.out.weight"])

    # the judges' numbers, with stand-ins for Whisper and the MOS predictor
    ds = MelDataset(data, "val", min_frames=1, max_frames=10**9)
    vocoder = load_vocoder("griffin-lim", "cpu")
    vocoder.model.n_iter = 2

    def synthesize(m, item, g):
        text = item["text"][None]
        return m.synthesize(text, torch.tensor([text.shape[1]]), torch.tensor([item["spk"]]), temperature=0.5,
                            generator=g)[0]

    audio = AudioEval(ds, vocoder, "cpu", synthesize, num=2, f0=True)
    audio.judges = Judges(asr=lambda w: "merhaba", sv=None, mos=lambda w: 3.5)
    res = audio(main_ema)
    assert res["mos"] == 3.5 and res["wer"] > 0 and res["length_ratio"] > 0 and audio(main_ema) == res


def test_train_evaluates_the_dit_in_the_loop(tmp_path):
    """The DiT's loop: the same EMAs and judged sentences (its objective has a memory bank, so it is not replayed on
    held-out batches)."""
    from drifting_tts.cli import main
    from drifting_tts.config import load_config
    from drifting_tts.models.mae import MelMAE
    from drifting_tts.utils import save_checkpoint
    from tests.test_data import _fake_parquet
    from tests.test_train import _tiny_train_args

    _fake_parquet(tmp_path / "x.parquet", n=14)
    _fake_parquet(tmp_path / "held.parquet", n=14)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "x.parquet"), "--val-glob", str(tmp_path / "held.parquet"),
          "--out", str(data), "--workers", "1", "--val-size", "4", "--no-trim", "--backend", "bigvgan"])
    mae = MelMAE(n_mels=100, base_channels=8, layers=(2, 2, 2, 2))
    save_checkpoint(tmp_path / "mae.pt", ema=mae.state_dict(), num_classes=0,
                    config=load_config("configs/mae.yaml", ["model.base_channels=8"]).to_dict())
    plain, work = tmp_path / "plain", tmp_path / "eval"
    common = ["train.cpu=true", "train.steps=2", "train.save_every=2"]
    main(_tiny_train_args(plain, data, tmp_path / "mae.pt", *common, "train.ema_decay=0.9"))
    main(_tiny_train_args(work, data, tmp_path / "mae.pt", *common, "train.ema_decays=[0.9,0.99]",
                          "train.eval_every=2", "train.eval.vocoder=griffin-lim", "train.eval.num=2",
                          "train.eval.asr=none", "train.eval.mos=none", "train.eval.f0=false",
                          "train.eval.at_start=true"))
    history = json.loads((work / "eval_history.json").read_text())
    assert [(r["step"], r["ema"]) for r in history] == [(s, e) for s in (0, 2) for e in ("0.9", "0.99")]
    assert {"pauses", "length_ratio"} <= set(history[0])
    a, b = (torch.load(w / "last.pt", weights_only=False) for w in (plain, work))
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])  # the training was not moved
    assert list(b["emas"]) == ["0.99"] and (work / "model_ema_0.99.pt").exists()
