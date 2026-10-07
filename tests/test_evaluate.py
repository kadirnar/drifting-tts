import json

import pytest
import torch

from drifting_tts.cli import main
from drifting_tts.config import load_config
from drifting_tts.evaluate import _plain, format_table, score_utterance, summarize
from drifting_tts.judges import Judges, sv_spec
from drifting_tts.metrics import error_counts
from drifting_tts.models.mae import MelMAE
from drifting_tts.utils import save_checkpoint
from tests.test_data import _fake_parquet

SPK = torch.nn.functional.normalize(torch.ones(4), dim=0)


def test_plain_normalises_like_the_references():
    assert _plain("Merhaba, Dünya! 3 elma; İSTANBUL'a.") == "merhaba dünya üç elma istanbula"
    assert _plain("  ...  ") == ""


def test_error_rates_match_jiwer():
    jiwer = pytest.importorskip("jiwer")
    refs = ["merhaba dünya", "bugün hava çok güzel", "peki ya yarın"]
    hyps = ["merhaba dunya", "bugün hava güzel", "peki yarın ne olacak"]
    s = summarize([{"ref": r, "hyp": h, **error_counts(r, h)} for r, h in zip(refs, hyps)])
    assert s["cer"] == pytest.approx(jiwer.cer(refs, hyps)) and s["wer"] == pytest.approx(jiwer.wer(refs, hyps))


def test_score_utterance_with_fake_judges():
    seen = []

    def asr(w16):
        seen.append(len(w16))
        return "Merhaba, dünya!"

    judges = Judges(asr=asr, sv=lambda w: SPK, mos=lambda w: 3.5)
    row, emb = score_utterance(judges, torch.zeros(24_000), "merhaba dünya", spk_ref=SPK)
    assert seen == [16_000]  # judges get 16 kHz audio
    assert row["hyp"] == "merhaba dünya" and row["char_errors"] == 0 and row["mos"] == 3.5
    assert row["speaker_sim"] == pytest.approx(1.0) and torch.equal(emb, SPK)
    row2, _ = score_utterance(judges, torch.zeros(24_000), "merhaba güzel dünya")  # no speaker reference
    assert "speaker_sim" not in row2 and row2["word_errors"] == 1 and row2["char_errors"] == 6
    s = summarize([row, row2])
    assert s["cer"] == pytest.approx(6 / (13 + 19)) and s["wer"] == pytest.approx(1 / 5)
    assert s["cer_ci"][0] <= s["cer"] <= s["cer_ci"][1]
    assert s["speaker_sim"] == pytest.approx(1.0) and s["mos"] == 3.5
    row3, emb3 = score_utterance(Judges(), torch.zeros(24_000), "x")  # every judge disabled
    assert row3 == {"ref": "x"} and emb3 is None and summarize([row3]) == {}
    assert "| system | cer |" in format_table([{"system": "a", "cer": 0.01}])


def test_speaker_model_aliases():
    assert sv_spec("wavlm-base-plus-sv") == sv_spec("microsoft/wavlm-base-plus-sv")  # old --sv value still works
    assert sv_spec("wavlm-large-ecapa")[2] == "ecapa"
    with pytest.raises(ValueError):
        sv_spec("unknown")


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A 1-step pitch-conditioned model on fake data with F0 and audio; items 0-2 -> val, 3-4 -> dev."""
    tmp = tmp_path_factory.mktemp("eval")
    _fake_parquet(tmp / "x.parquet", n=12)
    data = tmp / "prep"
    main(["prepare", "--parquet-glob", str(tmp / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--f0", "--save-audio"])
    lines = [json.loads(line) for line in open(data / "index.jsonl")]
    for i, e in enumerate(lines):
        e["split"] = "val" if i < 3 else "dev" if i < 5 else "train"
    (data / "index.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in lines))
    mae_cfg = load_config("configs/mae.yaml", ["model.base_channels=8"])
    save_checkpoint(tmp / "mae.pt", ema=MelMAE(n_mels=100, base_channels=8).state_dict(), config=mae_cfg.to_dict(),
                    num_classes=0)
    main(["train", "--workdir", str(tmp / "run"), f"data.root={data}", "data.min_quality=0",
          f"mae.path={tmp / 'mae.pt'}", "model.pitch.enabled=true", "train.cpu=true", "train.steps=1",
          "train.batch_size=3", "train.num_workers=0", "train.log_every=1", "train.save_every=1",
          "train.sample_every=0", "train.warmup=1", "drift.crop_frames=64", "drift.gen_per_cond=2",
          "drift.pos_views=2", "drift.uncond_per_cond=2", "drift.uncond_bank=16", "model.gen.hidden=32",
          "model.gen.depth=1", "model.gen.heads=2", "model.gen.noise_coords=2", "model.text.d=16",
          "model.text.layers=1", "model.text.ffn=32", "model.text.spk_dim=8"])
    return tmp


def test_evaluate_harmonic_cli(tiny_run):
    model = str(tiny_run / "run" / "model_ema.pt")
    results = []
    for k in range(2):
        out = tiny_run / f"harmonic{k}"
        main(["evaluate", "--model", model, "--harmonic", "--num", "3", "--temperature", "0.5", "1.0",
              "--cfg", "1.0", "--out", str(out), "--device", "cpu"])
        results.append(json.loads((out / "results.json").read_text()))
    r = results[0]
    assert r["pitch"] == "ground_truth" and r["voicing"] == "f0 > 0" and r["num_utterances"] == 3
    assert [row["temperature"] for row in r["rows"]] == [0.5, 1.0]
    for row in r["rows"]:
        assert all(row[f"hc_{b}"] > 0 and row[f"gv_{b}"] > 0 for b in ("low", "mid", "high"))
    assert results[0]["rows"] == results[1]["rows"]  # seeded per utterance: reproducible


def test_evaluate_cli_with_fake_judges(tiny_run, monkeypatch):
    import drifting_tts.judges
    import drifting_tts.vocoder

    class FakeVocoder:
        def __init__(self, *args, backend="vocos", **kwargs):
            self.mel, self.name = backend, "fake"

        def __call__(self, mel):
            return 0.1 * torch.sin(0.05 * torch.arange(mel.shape[-1] * 256, dtype=torch.float32))[None]

    fake = Judges(asr=lambda w: "merhaba dünya", sv=lambda w: SPK, mos=lambda w: len(w) / 16_000)
    monkeypatch.setattr(drifting_tts.vocoder, "load_vocoder", FakeVocoder)
    monkeypatch.setattr(drifting_tts.judges, "load_judges", lambda *args, **kwargs: fake)
    out = tiny_run / "full"
    main(["evaluate", "--model", str(tiny_run / "run" / "model_ema.pt"), "--split", "dev", "--num", "5",
          "--temperature", "0.3", "0.7", "--cfg", "1.0", "1.5", "--out", str(out), "--device", "cpu"])
    r = json.loads((out / "results.json").read_text())
    assert r["split"] == "dev" and r["num_utterances"] == 2 and r["speaker_reference"] == "recording"
    systems = [(row["system"], row.get("temperature"), row.get("cfg")) for row in r["rows"]]
    assert systems == [("recording", None, None), ("ground_truth_vocoded", None, None),
                       ("drifting_tts", 0.3, 1.0), ("drifting_tts", 0.3, 1.5),
                       ("drifting_tts", 0.7, 1.0), ("drifting_tts", 0.7, 1.5)]
    rec, gt, syn = r["rows"][0], r["rows"][1], r["rows"][2]
    assert "speaker_sim" not in rec and gt["speaker_sim"] == pytest.approx(1.0)
    assert all(k in syn for k in ("cer", "wer", "speaker_sim", "mos", "rtf", "cer_ci"))
    utts = [json.loads(line) for line in open(out / "utterances_T0.3_cfg1.5.jsonl")]
    assert [u["seed"] for u in utts] == [0, 1] and all(u["hyp"] == "merhaba dünya" for u in utts)
    assert (out / "results.md").exists() and len(list((out / "wav").glob("*.wav"))) == 12


def test_preferred_temperature_from_checkpoint(tiny_run):
    import shutil

    from drifting_tts.synthesize import DEFAULT_TEMPERATURE, preferred_temperature
    from drifting_tts.train import load_tts

    path = tiny_run / "calibrated.pt"
    shutil.copy(tiny_run / "run" / "model_ema.pt", path)
    assert preferred_temperature(load_tts(path)[0]) == DEFAULT_TEMPERATURE == 0.5
    main(["calibrate-durations", "--model", str(path), "--num", "3", "--temperature", "0.4", "--device", "cpu"])
    assert preferred_temperature(load_tts(path)[0]) == 0.4


def test_benchmark_text_sets_and_band_matching(tmp_path):
    import json

    import torch

    from drifting_tts.benchmark import band_match, load_texts

    (tmp_path / "t.txt").write_text("Merhaba.\n\nNasılsın?\n")
    assert [x["text"] for x in load_texts(str(tmp_path / "t.txt"))] == ["Merhaba.", "Nasılsın?"]
    (tmp_path / "t.jsonl").write_text(json.dumps({"id": "0007", "text": "Yarın görüşürüz."}) + "\n")
    assert load_texts(str(tmp_path / "t.jsonl"))[0]["id"] == "0007"
    t = torch.arange(24_000) / 24_000
    tone = torch.sin(2 * torch.pi * 6_000 * t)  # above the 4 kHz Nyquist limit of 8 kHz audio
    full, narrow = band_match(tone, 0), band_match(tone, 8000)
    assert len(full) == len(narrow) == 16_000
    assert (narrow ** 2).mean() < 0.01 * (full ** 2).mean()  # band-matching removes it
