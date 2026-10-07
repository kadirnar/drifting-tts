import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drifting_tts import score
from drifting_tts.cli import main
from drifting_tts.data import MelDataset, load_scores, parse_filters, passes

from .test_data import _fake_parquet


def _prepared(tmp_path, n=30, val=(0, 1, 2)):
    _fake_parquet(tmp_path / "x.parquet", n=n)
    out = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(out), "--workers", "1",
          "--val-size", "0", "--no-trim", "--save-audio"])
    lines = [json.loads(line) for line in open(out / "index.jsonl")]
    for i in val:
        lines[i]["split"] = "val"
    (out / "index.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in lines))
    return out


def _score(out, *extra):
    main(["score", "--data", str(out), "--device", "cpu", "--workers", "0", "--batch-size", "4", *extra])


def test_error_rates(monkeypatch):
    assert score.error_rates("Merhaba, dünya!", "merhaba dünya") == (0.0, 0.0)
    assert score.error_rates("2021'de geldi.", "iki bin yirmi birde geldi") == (0.0, 0.0)  # same normaliser
    cer, wer = score.error_rates("merhaba dünya", "merhaba dunya")
    assert cer == pytest.approx(1 / 13) and wer == pytest.approx(1 / 2)
    assert score.error_rates("merhaba dünya", "") == (1.0, 1.0)
    pairs = [("kitap okudum", "kitabı okudum"), ("", "abc"), ("bir iki üç", "iki üç dört beş")]
    dists = lambda: [score.edit_distance(a, b) for a, b in pairs] + [score.edit_distance(a.split(), b.split())
                                                                      for a, b in pairs]
    expected = dists()
    monkeypatch.setattr(score, "_levenshtein", None)  # the pure-python fallback agrees
    assert dists() == expected == [2, 3, 12, 1, 1, 3]


def _band_limited_noise(cutoff, seconds=3.0, sr=24_000, seed=0):
    x = np.random.default_rng(seed).standard_normal(int(seconds * sr))
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1 / sr) > cutoff] = 0
    return torch.from_numpy(0.1 * np.fft.irfft(spec, len(x))).float()


def test_bandwidth_estimate():
    cutoffs = [3000, 5500, 8000, 11000]
    wavs = [_band_limited_noise(c, seconds=2 + k) for k, c in enumerate(cutoffs)]
    fade = torch.ones(72_000)
    fade[:480], fade[-480:] = torch.linspace(0, 1, 480), torch.linspace(1, 0, 480)  # 20 ms ramps, no clicks
    wavs.append(torch.cat([torch.zeros(24_000), fade * _band_limited_noise(7000), torch.zeros(12_000)]))  # pauses
    est = score.bandwidth_hz(wavs, db=50.0)
    for e, c in zip(est, cutoffs + [7000]):
        assert abs(e - c) < 250, (e, c)
    t = torch.arange(48_000) / 24_000  # a 1 kHz tone with a -70 dB 6 kHz component: below the threshold
    tone = torch.sin(2 * torch.pi * 1000 * t) + 10 ** (-70 / 20) * torch.sin(2 * torch.pi * 6000 * t)
    assert abs(score.bandwidth_hz([tone], db=50.0)[0] - 1000) < 250


def test_speaker_purity():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((3, 64))
    spk = np.repeat(np.arange(3), 10)
    emb = centers[spk] + 0.3 * rng.standard_normal((30, 64))
    emb[4] = centers[2] + 0.3 * rng.standard_normal(64)  # labelled speaker 0, sounds like speaker 2
    sim, nxt, speakers, cent = score.speaker_purity(emb, spk)
    assert sim[4] < 0.5 and nxt[4] > sim[4]
    assert np.delete(sim, 4).min() > 0.8 and np.delete(nxt, 4).max() < 0.5
    assert list(speakers) == [0, 1, 2] and cent.shape == (3, 64)
    single = score.speaker_purity(emb[:2], np.array([0, 1]))[0]  # one utterance per speaker: undefined
    assert np.isnan(single).all()


def test_dnsmos_windows_and_features():
    assert score.DnsMos.windows(np.ones(5 * 16_000, np.float32)).shape == (1, 144_160)  # tiled to 9.01 s
    assert score.DnsMos.windows(np.ones(int(12.3 * 16_000), np.float32)).shape == (3, 144_160)
    librosa = pytest.importorskip("librosa")
    d = score.DnsMos.__new__(score.DnsMos)
    d.fb = torch.from_numpy(librosa.filters.mel(sr=16_000, n_fft=321, n_mels=120))
    d.window = torch.hann_window(321)
    x = np.random.default_rng(0).standard_normal((2, 144_000)).astype(np.float32) * np.array([[0.1], [0.01]],
                                                                                              np.float32)
    ref = [(librosa.power_to_db(librosa.feature.melspectrogram(y=w, sr=16_000, n_fft=321, hop_length=160,
                                                                n_mels=120), ref=np.max) + 40) / 40 for w in x]
    ours = d._p808_features(torch.from_numpy(x)).numpy()
    assert ours.shape == (2, 900, 120)
    np.testing.assert_allclose(ours, np.stack(ref).transpose(0, 2, 1), atol=1e-3)


def test_asr_batching_maps_segments_to_clips():
    calls = []

    def transcribe(audio, clip_timestamps=None, **kw):
        calls.append(clip_timestamps)
        if clip_timestamps is None:  # a long clip: VAD-chunked, may give several segments
            return iter([SimpleNamespace(start=0.0, text=" uzun"), SimpleNamespace(start=29.0, text=" kayıt")]), None
        return iter([SimpleNamespace(start=round(c["start"], 3), text=f" klip {k}")
                     for k, c in enumerate(clip_timestamps)]), None

    asr = score.AsrScorer.__new__(score.AsrScorer)
    asr.batched, asr.batch_size, asr.opts = SimpleNamespace(transcribe=transcribe), 4, {}
    wavs = [np.zeros(n, np.float32) for n in (16_000, 31 * 16_000, 8_000, 24_321)]
    assert asr.transcribe(wavs) == ["klip 0", "uzun kayıt", "klip 1", "klip 2"]
    assert calls[0] is None and len(calls[1]) == 3

    def busy_gpu(audio, clip_timestamps=None, **kw):  # more than 2 clips per decode do not fit
        if len(clip_timestamps) > 2:
            raise RuntimeError("CUDA failed with error out of memory")
        return transcribe(audio, clip_timestamps, **kw)

    asr.batched = SimpleNamespace(transcribe=busy_gpu)
    assert asr.transcribe([np.zeros(16_000, np.float32)] * 4) == ["klip 0", "klip 1"] * 2 and asr.batch_size == 2


def test_filters_parse_and_pass():
    rules = parse_filters({"max_cer": 0.1, "min_mos": 3.0, "rate": [8, None], "apply_to_val": True})
    assert rules == [("cer", -np.inf, 0.1), ("mos_ovrl", 3.0, np.inf), ("cps", 8.0, np.inf)]
    assert passes({"cer": 0.05, "mos_ovrl": 3.5, "cps": 12}, rules)
    assert not passes({"cer": 0.2}, rules) and not passes({"cps": 5}, rules)
    assert passes({}, rules) and passes(None, rules)  # unscored: kept
    with pytest.raises(ValueError):
        parse_filters({"cer": 0.1})


def test_score_filter_and_resume(tmp_path, capsys, monkeypatch):
    out = _prepared(tmp_path)
    filters = {"max_cer": 0.1, "rate": [1, 100]}
    assert len(MelDataset(out, "train", min_frames=1, filters=filters)) == 27  # no scores.jsonl: nothing dropped
    assert "nothing filtered" in capsys.readouterr().out

    seen = []
    bw = score.bandwidth_hz
    monkeypatch.setattr(score, "bandwidth_hz", lambda wavs, db: seen.extend(len(w) for w in wavs) or bw(wavs, db))
    _score(out, "--scorers", "rate,bandwidth", "--limit", "10")
    assert len(load_scores(out)) == 10 and len(seen) == 10
    with open(out / "scores.jsonl", "a") as f:  # an interrupted run: a truncated last line
        f.write('{"i": 3, "cer": 0.')
    _score(out, "--scorers", "rate,bandwidth")
    assert len(seen) == 30  # only the 20 new utterances were scored again
    lines = [json.loads(line) for line in open(out / "scores.jsonl")]
    assert [r["i"] for r in lines] == list(range(30))  # compacted: one line per utterance
    s = load_scores(out)
    assert all(abs(r["bw_hz"] - 220 - 10 * i) < 100 for i, r in s.items())  # sine tones: 200 + 10 i Hz
    e = json.loads(open(out / "index.jsonl").readline())  # "merhaba dünya sıfır." over 1 s
    assert s[0]["cps"] == round(17 / (e["frames"] / 93.75), 2) and 16 < s[0]["cps"] < 18

    # fake ASR scores: utterances 1 (val) and 5, 6 (train) mismatch their transcripts
    with open(out / "scores.jsonl", "a") as f:
        for i in range(30):
            f.write(json.dumps({"i": i, "cer": 0.5 if i in (1, 5, 6) else 0.02}) + "\n")
    capsys.readouterr()
    train = MelDataset(out, "train", min_frames=1, filters=filters)
    assert len(train) == 25 and "2/27 utterances fail; kept 25" in capsys.readouterr().out
    val = MelDataset(out, "val", min_frames=1, filters=filters)
    assert len(val) == 3 and "1/3 utterances fail; not filtered" in capsys.readouterr().out
    assert len(MelDataset(out, "val", min_frames=1, filters={**filters, "apply_to_val": True})) == 2
    assert len(MelDataset(out, "train", min_frames=1, filters={"rate": [12.5, 20]})) < 27

    _score(out, "--summary", "--filter", "max_cer=0.1", "--speakers-tsv", str(tmp_path / "spk.tsv"))
    text = capsys.readouterr().out
    assert "train: 25/27 utterances" in text and "val: 1/3 would fail" in text
    assert len(open(tmp_path / "spk.tsv").readlines()) == 4
