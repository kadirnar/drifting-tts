import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from drifting_tts.audio import FRAME_RATE, N_MELS, LogMel
from drifting_tts.cli import main
from drifting_tts.data import BucketBatchSampler, MelDataset, collate


def _wav_bytes(seconds: float, sr: int = 44_100, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    wav = 0.3 * np.sin(2 * np.pi * freq * t)[:, None].repeat(2, 1)  # stereo
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="FLAC")
    return buf.getvalue()


def _fake_parquet(path, n=30):
    rows = {
        "audio": [{"bytes": _wav_bytes(1.0 + 0.1 * i, freq=200 + 10 * i), "path": f"{i}.flac"} for i in range(n)],
        "text": [f"Merhaba dünya {i}." for i in range(n)],
        "duration_seconds": [1.0 + 0.1 * i for i in range(n)],
        "speaker": [f"spk_{i % 3}" for i in range(n)],
        "quality_score": [50.0 + i for i in range(n)],
    }
    pq.write_table(pa.table(rows), path)


def test_logmel_shape():
    mel = LogMel()(torch.zeros(24_000))
    assert mel.shape == (N_MELS, int(24_000 // 256) + 1)


def test_prepare_and_dataset(tmp_path):
    _fake_parquet(tmp_path / "x.parquet")
    out = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(out), "--workers", "2",
          "--val-size", "0", "--no-trim"])
    stats = json.loads((out / "stats.json").read_text())
    assert stats["utterances"] == 30 and stats["speakers"] == 3
    lines = [json.loads(line) for line in open(out / "index.jsonl")]
    assert abs(lines[0]["frames"] - FRAME_RATE * 1.0) < 3
    assert lines[3]["norm_text"] == "merhaba dünya üç."

    ds = MelDataset(out, "train", min_quality=60, min_frames=10)
    assert len(ds) == 20
    item = ds[0]
    assert item["mel"].shape[0] == N_MELS and item["text"][0] == 1
    sampler = BucketBatchSampler([ds.frames(i) for i in range(len(ds))], max_frames=1000, max_batch=4)
    batches = list(iter(sampler))
    assert all(len(b) <= 4 for b in batches)
    batch = collate([ds[i] for i in batches[0]])
    assert batch["mel"].shape[0] == len(batches[0]) and batch["mel"].shape[1] == N_MELS
    assert batch["mel_len"].max() == batch["mel"].shape[2]


def test_bigvgan_logmel_frames_and_prepare_backend(tmp_path):
    from drifting_tts.audio import BigVGANLogMel, make_logmel

    mel = BigVGANLogMel()(torch.zeros(2, 24_000))
    assert mel.shape == (2, N_MELS, 24_000 // 256) and torch.isfinite(mel).all()
    assert abs(mel.min().item() - torch.log(torch.tensor(1e-5)).item()) < 1e-3  # silence -> log(1e-5) floor
    assert isinstance(make_logmel("vocos"), LogMel)

    _fake_parquet(tmp_path / "x.parquet", n=6)
    out = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(out), "--workers", "1",
          "--val-size", "0", "--no-trim", "--backend", "bigvgan"])
    assert json.loads((out / "stats.json").read_text())["backend"] == "bigvgan"
    assert MelDataset(out, "train", min_frames=1).backend == "bigvgan"


def test_f0_is_sampled_at_the_mel_frame_centres_of_each_backend():
    import numpy as np

    from drifting_tts.audio import HOP_LENGTH, SAMPLE_RATE, extract_f0

    t = np.arange(int(1.5 * SAMPLE_RATE)) / SAMPLE_RATE
    freq = 120.0 + 200.0 * t  # steep harmonic chirp: half a hop (5.3 ms) is a ~1 Hz pitch offset
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    wav = 0.3 * sum(np.sin(k * phase) / k for k in range(1, 6))
    frames = len(t) // HOP_LENGTH
    i = np.arange(15, frames - 15)
    for backend, centre in (("vocos", 0.0), ("bigvgan", 0.5)):
        for method in ("dio", "harvest"):
            f0 = extract_f0(wav, frames, method=method, backend=backend)
            expected = 120.0 + 200.0 * (i + centre) * HOP_LENGTH / SAMPLE_RATE
            assert np.abs(f0[i] - expected).mean() < 0.2, (backend, method)


def test_dev_split_is_disjoint_and_leaves_val_unchanged():
    from drifting_tts.prepare import choose_validation

    index = [{"speaker": f"s{i % 4}", "quality_score": 80, "audio_seconds": 5.0} for i in range(200)]
    index += [{"speaker": "rare", "quality_score": 90, "audio_seconds": 5.0}] * 5  # < 20 utterances: never held out
    val = choose_validation(index, 20, seed=0)
    dev = choose_validation(index, 30, seed=1, exclude=val)
    assert len(val) == 20 and len(dev) == 30 and not val & dev and max(val | dev) < 200
    assert choose_validation(index, 20, seed=0) == val  # adding a dev split does not move val


def test_merge_data_keeps_the_base_and_appends_new_speakers(tmp_path):
    import json

    import numpy as np

    from drifting_tts.cli import main

    def fake(root, speakers, frames, split="train"):
        root.mkdir()
        index, off, aoff = [], 0, 0
        for i, (spk, n) in enumerate(zip(speakers, frames)):
            index.append({"speaker": spk, "spk_id": sorted(set(speakers)).index(spk), "offset": off, "frames": n,
                          "audio_offset": aoff, "audio_samples": n * 256, "split": split if i == 0 else "train",
                          "norm_text": "a", "quality_score": None})
            off, aoff = off + n, aoff + n * 256
        (root / "index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in index))
        (root / "speakers.json").write_text(json.dumps({s: i for i, s in enumerate(sorted(set(speakers)))}))
        (root / "stats.json").write_text(json.dumps({"backend": "bigvgan", "mean": float(len(root.name)), "std": 1.0}))
        np.arange(off * 100, dtype=np.float16).tofile(root / "mels.bin")
        np.arange(off, dtype=np.float16).tofile(root / "f0.bin")
        np.arange(aoff, dtype=np.float16).tofile(root / "audio.bin")

    fake(tmp_path / "base", ["a", "b"], [3, 4], split="val")
    fake(tmp_path / "newvoice", ["new", "new"], [5, 2], split="val")
    out = tmp_path / "merged"
    main(["merge-data", str(tmp_path / "base"), str(tmp_path / "newvoice"), "--out", str(out), "--repeat", "1", "2"])
    idx = [json.loads(line) for line in open(out / "index.jsonl")]
    assert json.loads((out / "speakers.json").read_text()) == {"a": 0, "b": 1, "new": 2}
    assert json.loads((out / "stats.json").read_text())["mean"] == 4.0  # base normalisation kept ("base")
    assert len(idx) == 2 + 2 + 1  # the new voice's training utterance is listed twice, its held-out one once
    mels = np.fromfile(out / "mels.bin", dtype=np.float16).reshape(-1, 100)
    f0, audio = np.fromfile(out / "f0.bin", dtype=np.float16), np.fromfile(out / "audio.bin", dtype=np.float16)
    new = [e for e in idx if e["speaker"] == "new"][0]
    assert new["spk_id"] == 2 and new["offset"] == 7 and new["audio_offset"] == 7 * 256
    assert mels.shape[0] == f0.shape[0] == 14 and audio.shape[0] == 14 * 256
    assert mels[new["offset"], 0] == 0 and f0[new["offset"]] == 0  # first frame of the new voice's data
