"""restore-audio (#71): Sidon's recipe around stand-in networks (the real ones are 1 GB of TorchScript), and a shard
rewritten with the other columns unchanged."""

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from drifting_tts.restore import RATE, Sidon, restore_shard


class Features:
    """Stand-in for the w2v-BERT 2.0 filterbanks: one 160-dim frame per 20 ms (320 samples at 16 kHz)."""

    def __call__(self, wav, sampling_rate=16_000, return_tensors="pt"):
        n = wav.shape[-1] // 320
        return {"input_features": torch.zeros(1, n, 160)}


class FeatureNet(torch.nn.Module):
    def forward(self, x):
        return {"last_hidden_state": torch.ones(x.shape[0], x.shape[1], 8)}


class Decoder(torch.nn.Module):
    """960 samples at 48 kHz per 20 ms feature frame, as Sidon's decoder."""

    def forward(self, f):
        return torch.full((1, 1, f.shape[-1] * 960), 0.25)


def _sidon():
    return Sidon("cpu", fe=FeatureNet(), decoder=Decoder(), preprocessor=Features())


def test_restore_returns_48k_audio_of_the_input_length():
    for sr, seconds in ((16_000, 1.3), (24_000, 2.0)):
        out = _sidon().restore(np.random.default_rng(0).standard_normal(int(sr * seconds)).astype(np.float32), sr)
        assert out.dtype == np.float32 and out.shape == (int(RATE * seconds),)


def test_restore_shard_rewrites_only_the_audio(tmp_path):
    rows = []
    for i in range(3):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16_000 + 800 * i, dtype=np.float32), 16_000, format="WAV")
        rows.append({"audio": {"bytes": buf.getvalue(), "path": f"{i}.wav"}, "text": f"metin {i}", "speaker_id": "a"})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "in.parquet")
    n = restore_shard(str(tmp_path / "in.parquet"), tmp_path / "out" / "in.parquet", _sidon(), 24_000, limit=2)
    out = pq.read_table(tmp_path / "out" / "in.parquet").to_pylist()
    assert n == 2 and len(out) == 2
    assert [r["text"] for r in out] == ["metin 0", "metin 1"] and out[1]["speaker_id"] == "a"
    wav, sr = sf.read(io.BytesIO(out[1]["audio"]["bytes"]))
    assert sr == 24_000 and len(wav) == int(24_000 * 16_800 / 16_000) and out[1]["audio"]["path"] == "1.flac"
