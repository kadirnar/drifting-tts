import json

import numpy as np
import pytest
import torch

from drifting_tts import latents
from drifting_tts.cli import main
from drifting_tts.config import Config
from drifting_tts.data import MelDataset, MelStats, collate
from drifting_tts.extract_latents import resample_f0
from drifting_tts.models.tts import DriftingTTS
from tests.test_data import _fake_parquet


class FakeVAE:
    """25 Hz, 6-d latents from 24 kHz audio: frame t is a fixed function of samples [960 t, 960 (t + 1))."""

    name, dim, frame_rate, input_rate, output_rate = "voxcpm2", 6, 25.0, 24000, 24000

    def num_frames(self, samples: int, sr: int) -> int:
        return -(-samples // 960)

    def encode(self, wav, sr):
        t = self.num_frames(wav.shape[-1], sr)
        x = torch.nn.functional.pad(wav.reshape(1, -1), (0, t * 960 - wav.shape[-1])).reshape(t, 960)
        feats = torch.stack([x.mean(1), x.std(1), x.abs().max(1).values, x[:, 0], x[:, -1], torch.arange(t) / t], 0)
        return feats[None] * 3 + 1


def test_per_channel_stats_round_trip_and_floor():
    st = MelStats([1.0, -2.0, 0.5], [2.0, 0.5, 0.0])
    assert st.per_channel and st.std.flatten().tolist() == pytest.approx([2.0, 0.5, 1e-3])  # collapsed: floored
    x = torch.randn(2, 3, 7)
    torch.testing.assert_close(st.denormalize(st.normalize(x)), x)
    assert st.to_dict()["mean"] == [1.0, -2.0, 0.5]
    scalar = MelStats(-5.0, 2.0)
    assert not scalar.per_channel and scalar.to_dict() == {"mean": -5.0, "std": 2.0}


def test_resample_f0_picks_the_frame_containing_each_centre():
    f0 = np.array([100.0, 0.0, 200.0, 300.0], np.float32)  # 4 frames at 2 Hz: [0, 0.5), [0.5, 1), ...
    out = resample_f0(f0, 2.0, 8, 4.0)  # centres 0.125, 0.375, 0.625, ... s
    assert out.tolist() == [100, 100, 0, 0, 200, 200, 300, 300]


@pytest.fixture(scope="module")
def latent_root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("latents")
    _fake_parquet(tmp / "x.parquet", n=8)
    main(["prepare", "--parquet-glob", str(tmp / "*.parquet"), "--out", str(tmp / "mel"), "--workers", "1",
          "--val-size", "0", "--no-trim", "--f0", "--save-audio"])
    original = latents.load_backend
    latents.load_backend = lambda name, device="cpu", **kw: FakeVAE()
    try:
        main(["extract-latents", "--data", str(tmp / "mel"), "--backend", "voxcpm2", "--out", str(tmp / "lat"),
              "--device", "cpu"])
    finally:
        latents.load_backend = original
    return tmp


def test_extract_latents_layout(latent_root):
    st = json.loads((latent_root / "lat" / "stats.json").read_text())
    assert st["dim"] == 6 and st["frame_rate"] == 100.0 and st["latent_repeat"] == 4 and len(st["mean"]) == 6
    src = [json.loads(line) for line in open(latent_root / "mel" / "index.jsonl")]
    dst = [json.loads(line) for line in open(latent_root / "lat" / "index.jsonl")]
    assert [e["norm_text"] for e in src] == [e["norm_text"] for e in dst]  # line order (scores.jsonl keys) kept
    for e in dst:
        assert e["frames"] == 4 * -(-e["audio_samples"] // 960)
    frames = sum(e["frames"] for e in dst)
    assert (latent_root / "lat" / "mels.bin").stat().st_size == frames * 6 * 2
    assert (latent_root / "lat" / "f0.bin").stat().st_size == frames * 2
    ds = MelDataset(latent_root / "lat", "train", min_frames=1, with_f0=True, with_audio=True)
    batch = collate([ds[i] for i in range(len(ds))])
    assert ds.dim == 6 and batch["mel"].shape[1] == 6 and batch["f0"].shape == batch["mel"][:, 0].shape
    item = ds[0]  # the 4 repeats of a latent frame are equal, and normalisation is per channel
    torch.testing.assert_close(item["mel"][:, 0], item["mel"][:, 3])


def test_patch4_model_on_latent_frames(latent_root):
    cfg = {"text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
           "gen": {"hidden": 32, "depth": 1, "heads": 2, "patch": 4, "mlp_ratio": 4.0, "n_registers": 2,
                   "noise_classes": 4, "noise_coords": 2, "residual_prior": True, "num_steps": 1},
           "pitch": {"enabled": True}}
    ds = MelDataset(latent_root / "lat", "train", min_frames=1)
    model = DriftingTTS(Config(cfg), num_speakers=ds.num_speakers, n_mels=ds.dim).eval()
    it = ds[0]
    with torch.no_grad():
        mel, y_len = model.synthesize(it["text"][None], torch.tensor([it["text"].numel()]), torch.tensor([0]))
    assert mel.shape[1] == 6 and mel.shape[2] == int(y_len)  # any frame count, not only multiples of the patch
