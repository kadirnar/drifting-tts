"""Kyutai's Mimi backend and extract-latents from parquet files (a fake backend; Mimi itself when available)."""

import json

import numpy as np
import pytest
import torch

from drifting_tts.latents.base import AudioBackend


class _Fake(AudioBackend):
    """A causal stand-in: frame t = (mean, std) of its 1920 input samples, x 2 channels."""

    name, input_rate, output_rate, frame_rate, dim, causal = "fake", 24000, 24000, 12.5, 4, True

    def encode(self, wav, sr):
        x = self._input(wav, sr)[:, 0]
        n = self.num_frames(x.shape[-1], self.input_rate) * self.hop_in
        x = torch.nn.functional.pad(x, (0, n - x.shape[-1])).reshape(x.shape[0], -1, self.hop_in)
        f = torch.stack([x.mean(-1), x.std(-1)], 1)
        return torch.cat([f, 2 * f], 1)

    def decode(self, latent):
        return latent[:, :1].repeat_interleave(self.hop_out, -1)


def test_extract_latents_from_parquet(tmp_path, monkeypatch):
    from drifting_tts import latents
    from drifting_tts.cli import main
    from drifting_tts.data import MelDataset
    from tests.test_text_en import _english_parquet

    monkeypatch.setattr(latents, "load_backend", lambda name, device="cpu", **kw: _Fake())
    _english_parquet(tmp_path / "train-0.parquet", 8)
    _english_parquet(tmp_path / "held-0.parquet", 6, start=100, seconds=2.5)
    out = tmp_path / "lat"
    main(["extract-latents", "--parquet-glob", str(tmp_path / "*.parquet"), "--val-glob", str(tmp_path / "held-*"),
          "--val-size", "4", "--backend", "fake", "--language", "en", "--speaker-column", "speaker_id",
          "--out", str(out), "--workers", "1", "--prefetch", "1", "--batch-seconds", "3", "--device", "cpu"])
    st = json.loads((out / "stats.json").read_text())
    assert st["language"] == "en" and st["dim"] == 4 and st["frame_rate"] == 50.0 and st["latent_repeat"] == 4
    index = [json.loads(line) for line in open(out / "index.jsonl")]
    assert len(index) == 14 and sorted(e["split"] for e in index).count("val") == 4
    assert sum(e["split"] == "dev" for e in index) == 2 and sum(e["split"] == "train" for e in index) == 8
    ds = MelDataset(out, "val", min_frames=1)
    assert ds.language == "en" and ds.dim == 4 and ds[0]["mel"].shape[0] == 4
    # each stored utterance equals its own encoding, alone (batching and padding change nothing), repeated 4 times
    import pyarrow.parquet as pq

    from drifting_tts.audio import prepare_waveform

    rows = {r["text"]: r["audio"]["bytes"] for r in pq.read_table(tmp_path / "held-0.parquet").to_pylist()}
    mels = np.memmap(out / "mels.bin", dtype=np.float16, mode="r").reshape(-1, 4)
    e = next(e for e in index if e["split"] == "val")
    alone = _Fake().encode(prepare_waveform(rows[e["text"]]), 24000)[0].repeat_interleave(4, 1).T.numpy()
    np.testing.assert_allclose(np.array(mels[e["offset"]: e["offset"] + e["frames"]], dtype=np.float32), alone,
                               atol=2e-3)


def test_extract_latents_from_parquet_with_a_posterior(tmp_path, monkeypatch):
    """A codec with a posterior, straight from parquet shards: the frames stored once (``--native``), the
    posterior's standard deviations beside them, and the channels the codec does not use named in the statistics."""
    from drifting_tts import latents
    from drifting_tts.cli import main
    from drifting_tts.data import MelDataset
    from tests.test_text_en import _english_parquet

    class WithPosterior(_Fake):
        def posterior(self, wav, sr):
            z = self.encode(wav, sr)
            std = torch.full_like(z, 1e-7)  # far below what the data does in the first three channels
            std[:, 3] = 50.0  # the last channel: a posterior wider than anything the data does there
            return z, std

    monkeypatch.setattr(latents, "load_backend", lambda name, device="cpu", **kw: WithPosterior())
    _english_parquet(tmp_path / "train-0.parquet", 8)
    _english_parquet(tmp_path / "held-0.parquet", 6, start=100, seconds=2.5)
    out = tmp_path / "lat"
    main(["extract-latents", "--parquet-glob", str(tmp_path / "*.parquet"), "--val-glob", str(tmp_path / "held-*"),
          "--val-size", "4", "--backend", "fake", "--language", "en", "--speaker-column", "speaker_id",
          "--out", str(out), "--workers", "1", "--prefetch", "1", "--batch-seconds", "3", "--device", "cpu",
          "--native", "--save-std"])
    st = json.loads((out / "stats.json").read_text())
    assert st["stored_repeat"] == 1 and st["latent_repeat"] == 4 and st["posterior_std"]
    assert st["active"] == [True, True, True, False] and 0 < st["posterior_rel"] < 1
    index = [json.loads(line) for line in open(out / "index.jsonl")]
    frames = sum(e["frames"] for e in index)
    assert (out / "mels.bin").stat().st_size == frames // 4 * 4 * 2 == (out / "stds.bin").stat().st_size
    ds = MelDataset(out, "train", min_frames=1, with_std=True)
    it = ds[0]
    assert it["mel"].shape == it["mel_std"].shape and it["mel"].shape[1] == ds.frames(0)
    assert not it["mel"][3].any() and not it["mel_std"][3].any()  # the unused channel is left out
    assert torch.equal(it["mel"][:, 0], it["mel"][:, 3]) and it["mel_std"][0].min() > 0  # copies made on reading


def test_mimi_backend_frames():
    pytest.importorskip("pocket_tts")
    from drifting_tts.latents import load_backend

    try:
        be = load_backend("mimi", "cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:  # gated weights not accepted / offline
        pytest.skip(f"Mimi weights unavailable: {e}")
    for n in (24000, 1920 * 7, 30001):
        z = be.encode(torch.randn(n) * 0.05, 24000)
        assert z.shape == (1, 32, be.num_frames(n, 24000))
        assert be.decode(z).shape == (1, z.shape[-1] * 1920)


def test_mimi_decoder_training_path():
    """decode_train (finetune-vocoder) is decode with gradients; a fine-tuned decoder state loads back."""
    pytest.importorskip("pocket_tts")
    from drifting_tts.latents import load_backend
    from drifting_tts.latents.mimi import MimiBackend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        be = load_backend("mimi", device)
    except Exception as e:  # gated weights not accepted / offline
        pytest.skip(f"Mimi weights unavailable: {e}")
    torch.manual_seed(0)
    z = be.encode(torch.randn(24000 * 2) * 0.05, 24000)
    with torch.no_grad():
        torch.testing.assert_close(be.decode_train(z).clamp(-1, 1), be.decode(z))
    z.requires_grad_(True)
    be.decode_train(z).pow(2).mean().backward()
    assert z.grad.abs().sum() > 0
    dec = be.trainable_decoder()
    assert set(dict(dec.named_children())) == {"quantizer", "upsample", "decoder_transformer", "decoder"}
    state = {k: v + 0.01 * torch.randn_like(v) for k, v in dec.state_dict().items()}
    tuned = MimiBackend(device, decoder={"decoder": state, "backend": "mimi"})
    for k, v in tuned.trainable_decoder().state_dict().items():
        torch.testing.assert_close(v, state[k])
    assert not torch.allclose(tuned.decode(z.detach()), be.decode(z.detach()), atol=1e-3)


def test_mimi_local_weights(tmp_path, monkeypatch):
    """$DRIFTING_TTS_MIMI_WEIGHTS (a Space without a Hub token): save_weights' file loads the same Mimi, no download."""
    pytest.importorskip("pocket_tts")
    import huggingface_hub

    from drifting_tts.latents import mimi

    if not isinstance(huggingface_hub.try_to_load_from_cache(mimi.REPO, mimi.FILE, revision=mimi.REVISION), str):
        pytest.skip("Mimi weights not cached")
    path = tmp_path / "mimi.safetensors"
    mimi.save_weights(path)
    released = mimi.load_mimi("cpu").state_dict()

    def offline(*a, **k):
        raise RuntimeError("no download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    monkeypatch.setenv(mimi.WEIGHTS_ENV, str(path))
    assert path.stat().st_size < 50e6
    for k, v in mimi.load_mimi("cpu").state_dict().items():
        torch.testing.assert_close(v, released[k])
