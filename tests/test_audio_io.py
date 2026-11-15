"""The audio path before any model (#94): the resampler, the rate the codecs are fed, latent crops and tails."""

import json
import math
import shutil

import numpy as np
import pytest
import torch
import torchaudio

from drifting_tts import latents
from drifting_tts import train as train_mod
from drifting_tts.audio import SAMPLE_RATE, prepare_waveform, resample
from drifting_tts.cli import main
from drifting_tts.config import load_config
from drifting_tts.extract_latents import encode_batch, source_rate
from drifting_tts.latents.base import AudioBackend
from drifting_tts.latents.vocoder import LatentVocoder, native_frames
from drifting_tts.models.mae import MelMAE
from drifting_tts.models.tts import DriftingTTS
from drifting_tts.train import CropBank, crop_grid, training_step
from drifting_tts.utils import random_crop
from tests.test_data import _fake_parquet, _wav_bytes
from tests.test_latents import tiny_dacvae
from tests.test_train import TINY


def tone(freq: float, sr: int, seconds: float = 1.0) -> torch.Tensor:
    return torch.sin(2 * math.pi * freq * torch.arange(int(seconds * sr)) / sr)


def rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def test_resampler_keeps_the_top_octave():
    """A 10.5 kHz tone through 48 -> 24 kHz and 24 -> 48 kHz: full level and no image, where torchaudio's default
    filter loses 1.8 dB and leaves the image 15 dB below the tone."""
    x = tone(10_500, 48_000)
    sharp, soft = resample(x, 48_000, 24_000), torchaudio.functional.resample(x, 48_000, 24_000)
    assert sharp.shape == soft.shape == (24_000,)
    assert abs(rms(sharp[2000:-2000]) / rms(x) - 1) < 0.005 and rms(soft[2000:-2000]) / rms(x) < 0.85

    x = tone(10_500, 24_000)
    for up, limit in ((resample(x, 24_000, 48_000), -80), (torchaudio.functional.resample(x, 24_000, 48_000), None)):
        power = torch.fft.rfft(up[4000:-4000] * torch.hann_window(40_000)).abs() ** 2
        k = 12_000 * 40_000 // 48_000  # above the source's Nyquist: the image at 13.5 kHz
        image_db = 10 * math.log10(float(power[k:].sum() / power[:k].sum()))
        assert image_db < limit if limit else image_db > -20
    assert resample(x, 24_000, 24_000) is x


def test_prepare_waveform_at_another_rate():
    data = _wav_bytes(1.0, sr=44_100, freq=440.0)
    w24 = prepare_waveform(data, trim=False)
    w48 = prepare_waveform(data, trim=False, sample_rate=48_000)
    both = prepare_waveform(data, trim=False, also=48_000)
    assert w24.shape == (24_000,) and w48.shape == (48_000,)
    assert torch.equal(both[0], w24) and torch.equal(both[1], w48)
    torch.testing.assert_close(resample(w48, 48_000, 24_000)[500:-500], w24[500:-500], atol=2e-3, rtol=0)


class Fake48(AudioBackend):
    """A 25 Hz "codec" that takes 48 kHz audio: frame t holds the mean and the level of its 1920 samples."""

    name, dim, frame_rate, input_rate, output_rate, causal = "fake48", 2, 25.0, 48_000, 48_000, True

    def encode(self, wav, sr):
        x = self._input(wav, sr)[:, 0]
        x = torch.nn.functional.pad(x, (0, -x.shape[-1] % 1920)).unflatten(-1, (-1, 1920))
        return torch.stack([x.mean(-1), x.pow(2).mean(-1).sqrt()], 1)

    def decode(self, latent):
        return latent[:, 1].repeat_interleave(1920, -1)


@pytest.fixture(scope="module")
def hires_root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hires")
    _fake_parquet(tmp / "x.parquet", n=6)  # 44.1 kHz sources
    main(["prepare", "--parquet-glob", str(tmp / "*.parquet"), "--out", str(tmp / "mel"), "--workers", "1",
          "--val-size", "0", "--no-trim", "--f0", "--save-audio", "--hires-rate", "48000"])
    return tmp


def test_prepare_stores_the_audio_at_a_second_rate(hires_root, tmp_path):
    root = hires_root / "mel"
    assert json.loads((root / "stats.json").read_text())["hr_rate"] == 48_000
    index = [json.loads(line) for line in open(root / "index.jsonl")]
    hr = np.memmap(root / "audio_hr.bin", dtype=np.float16, mode="r")
    lo = np.memmap(root / "audio.bin", dtype=np.float16, mode="r")
    assert sum(e["hr_samples"] for e in index) == hr.size and index[1]["hr_offset"] == index[0]["hr_samples"]
    for e in index:  # the same utterance at twice the rate
        assert abs(e["hr_samples"] - 2 * e["audio_samples"]) <= 1
        a = torch.from_numpy(np.array(hr[e["hr_offset"]: e["hr_offset"] + e["hr_samples"]], dtype=np.float32))
        b = torch.from_numpy(np.array(lo[e["audio_offset"]: e["audio_offset"] + e["audio_samples"]], dtype=np.float32))
        torch.testing.assert_close(resample(a, 48_000, 24_000)[500:-500], b[500:-500], atol=3e-3, rtol=0)
    with pytest.raises(SystemExit, match="--save-audio"):
        main(["prepare", "--parquet-glob", str(hires_root / "*.parquet"), "--out", str(tmp_path / "x"),
              "--hires-rate", "48000"])


def test_a_48_khz_codec_is_fed_48_khz_audio(hires_root, monkeypatch, capsys):
    monkeypatch.setattr(latents, "load_backend", lambda name, device="cpu", **kw: Fake48())
    src = hires_root / "mel"
    index = [json.loads(line) for line in open(src / "index.jsonl")]
    runs = {}
    for tag, extra in (("native", []), ("band", ["--audio-rate", "24000"])):
        out = hires_root / f"lat_{tag}"
        main(["extract-latents", "--data", str(src), "--backend", "fake48", "--out", str(out), "--device", "cpu",
              *extra])
        runs[tag] = (json.loads((out / "stats.json").read_text()), [json.loads(x) for x in open(out / "index.jsonl")],
                     np.memmap(out / "mels.bin", dtype=np.float16, mode="r").reshape(-1, 2), capsys.readouterr().out)
    st, idx, mels, note = runs["native"]
    assert st["audio_rate"] == 48_000 and "gets" not in note
    hr = np.memmap(src / "audio_hr.bin", dtype=np.float16, mode="r")
    for e, d in zip(index, idx):  # the stored latents are the encoding of the 48 kHz waveform itself
        wav = torch.from_numpy(np.array(hr[e["hr_offset"]: e["hr_offset"] + e["hr_samples"]], dtype=np.float32))
        z = Fake48().encode(wav, 48_000)[0].repeat_interleave(4, 1).T
        assert d["frames"] == 4 * math.ceil(e["hr_samples"] / 1920)
        np.testing.assert_allclose(np.array(mels[d["offset"]: d["offset"] + d["frames"]], dtype=np.float32), z,
                                   atol=1e-3)
    st, idx, _, note = runs["band"]  # as before #94: the 24 kHz waveforms, and a note that says so
    assert st["audio_rate"] == 24_000 and "48000 Hz audio and gets 24000 Hz" in note
    with pytest.raises(SystemExit, match="holds audio at 24000 Hz and 48000 Hz"):
        main(["extract-latents", "--data", str(src), "--backend", "fake48", "--out", str(hires_root / "x"),
              "--device", "cpu", "--audio-rate", "32000"])


def test_source_rate():
    be48, be16 = Fake48(), Fake48()
    be16.input_rate = 16_000
    assert source_rate(be48, None) == 48_000 and source_rate(be16, None) == SAMPLE_RATE  # parquet: any rate
    assert source_rate(be48, None, hires=SAMPLE_RATE) == SAMPLE_RATE  # a root without audio_hr.bin
    assert source_rate(be48, None, hires=44_100) == 44_100 and source_rate(be16, None, hires=48_000) == SAMPLE_RATE
    assert source_rate(be48, 24_000, hires=48_000) == 24_000


def test_merge_data_keeps_hires_audio_only_when_every_input_has_it(hires_root, tmp_path):
    main(["prepare", "--parquet-glob", str(hires_root / "*.parquet"), "--out", str(tmp_path / "plain"),
          "--workers", "1", "--val-size", "0", "--no-trim", "--f0", "--save-audio"])
    src = hires_root / "mel"
    shutil.copytree(src, tmp_path / "copy")
    main(["merge-data", str(src), str(tmp_path / "copy"), "--out", str(tmp_path / "both")])
    index = [json.loads(line) for line in open(tmp_path / "both" / "index.jsonl")]
    size = (src / "audio_hr.bin").stat().st_size
    assert (tmp_path / "both" / "audio_hr.bin").stat().st_size == 2 * size
    assert index[6]["hr_offset"] == size // 2 and index[6]["hr_samples"] == index[0]["hr_samples"]
    assert json.loads((tmp_path / "both" / "stats.json").read_text())["hr_rate"] == 48_000

    main(["merge-data", str(src), str(tmp_path / "plain"), "--out", str(tmp_path / "mixed")])
    index = [json.loads(line) for line in open(tmp_path / "mixed" / "index.jsonl")]
    assert not (tmp_path / "mixed" / "audio_hr.bin").exists() and (tmp_path / "mixed" / "audio.bin").exists()
    assert all("hr_offset" not in e and "audio_offset" in e for e in index)
    assert "hr_rate" not in json.loads((tmp_path / "mixed" / "stats.json").read_text())


def test_batched_encoding_pads_each_row_as_the_encoder_does():
    """DAC-VAE reflect-pads an utterance to whole frames: rows of one frame count are encoded in a batch exactly as
    alone, whatever their lengths; with zeros in place of the reflection they are not."""
    be = tiny_dacvae()  # 800 Hz, 8 samples per frame
    assert be.frame_pad == "reflect" and be.hop_in == 8
    torch.manual_seed(0)
    wavs = [0.3 * torch.randn(n).numpy() for n in (157, 153, 160)]  # 20 frames each
    batch = encode_batch(be, wavs, 800)
    for w, z in zip(wavs, batch):
        assert z.shape == (be.dim, 20)
        torch.testing.assert_close(z, be.encode(torch.from_numpy(w), 800)[0], atol=1e-6, rtol=0)
    x = torch.zeros(3, 160)
    for k, w in enumerate(wavs):
        x[k, : len(w)] = torch.from_numpy(w)
    zero_padded = be.encode(x, 800)
    assert not torch.allclose(zero_padded[1], batch[1], atol=1e-4)

    # rows of different frame counts: each has its own frames, and at another input rate the row is resampled first
    wavs = [0.3 * torch.randn(n).numpy() for n in (310, 123)]
    assert [z.shape[-1] for z in encode_batch(be, wavs, 1600)] == [be.num_frames(310, 1600), be.num_frames(123, 1600)]
    alone = be.encode(torch.from_numpy(wavs[0]), 1600)[0]  # the longest row has no padding after its own
    torch.testing.assert_close(encode_batch(be, wavs, 1600)[0], alone, atol=1e-6, rtol=0)


def test_native_frames_keep_a_partial_last_group():
    x = torch.arange(20.0).reshape(1, 2, 10)
    out = native_frames(x, 4)
    assert out.shape == (1, 2, 3)
    torch.testing.assert_close(out[0, 0], torch.tensor([1.5, 5.5, 8.5]))  # (8 + 9) / 2: not dropped
    assert torch.equal(native_frames(x[..., :8], 4), x[..., :8].unflatten(-1, (2, 4)).mean(-1))
    assert native_frames(x, 1) is x


def test_latent_vocoder_decodes_every_frame_at_either_rate(monkeypatch):
    monkeypatch.setattr("drifting_tts.latents.vocoder.load_backend", lambda name, device="cpu", **kw: Fake48())
    z = torch.rand(1, 2, 10) * 0.5  # 2.5 VAE frames: the last two frames were dropped before #94
    native = LatentVocoder("fake48", "cpu", repeat=4, sample_rate=None)
    assert native.sample_rate == 48_000 and native(z).shape == (1, 10 * 480)
    torch.testing.assert_close(native(z)[0, -1], z[0, 1, 8:].mean())  # the partial group, averaged
    assert native(z[..., :8]).shape == (1, 8 * 480)
    down = LatentVocoder("fake48", "cpu", repeat=4)
    assert down.sample_rate == SAMPLE_RATE and down(z).shape == (1, 10 * 240)

    # 48 -> 24 kHz with the filter the decoder was fine-tuned through: torchaudio's default before #94
    z = torch.rand(1, 2, 40) * 0.5
    y48 = native(z)
    old = LatentVocoder("fake48", "cpu", repeat=4, decoder={"decoder": {}, "backend": "fake48"})
    new = LatentVocoder("fake48", "cpu", repeat=4, decoder={"decoder": {}, "backend": "fake48", "resample": "sharp"})
    torch.testing.assert_close(old(z), torchaudio.functional.resample(y48, 48_000, 24_000).clamp(-1, 1))
    torch.testing.assert_close(new(z), down(z))
    torch.testing.assert_close(down(z), resample(y48, 48_000, 24_000).clamp(-1, 1))
    assert not torch.allclose(old(z), new(z), atol=1e-3)


def test_crops_start_on_whole_vae_frames(monkeypatch):
    cfg = load_config("configs/tts.yaml", TINY)
    assert crop_grid(cfg, 1) == 1 and crop_grid(cfg, 4) == 4  # default: the data's latent_repeat
    cfg.drift.crop_grid = 1
    assert crop_grid(cfg, 4) == 1  # the earlier behaviour, for an A/B
    cfg.drift.crop_grid = 5
    with pytest.raises(ValueError, match="multiple of drift.crop_grid"):
        crop_grid(cfg, 4)

    seen = []
    crop = train_mod.crop_frames

    def spy(x, starts, n):
        seen.append(starts.clone())
        return crop(x, starts, n)

    monkeypatch.setattr(train_mod, "crop_frames", spy)
    torch.manual_seed(0)
    model = DriftingTTS(cfg.model, num_speakers=2)
    mae = MelMAE(n_mels=100, base_channels=8).eval().requires_grad_(False)
    batch = {"text": torch.randint(2, 39, (3, 20)), "text_len": torch.tensor([20, 18, 15]),
             "mel": torch.randn(3, 100, 120), "mel_len": torch.tensor([120, 111, 97]), "spk": torch.tensor([0, 1, 0])}
    for grid in (4, 1):
        cfg.drift.crop_grid = grid
        seen.clear()
        for _ in range(4):
            training_step(model, mae, batch, CropBank(16, 100, 64, "cpu"), cfg, "cpu")
        starts = torch.cat(seen)
        assert (starts + 64 <= batch["mel_len"].repeat(len(starts) // 3)).all()
        assert bool((starts % 4 == 0).all()) == (grid == 4)

    torch.manual_seed(0)
    _, starts = random_crop(torch.randn(64, 2, 90), torch.full((64,), 90), 32, grid=4)
    assert (starts % 4 == 0).all() and starts.max() <= 56 and len(starts.unique()) > 4
