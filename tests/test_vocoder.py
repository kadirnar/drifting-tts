import os

import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from drifting_tts.audio import HOP_LENGTH, BigVGANLogMel, LogMel
from drifting_tts.finetune_vocoder import random_segments, segment_samples

_SNAPSHOTS = os.path.join(HF_HUB_CACHE, "models--nvidia--bigvgan_v2_24khz_100band_256x", "snapshots")


def _cached(name: str) -> bool:
    return os.path.isdir(_SNAPSHOTS) and any(os.path.exists(os.path.join(_SNAPSHOTS, s, name))
                                             for s in os.listdir(_SNAPSHOTS))


# tiny BigVGAN-v2-shaped generator (256x upsampling) and discriminators: random init, no weights needed
TINY = {"upsample_rates": [8, 8, 4], "upsample_kernel_sizes": [16, 16, 8], "upsample_initial_channel": 16,
        "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3]], "mpd_reshapes": [2, 3],
        "discriminator_channel_mult": 0.125, "cqtd_filters": 8, "cqtd_max_filters": 16, "cqtd_hop_lengths": [256],
        "cqtd_n_octaves": [9], "cqtd_bins_per_octaves": [12]}


@pytest.mark.parametrize("backend", ["vocos", "bigvgan"])
def test_segment_alignment_is_exact(backend):
    """Recomputing the mel of the cut waveform reproduces the cut mel (frames whose window lies inside the segment)."""
    logmel = BigVGANLogMel() if backend == "bigvgan" else LogMel()
    g = torch.Generator().manual_seed(0)
    audio = 0.1 * torch.randn(3, 40 * HOP_LENGTH + 77, generator=g)  # length not a multiple of the hop
    mel = logmel(audio)
    mel_len = torch.full((3,), mel.shape[-1])
    frames = 24
    torch.manual_seed(1)
    m, a = random_segments(mel, audio, mel_len, frames, backend)
    assert m.shape == (3, 100, frames) and a.shape == (3, segment_samples(frames, backend))
    re = logmel(a)
    assert re.shape == m.shape  # F frames <-> F * hop (BigVGAN) or (F - 1) * hop (Vocos) samples
    inner = slice(3, frames - 3)  # edge frames see the segment's own padding instead of the neighbouring audio
    assert (re[..., inner] - m[..., inner]).abs().max() < 1e-4
    assert (re[..., 3: frames - 4] - m[..., 4: frames - 3]).abs().mean() > 0.1  # one frame off is clearly wrong


def test_bigvgan_discriminators_and_losses():
    pytest.importorskip("nnAudio")
    from drifting_tts.models import bigvgan_disc as bd

    h = {"sampling_rate": 24_000, **TINY}
    mpd, cqtd = bd.MultiPeriodDiscriminator(h), bd.MultiScaleSubbandCQTDiscriminator(h)
    y, y_hat = 0.1 * torch.randn(2, 1, 32 * HOP_LENGTH), (0.1 * torch.randn(2, 1, 32 * HOP_LENGTH)).requires_grad_()
    r_p, g_p, f_r_p, f_g_p = mpd(y, y_hat)
    r_c, g_c, f_r_c, f_g_c = cqtd(y, y_hat)
    assert len(r_p) == 2 and len(r_c) == 1 and len(f_r_p[0]) == 6
    assert any(k.endswith("weight_g") for k in mpd.state_dict())  # old-style weight norm, as the released ckpt
    l_d = bd.discriminator_loss(r_p, g_p) + bd.discriminator_loss(r_c, g_c)
    l_g = bd.generator_loss(g_p) + bd.generator_loss(g_c) + bd.feature_loss(f_r_p, f_g_p) + bd.feature_loss(f_r_c,
                                                                                                         f_g_c)
    (l_d + l_g).backward()
    assert torch.isfinite(l_d + l_g) and y_hat.grad.abs().sum() > 0

    mel_loss = bd.MultiScaleMelLoss()
    assert mel_loss(y, y).item() == 0 and mel_loss(y_hat, y).item() > 0
    librosa = pytest.importorskip("librosa")
    for w, m in zip(mel_loss.windows, (5, 10, 20, 40, 80, 160, 320)):  # Slaney filters == librosa (reference)
        ref = torch.from_numpy(librosa.filters.mel(sr=24_000, n_fft=w, n_mels=m)).T
        assert torch.allclose(getattr(mel_loss, f"fb{w}"), ref, atol=1e-5)


@pytest.mark.skipif(not _cached("bigvgan_discriminator_optimizer.pt"), reason="released discriminators not cached")
def test_released_bigvgan_discriminators_load():
    pytest.importorskip("nnAudio")
    import json

    from drifting_tts.models import bigvgan_disc as bd

    snap = next(os.path.join(_SNAPSHOTS, s) for s in os.listdir(_SNAPSHOTS)
                if os.path.exists(os.path.join(_SNAPSHOTS, s, "bigvgan_discriminator_optimizer.pt")))
    h = json.load(open(os.path.join(snap, "config.json")))
    ck = torch.load(os.path.join(snap, "bigvgan_discriminator_optimizer.pt"), map_location="cpu", weights_only=False)
    mpd, cqtd = bd.MultiPeriodDiscriminator(h), bd.MultiScaleSubbandCQTDiscriminator(h)
    mpd.load_state_dict(ck["mpd"])  # strict
    cqtd.load_state_dict(ck["mrd"])
    params = list(cqtd.parameters()) + list(mpd.parameters())  # reference optimizer order: chain(mrd, mpd)
    assert [ck["optim_d"]["state"][i]["exp_avg"].shape for i in range(len(params))] == [p.shape for p in params]


@pytest.mark.skipif(not _cached("bigvgan_generator.pt"), reason="BigVGAN-v2 weights not cached (large download)")
def test_bigvgan_vocoder_roundtrip_length():
    pytest.importorskip("librosa")
    from drifting_tts.vocoder import Vocoder

    voc = Vocoder("cpu", backend="bigvgan")
    wav = 0.1 * torch.sin(torch.arange(24_000) / 24_000 * 2 * torch.pi * 220)[None]
    out = voc(BigVGANLogMel()(wav))
    assert out.shape == (1, 24_000 // 256 * 256) and torch.isfinite(out).all()


@pytest.mark.skipif(not _cached("bigvgan.py"), reason="BigVGAN-v2 code (HF repo) not cached")
def test_finetune_bigvgan_cpu(tmp_path):
    """prepare --backend bigvgan -> tiny TTS -> BigVGAN GTA fine-tuning (tiny random G / D) -> resume -> Vocoder."""
    pytest.importorskip("nnAudio")
    pytest.importorskip("librosa")
    from drifting_tts.cli import main
    from drifting_tts.config import load_config
    from drifting_tts.data import MelDataset
    from drifting_tts.models.tts import DriftingTTS
    from drifting_tts.utils import save_checkpoint
    from drifting_tts.vocoder import Vocoder
    from tests.test_data import _fake_parquet

    _fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--save-audio", "--backend", "bigvgan"])
    ds = MelDataset(data, min_frames=1)
    cfg = load_config("configs/tts.yaml", [f"data.root={data}", "model.gen.hidden=32", "model.gen.depth=1",
                                           "model.gen.heads=2", "model.gen.noise_coords=2", "model.text.d=16",
                                           "model.text.layers=1", "model.text.ffn=32", "model.text.spk_dim=8"])
    tts = DriftingTTS(cfg.model, num_speakers=ds.num_speakers)  # untrained: GTA mels only need the right shapes
    save_checkpoint(tmp_path / "tts.pt", ema=tts.state_dict(), config=cfg.to_dict(), num_speakers=ds.num_speakers,
                    stats={"mean": ds.stats.mean, "std": ds.stats.std, "backend": ds.backend})
    work = tmp_path / "voc"
    args = ["finetune-vocoder", "--config", "configs/vocoder_bigvgan.yaml", "--workdir", str(work),
            f"data.root={data}", "data.min_quality=0", f"tts.path={tmp_path / 'tts.pt'}",
            "vocoder.pretrained=false", "vocoder.pretrained_discriminators=false", f"vocoder.hparams={TINY}",
            "train.cpu=true", "train.batch_size=2", "train.num_workers=0", "train.log_every=1", "train.save_every=2",
            "train.sample_every=2", "train.segment_frames=32", "train.gta_prob=1.0", "train.compile=false"]
    main(args + ["train.steps=2"])
    ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and "generator" in ck and "sched_g" in ck
    main(args + ["train.steps=3"])  # resumes from last.pt
    assert torch.load(work / "bigvgan_ft.pt", map_location="cpu", weights_only=False)["step"] == 3
    voc = Vocoder("cpu", finetuned=str(work / "bigvgan_ft.pt"), backend="bigvgan")
    assert not any(k.endswith("weight_v") for k in voc.model.state_dict())  # weight norm removed for inference
    out = voc(torch.randn(1, 100, 10) - 5)
    assert out.shape == (1, 10 * 256) and torch.isfinite(out).all()
