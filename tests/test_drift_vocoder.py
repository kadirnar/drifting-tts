"""GAN-free (drifting) vocoder: tiny generators and tiny feature extractors on CPU."""

import os

import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from drifting_tts.config import Config
from drifting_tts.drift_vocoder import (
    DiscriminatorFeatures,
    FeatureSpace,
    NoisyVocoder,
    SSLFeatures,
    STFTFeatures,
    shifted_views,
    split_samples,
    summarize,
    vocoder_drift_loss,
)
from drifting_tts.finetune_vocoder import DriftVocoder, VocosGAN, build_trainer
from drifting_tts.vocoder import Vocoder, extend_input_conv, fold_noise_channels
from tests.test_train import TINY_VOCOS
from tests.test_vocoder import TINY

TINY_MRD = {"mpd_reshapes": [2, 3], "discriminator_channel_mult": 0.125, "discriminator": "mrd",
            "resolutions": [[512, 50, 240], [256, 32, 128]], "mrd_channel_mult": 0.25}
TINY_CQT = {"mpd_reshapes": [2], "discriminator_channel_mult": 0.125, "use_cqtd_instead_of_mrd": True,
            "cqtd_filters": 8, "cqtd_max_filters": 16, "cqtd_hop_lengths": [256], "cqtd_n_octaves": [9],
            "cqtd_bins_per_octaves": [12]}
_SNAPSHOTS = os.path.join(HF_HUB_CACHE, "models--nvidia--bigvgan_v2_24khz_100band_256x", "snapshots")
_BIGVGAN_CODE = os.path.isdir(_SNAPSHOTS) and any(os.path.exists(os.path.join(_SNAPSHOTS, s, "bigvgan.py"))
                                                  for s in os.listdir(_SNAPSHOTS))


def _tiny_space(**kw) -> FeatureSpace:
    return FeatureSpace({"disc": DiscriminatorFeatures(None, hparams=TINY_MRD, pretrained=False),
                         "stft": STFTFeatures((256, 512), (8, 2))}, **kw)


def _cfg(tmp_path, arch: str = "vocos", **drift) -> Config:
    (tmp_path / "tiny_vocos.yaml").write_text(TINY_VOCOS)
    vocoder = {"objective": "drift", "arch": arch, "init": str(tmp_path / "tiny_vocos.yaml"), "noise_channels": 3}
    if arch == "bigvgan":
        vocoder.update(repo="nvidia/bigvgan_v2_24khz_100band_256x", pretrained=False, hparams=TINY)
    return Config({
        "vocoder": vocoder,
        "features": {"discriminators": {"repo": None, "hparams": TINY_MRD, "pretrained": False},
                     "stft": {"n_ffts": [256, 512], "patch": [8, 2]}},
        "drift": {"samples": 3, "pairing": {"conditional": 1.0, "pooled": 0.5}, "max_locations": 64,
                  "pooled_locations": 32, **drift},
        "train": {"lr": 1e-3, "grad_clip": 10.0, "mel_loss_coeff": 15.0, "drift_coeff": 1.0},
    })


@pytest.mark.parametrize("weight_norm", [False, True])
def test_extend_and_fold_input_conv(weight_norm):
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(5, 4, 3, padding=1)
    if weight_norm:
        conv = torch.nn.utils.weight_norm(conv)
    x, z = torch.randn(2, 5, 7), torch.randn(2, 2, 7)
    ref = conv(x).detach()
    extend_input_conv(conv, 2)
    assert conv.in_channels == 7 and torch.allclose(conv(torch.cat([x, z], 1)), ref, atol=1e-6)  # zero-initialised
    name = "weight_v" if weight_norm else "weight"
    with torch.no_grad():
        getattr(conv, name)[:, 5:] += 0.5  # the noise weights learned something
    noisy, at_zero = conv(torch.cat([x, z], 1)), conv(torch.cat([x, torch.zeros_like(z)], 1)).detach()
    assert not torch.allclose(noisy, at_zero)
    if weight_norm:
        with pytest.raises(ValueError):
            fold_noise_channels(conv, 2)
        torch.nn.utils.remove_weight_norm(conv)
    fold_noise_channels(conv, 2)
    assert conv.in_channels == 5 and torch.allclose(conv(x), at_zero, atol=1e-6)  # exact at z = 0


def test_noisy_vocos_same_padding_and_initial_identity(tmp_path):
    from drifting_tts.vocoder import build_vocos

    (tmp_path / "v.yaml").write_text(TINY_VOCOS)
    torch.manual_seed(0)
    net = build_vocos(str(tmp_path / "v.yaml"))
    net.head.istft.padding = "same"
    mel = torch.randn(2, 100, 20) - 5
    ref = net.head(net.backbone(mel)).detach()
    gen = NoisyVocoder(net, noise_channels=4)
    out = gen(mel, torch.randn(2, 4, 20))
    assert out.shape == (2, 20 * 256) and torch.allclose(out, ref, atol=1e-5)  # F frames -> F * hop samples
    assert torch.allclose(gen(mel), ref, atol=1e-5)  # z = None means z = 0


def test_feature_space_shapes():
    space = _tiny_space(global_stats=True)
    feats = space(0.1 * torch.randn(3, 4096))
    assert {"mpd2_0", "mpd3_5", "mrd0_0", "mrd1_5", "stft256", "stft512", "stft256_mean", "mpd2_0_std"} <= set(feats)
    assert all(v.dim() == 3 and v.shape[0] == 3 and v.dtype == torch.float32 for v in feats.values())
    assert feats["mpd2_0"].shape[-1] == 4 and feats["mpd2_5"].shape[-1] == 1  # 32 * 0.125 channels; the output map
    assert feats["stft256_mean"].shape[1] == 1 and feats["stft512"].shape[-1] == 16  # 8 bins x 2 frames
    assert not any(p.requires_grad for p in space.parameters())
    assert len(space.parts()) == 2 + 2 + 1  # MPD periods, MRD resolutions, the STFT set


def test_cqt_discriminator_features():
    pytest.importorskip("nnAudio")
    d = DiscriminatorFeatures(None, hparams=TINY_CQT, pretrained=False)
    feats = d(0.1 * torch.randn(2, 8192))
    assert [k for k in feats if k.startswith("cqtd")] == [f"cqtd0_{i}" for i in range(6)]  # 5 convs + output map
    assert feats["cqtd0_5"].shape[-1] == 1 and feats["cqtd0_0"].shape[-1] == 8


def test_ssl_features_tiny():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.WavLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                                   conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
                                   num_conv_pos_embedding_groups=2, num_buckets=8, max_bucket_distance=16)
    ssl = SSLFeatures(layers=(1, 2), normalize=True, model=transformers.WavLMModel(cfg))
    wave = (0.1 * torch.randn(2, 4800)).requires_grad_()
    feats = FeatureSpace({"ssl": ssl})(wave)
    assert set(feats) == {"ssl1", "ssl2"} and feats["ssl2"].shape[0] == 2 and feats["ssl2"].shape[-1] == 16
    feats["ssl2"].sum().backward()
    assert wave.grad.abs().sum() > 0


@pytest.mark.parametrize("pairing", [{"conditional": 1.0}, {"pooled": 1.0}, {"conditional": 1.0, "pooled": 0.5}])
@pytest.mark.parametrize("kyutai", [False, True])
def test_drift_loss_is_finite_and_differentiable(pairing, kyutai):
    torch.manual_seed(0)
    space, B, S = _tiny_space(), 2, 3
    wave = (0.1 * torch.randn(B * S, 4096)).requires_grad_()
    taus = {k: torch.tensor(1.0, requires_grad=True) for k in pairing} if kyutai else None
    gen = split_samples(space(wave), B)
    with torch.no_grad():
        pos = split_samples(space(shifted_views(0.1 * torch.randn(B, 4096), (-2, 3)).flatten(0, 1)), B)
    assert gen["mpd2_0"].shape[:2] == (B, S) and pos["mpd2_0"].shape[:2] == (B, 3)
    loss, info = vocoder_drift_loss(gen, pos, pairing, max_locations=64, pooled_locations=32, taus=taus)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(wave.grad).all() and wave.grad.abs().sum() > 0
    if kyutai:
        assert all(t.grad is not None and torch.isfinite(t.grad) for t in taus.values())
    stats = summarize(info)
    assert stats and all(map(lambda v: v == v, stats.values()))


def test_field_vanishes_when_samples_equal_real():
    """Conditional: S copies of the real segment -> the raw (pre-normalisation) field is 0. Pooled: generated and real
    patches from one distribution -> a much smaller raw field than from a shifted one."""
    torch.manual_seed(0)
    space, B, S = _tiny_space(), 2, 3
    real = 0.1 * torch.randn(B, 4096)
    pos = split_samples(space(real), B)
    same = split_samples(space(real.repeat_interleave(S, 0)), B)
    other = split_samples(space(real.repeat_interleave(S, 0) + 0.05 * torch.randn(B * S, 4096)), B)

    def force(gen, pairing, **kw):
        _, info = vocoder_drift_loss(gen, pos, pairing, max_locations=None, **kw)
        return max(v for k, v in summarize(info).items() if "force" in k)

    assert force(same, {"conditional": 1.0}) < 1e-10 < 1e-3 < force(other, {"conditional": 1.0})

    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(1, 2048, 1, 6, generator=g), torch.randn(1, 2048, 1, 6, generator=g)
    _, same_info = vocoder_drift_loss({"m": x}, {"m": y}, {"pooled": 1.0}, pooled_locations=None)
    _, shift_info = vocoder_drift_loss({"m": x + 1.0}, {"m": y}, {"pooled": 1.0}, pooled_locations=None)
    f_same, f_shift = summarize(same_info), summarize(shift_info)
    assert all(f_same[k] < 0.1 * f_shift[k] for k in f_same if "force" in k)


def test_partwise_gradient_matches_full_graph(tmp_path):
    """DriftVocoder.drift_grad (one feature part at a time) == autograd through the whole feature space."""
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, max_locations=None, pooled_locations=None)
    cfg.features.discriminators.spectral = False  # MPD + STFT: few enough locations to drift all of them
    tr = DriftVocoder(cfg, "cpu", mel="bigvgan", arch="vocos")
    B, wave, audio = 2, 0.1 * torch.randn(6, 1024), 0.1 * torch.randn(2, 1024)  # short: no location subsampling
    grad, loss, _ = tr.drift_grad(wave, audio, B)
    x = wave.clone().requires_grad_()
    ref, _ = vocoder_drift_loss(split_samples(tr.features(x), B), split_samples(tr.features(audio), B),
                                tr.pairing, max_locations=None, pooled_locations=None)
    ref.backward()
    assert torch.allclose(loss, ref.detach(), rtol=1e-5)
    assert (grad - x.grad).norm() < 1e-4 * x.grad.norm()


@pytest.mark.parametrize("mode", ["official", "kyutai"])
def test_drift_vocoder_vocos_step_export_and_load(tmp_path, mode):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, mode=mode, pos_shifts=[-1, 2])
    tr = build_trainer(cfg, "cpu", "vocos", "bigvgan")
    assert isinstance(tr, DriftVocoder) and tr.gen.net.head.istft.padding == "same"
    mel, audio = torch.randn(2, 100, 16) - 5, 0.1 * torch.randn(2, 16 * 256)
    before = [p.detach().clone() for p in tr.gen.parameters()]
    for step in range(3):
        metrics = tr.step(mel, audio, step)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert metrics["spread"] > 0  # the noise weights left zero: samples differ
    assert any(not torch.equal(a, b) for a, b in zip(before, tr.gen.parameters()))
    if mode == "kyutai":
        assert "cond_tau" in metrics and "pool_tau" in metrics
    tr.export(tmp_path / "vocos_ft.pt", 3)
    ck = torch.load(tmp_path / "vocos_ft.pt", map_location="cpu", weights_only=False)
    assert ck["noise_channels"] == 3 and ck["head_padding"] == "same" and ck["mel"] == "bigvgan"
    voc = Vocoder("cpu", finetuned=str(tmp_path / "vocos_ft.pt"), backend="bigvgan")  # z = 0, noise folded away
    assert voc.model.backbone.embed.in_channels == 100
    with torch.no_grad():
        ref = tr.generate(mel).clamp(-1, 1)
    assert torch.allclose(voc(mel), ref, atol=1e-5)
    seeded = Vocoder("cpu", finetuned=str(tmp_path / "vocos_ft.pt"), backend="bigvgan", noise_seed=0)
    out = seeded(mel)
    assert out.shape == ref.shape and torch.equal(out, seeded(mel)) and not torch.allclose(out, ref, atol=1e-6)
    tr2 = DriftVocoder(cfg, "cpu", mel="bigvgan", arch="vocos")  # resume
    tr2.load_state_dict(tr.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(tr.gen.parameters(), tr2.gen.parameters()))


@pytest.mark.skipif(not _BIGVGAN_CODE, reason="BigVGAN code (HF repo) not cached")
def test_drift_vocoder_bigvgan_step_export_and_load(tmp_path):
    torch.manual_seed(0)
    tr = build_trainer(_cfg(tmp_path, arch="bigvgan"), "cpu", "bigvgan", "bigvgan")
    assert hasattr(tr.gen.net.conv_pre, "weight_v")  # trained with weight norm, as released
    mel, audio = torch.randn(2, 100, 16) - 5, 0.1 * torch.randn(2, 16 * 256)
    for step in range(2):
        metrics = tr.step(mel, audio, step)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    tr.export(tmp_path / "bigvgan_ft.pt", 2)
    voc = Vocoder("cpu", finetuned=str(tmp_path / "bigvgan_ft.pt"), backend="bigvgan")
    assert voc.model.conv_pre.in_channels == 100  # a standard BigVGAN after folding
    with torch.no_grad():
        ref = tr.generate(mel).clamp(-1, 1)
    assert voc(mel).shape == (2, 16 * 256) and torch.allclose(voc(mel), ref, atol=1e-5)


def test_objective_selection(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.vocoder.pop("objective")
    assert isinstance(build_trainer(cfg, "cpu", "vocos", "vocos"), VocosGAN)  # default: the GAN recipe
    cfg.vocoder.objective = "gan"
    assert isinstance(build_trainer(cfg, "cpu", "vocos", "vocos"), VocosGAN)
    cfg.vocoder.objective = "flow"
    with pytest.raises(ValueError):
        build_trainer(cfg, "cpu", "vocos", "vocos")


def test_finetune_drift_vocoder_cli(tmp_path):
    """prepare --backend bigvgan -> tiny TTS -> drift fine-tuning of a tiny Vocos (CLI) -> resume -> Vocoder."""
    pytest.importorskip("librosa")
    from drifting_tts.cli import main
    from drifting_tts.config import load_config
    from drifting_tts.data import MelDataset
    from drifting_tts.models.tts import DriftingTTS
    from drifting_tts.utils import save_checkpoint
    from tests.test_data import _fake_parquet

    _fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--save-audio", "--backend", "bigvgan"])
    ds = MelDataset(data, min_frames=1)
    cfg = load_config("configs/tts.yaml", [f"data.root={data}", "model.gen.hidden=32", "model.gen.depth=1",
                                           "model.gen.heads=2", "model.gen.noise_coords=2", "model.text.d=16",
                                           "model.text.layers=1", "model.text.ffn=32", "model.text.spk_dim=8"])
    tts = DriftingTTS(cfg.model, num_speakers=ds.num_speakers)
    save_checkpoint(tmp_path / "tts.pt", ema=tts.state_dict(), config=cfg.to_dict(), num_speakers=ds.num_speakers,
                    stats={"mean": ds.stats.mean, "std": ds.stats.std, "backend": ds.backend})
    (tmp_path / "tiny_vocos.yaml").write_text(TINY_VOCOS)
    work = tmp_path / "voc"
    args = ["finetune-vocoder", "--config", "configs/vocoder_drift_vocos.yaml", "--workdir", str(work),
            f"data.root={data}", "data.min_quality=0", f"tts.path={tmp_path / 'tts.pt'}",
            f"vocoder.init={tmp_path / 'tiny_vocos.yaml'}", "vocoder.noise_channels=2",
            f"features.discriminators={{repo: null, pretrained: false, hparams: {TINY_MRD}}}",
            "features.stft={n_ffts: [256, 512], patch: [8, 2]}", "drift.samples=2", "drift.max_locations=32",
            "drift.pooled_locations=16", "train.cpu=true", "train.batch_size=2", "train.num_workers=0",
            "train.log_every=1", "train.save_every=2", "train.sample_every=2", "train.segment_frames=32",
            "train.gta_prob=1.0"]
    main(args + ["train.steps=2"])
    ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and "generator" in ck and "opt" in ck
    main(args + ["train.steps=3"])  # resumes from last.pt
    assert torch.load(work / "vocos_ft.pt", map_location="cpu", weights_only=False)["step"] == 3
    voc = Vocoder("cpu", finetuned=str(work / "vocos_ft.pt"), backend="bigvgan")
    out = voc(torch.randn(1, 100, 10) - 5)
    assert out.shape == (1, 10 * 256) and torch.isfinite(out).all()
