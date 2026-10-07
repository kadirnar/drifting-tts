import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from drifting_tts.latents import BACKENDS, LatentStats, load_backend, stream_decode
from drifting_tts.latents.dacvae import DACVAE, DACVAEBackend
from drifting_tts.latents.dacvae import REPO as DACVAE_REPO
from drifting_tts.latents.dacvae import REVISION as DACVAE_REVISION
from drifting_tts.latents.layers import fold_weight_norm
from drifting_tts.latents.voxcpm import MODELS as VOXCPM
from drifting_tts.latents.voxcpm import AudioVAE, VoxCPMBackend


def tiny_dacvae() -> DACVAEBackend:
    torch.manual_seed(0)
    model = DACVAE(encoder_dim=4, encoder_rates=[2, 4], latent_dim=16, decoder_dim=48, decoder_rates=[4, 2],
                   codebook_dim=8, wm_rates=[2, 2], wm_latent=8, sample_rate=800)
    return DACVAEBackend("cpu", model)


def tiny_voxcpm(v2: bool) -> VoxCPMBackend:
    torch.manual_seed(0)
    config = {"encoder_dim": 4, "encoder_rates": [2, 3], "latent_dim": 4, "sample_rate": 600}
    config.update({"decoder_dim": 32, "decoder_rates": [3, 2, 2], "out_sample_rate": 1200} if v2
                  else {"decoder_dim": 16, "decoder_rates": [3, 2]})
    return VoxCPMBackend("voxcpm2" if v2 else "voxcpm1.5", "cpu", AudioVAE(config, v2=v2))


@pytest.mark.parametrize("make", [tiny_dacvae, lambda: tiny_voxcpm(True), lambda: tiny_voxcpm(False)],
                         ids=["dacvae", "voxcpm2", "voxcpm1.5"])
def test_frames_and_samples(make):
    be = make()
    assert be.hop_in * be.frame_rate == be.input_rate and be.hop_out * be.frame_rate == be.output_rate
    for sr, n in ((be.input_rate, 123), (2 * be.input_rate, 251), (be.input_rate, 5 * be.hop_in)):
        wav = 0.1 * torch.randn(2, n)
        z = be.encode(wav, sr)
        assert z.shape == (2, be.dim, be.num_frames(n, sr))
        y = be.decode(z)
        assert y.shape == (2, z.shape[-1] * be.hop_out) and y.abs().max() <= 1
        assert torch.equal(y, be.decode(z))  # deterministic (DAC-VAE: fixed watermark message)
    mean, std = be.posterior(wav[0], sr)
    assert torch.equal(mean, be.encode(wav[0], sr)) and (std > 0).all()


def test_dacvae_watermark_message():
    be = tiny_dacvae()
    z = be.encode(0.1 * torch.randn(200), 800)
    other = DACVAEBackend("cpu", be.model, message=torch.ones(16))
    assert not torch.allclose(be.decode(z), other.decode(z))


@pytest.mark.parametrize("v2", [True, False])
def test_voxcpm_is_causal(v2):
    be = tiny_voxcpm(v2)
    wav = 0.1 * torch.randn(1, 40 * be.hop_in)
    later = wav.clone()
    later[:, 25 * be.hop_in:] += 0.1
    z, z2 = be.encode(wav, be.input_rate), be.encode(later, be.input_rate)
    assert torch.allclose(z[..., :25], z2[..., :25]) and not torch.allclose(z[..., 25:], z2[..., 25:])
    z2 = z.clone()
    z2[..., 30:] += 1
    y, y2 = be.decode(z), be.decode(z2)
    cut = 30 * be.hop_out
    assert torch.allclose(y[..., :cut], y2[..., :cut]) and not torch.allclose(y[..., cut:], y2[..., cut:])


def test_voxcpm2_rate_condition():
    be = tiny_voxcpm(True)
    z = be.encode(0.1 * torch.randn(600), 600)
    full = be.decode(z)
    be.target_rate = 1200  # the model's out_sample_rate: same bucket as the default
    assert torch.equal(full, be.decode(z))
    be.target_rate = 25_000
    assert not torch.allclose(full, be.decode(z))


def test_stream_decode():
    be = tiny_voxcpm(True)
    z = torch.randn(1, be.dim, 100)
    whole = be.decode(z)[0]
    pieces = list(stream_decode(be, z, first=8, chunk=16, context=64))  # beyond the receptive field (~55 frames)
    assert [len(p) for p in pieces] == [8 * be.hop_out] + [16 * be.hop_out] * 5 + [12 * be.hop_out]
    assert torch.allclose(torch.cat(pieces), whole, atol=1e-5)
    assert not torch.allclose(torch.cat(list(stream_decode(be, z, 8, 16, context=0))), whole, atol=1e-3)
    dac = tiny_dacvae()  # non-causal: right context too; same length
    z = torch.randn(1, dac.dim, 30)
    assert torch.cat(list(stream_decode(dac, z, 4, 8, 4))).shape == dac.decode(z)[0].shape


def test_fold_weight_norm():
    torch.manual_seed(0)
    for plain in (torch.nn.Conv1d(4, 6, 3, groups=2), torch.nn.ConvTranspose1d(4, 6, 4, stride=2)):
        wn = torch.nn.utils.weight_norm(type(plain)(**{k: getattr(plain, k) for k in
                                                       ("in_channels", "out_channels", "kernel_size", "stride",
                                                        "groups")}))
        with torch.no_grad():
            wn.weight_g.mul_(torch.rand_like(wn.weight_g) + 0.5)
        plain.load_state_dict(fold_weight_norm(wn.state_dict()))
        x = torch.randn(2, 4, 9)
        assert torch.allclose(plain(x), wn(x), atol=1e-6)


def test_latent_stats():
    torch.manual_seed(0)
    zs = [torch.randn(3, 50) * torch.tensor([[1.0], [2.0], [0.5]]) + torch.tensor([[0.0], [3.0], [-1.0]]),
          torch.randn(2, 3, 20)]
    stats = LatentStats.from_latents(zs)
    flat = torch.cat([zs[0], zs[1].transpose(0, 1).flatten(1)], 1)
    assert torch.allclose(stats.mean, flat.mean(1), atol=1e-5)
    assert torch.allclose(stats.std, flat.std(1, unbiased=False), atol=1e-5)
    n = stats.normalize(flat[None])
    assert torch.allclose(n.mean(-1), torch.zeros(3), atol=1e-5) and torch.allclose(n.std(-1, unbiased=False),
                                                                                     torch.ones(3), atol=1e-4)
    back = LatentStats.from_dict(stats.to_dict())
    assert torch.allclose(back.denormalize(n), flat[None], atol=1e-4)


def test_bigvgan_mel_frames():
    pytest.importorskip("torchaudio")
    be = load_backend("bigvgan", "cpu")
    for sr, n in ((24_000, 24_077), (16_000, 16_000)):
        assert be.encode(0.1 * torch.randn(n), sr).shape == (1, 100, be.num_frames(n, sr))
    with pytest.raises(ValueError):
        load_backend("nope")


WEIGHTS = {"dacvae": (DACVAE_REPO, "weights.pth", DACVAE_REVISION),
           **{name: (repo, "audiovae.pth", revision) for name, (repo, revision) in VOXCPM.items()},
           "bigvgan": ("nvidia/bigvgan_v2_24khz_100band_256x", "bigvgan_generator.pt", None)}


@pytest.mark.parametrize("name", BACKENDS)
def test_released_backend_roundtrip(name):
    repo, filename, revision = WEIGHTS[name]
    if not isinstance(try_to_load_from_cache(repo, filename, revision=revision), str):
        pytest.skip(f"{name} weights not cached (large download)")
    if name == "bigvgan":
        pytest.importorskip("librosa")
    be = load_backend(name, "cpu")
    wav = 0.1 * torch.sin(torch.arange(12_000) / 24_000 * 2 * torch.pi * 220)
    z = be.encode(wav, 24_000)
    y = be.decode(z)
    assert z.shape == (1, be.dim, be.num_frames(len(wav), 24_000)) and y.shape == (1, z.shape[-1] * be.hop_out)
    assert torch.isfinite(y).all() and y.abs().max() > 0.01
