"""GTA fine-tuning of a VAE decoder (``finetune-vocoder`` with ``vocoder.arch: vae_decoder``, issue #27): tiny VoxCPM2-
shaped models and discriminators on the CPU, no downloads."""

import pytest
import torch

from drifting_tts import finetune_vocoder as fv
from drifting_tts import latents
from drifting_tts.config import load_config
from drifting_tts.latents.vocoder import LatentVocoder, load_decoder_checkpoint
from drifting_tts.latents.voxcpm import AudioVAE, VoxCPMBackend
from tests.test_latent_tts import latent_root  # noqa: F401 (module fixture: a latent root made with a fake VAE)
from tests.test_vocoder import TINY

# BigVGAN-v2's discriminator config.json, shrunk (no download)
DISC = {**TINY, "sampling_rate": 24_000, "adam_b1": 0.8, "adam_b2": 0.99, "use_cqtd_instead_of_mrd": True,
        "cqtd_filters_scale": 1, "cqtd_dilations": [1, 2, 4]}


def tiny_voxcpm2(dim: int = 6) -> AudioVAE:
    """VoxCPM2's frame layout (16 kHz in, 25 Hz frames, 48 kHz out through 8·6·5·2·2·2) with a few channels."""
    torch.manual_seed(0)
    return AudioVAE({"encoder_dim": 2, "latent_dim": dim, "decoder_dim": 64}, v2=True)


@pytest.fixture
def tiny_backend(monkeypatch):
    """``load_backend("voxcpm2", ...)`` builds the tiny model (with the released-weights path replaced); the
    discriminators get the tiny config."""
    model = tiny_voxcpm2()
    stock = {k: v.clone() for k, v in model.state_dict().items()}

    def load(name, device="cpu", **kw):
        m = tiny_voxcpm2()
        m.load_state_dict(stock)
        return VoxCPMBackend("voxcpm2", "cpu", m, **kw)

    monkeypatch.setattr(latents, "load_backend", load)
    monkeypatch.setattr("drifting_tts.latents.vocoder.load_backend", load)
    monkeypatch.setattr(fv, "discriminator_hparams", lambda repo, overrides=None: {**DISC, **(overrides or {})})
    return load


def _cfg(*overrides):
    return load_config("configs/vocoder_voxcpm2_decoder.yaml", ["vocoder.pretrained_discriminators=false",
                                                               "train.segment_frames=3", "train.context_frames=4",
                                                               *overrides])


def test_latent_windows_start_on_vae_frames():
    torch.manual_seed(0)
    B, dim, repeat, hop, seg, ctx = 6, 3, 4, 960, 5, 4
    native = torch.randn(B, dim, 30)
    frames = native.repeat_interleave(repeat, -1)
    frame_len = torch.tensor([30, 30, 12, 10, 25, 11]) * repeat
    audio = torch.arange(30 * hop).float().repeat(B, 1)
    for _ in range(20):
        z, offsets, y = fv.latent_windows(frames, audio, frame_len, repeat, seg, ctx, hop)
        assert z.shape == (B, dim, ctx + seg + 1) and y.shape == (B, seg * hop)
        for i in range(B):
            s = int(y[i, 0]) // hop  # the target audio starts on a VAE frame
            assert y[i, 0] == s * hop and s + seg + 1 <= frame_len[i] // repeat
            a = s - int(offsets[i])
            assert a == max(0, s - ctx)  # full left context, or the utterance start
            torch.testing.assert_close(z[i], native[i, :, a: a + ctx + seg + 1])  # repeats averaged back
    with pytest.raises(ValueError, match="shorter than a window"):
        fv.latent_windows(frames, audio, torch.full((B,), 9 * repeat), repeat, seg, ctx, hop)


def test_windowed_decode_matches_whole_utterance(tiny_backend):
    """A segment decoded in its window (left context, cropped after resampling to 24 kHz) is the same audio as the
    whole utterance through LatentVocoder: exactly when the window starts at the utterance start, and within the
    decoder's receptive field otherwise."""
    pytest.importorskip("nnAudio")
    gan = fv.VAEDecoderGAN(_cfg("train.context_frames=24", "vocoder.weight_norm=false"), "cpu", "voxcpm2", 4)
    voc = LatentVocoder("voxcpm2", "cpu", repeat=4)
    torch.manual_seed(1)
    frames = torch.randn(1, 6, 40).repeat_interleave(4, -1)
    whole = voc(frames)[0]
    assert whole.shape == (40 * 960,) and gan.audio_samples(160) == whole.numel()
    audio = torch.arange(40 * 960).float()[None]  # sample indices: tell where each segment starts
    seen = set()
    for _ in range(30):
        x, y = gan.segments(frames, audio, torch.tensor([160]))
        start = int(y[0, 0])
        with torch.no_grad():
            seg = gan.forward(x)[0, 0]
        ref = whole[start: start + seg.numel()]
        tol = 1e-5 if start // 960 < 24 else 1e-4
        assert (seg.clamp(-1, 1) - ref).abs().max() < tol
        seen.add(start // 960 < 24)
    assert seen == {True, False}


def test_weight_norm_starts_from_released_directions(tiny_backend, monkeypatch):
    """Training re-parametrizes the convolutions with the released (unfolded) ``weight_g`` / ``weight_v`` where they
    reproduce the weights, so the directions keep their trained scale; elsewhere ``v`` is the weight itself."""
    pytest.importorskip("nnAudio")
    convs = (torch.nn.Conv1d, torch.nn.ConvTranspose1d)

    def released(self):
        out = {}
        for name, m in self.model.decoder.named_modules():
            if isinstance(m, convs):
                w = m.weight.detach()
                out[name] = (w.flatten(1).norm(dim=1).view(-1, *[1] * (w.ndim - 1)), 10 * w)
        out["model.0"] = (out["model.0"][0], torch.randn_like(out["model.0"][1]))  # does not reproduce the weights
        return out

    monkeypatch.setattr(VoxCPMBackend, "decoder_weight_norm", released)
    stock = tiny_backend("voxcpm2")
    gan = fv.VAEDecoderGAN(_cfg(), "cpu", "voxcpm2", 4)
    for name, m in gan.gen.named_modules():
        if isinstance(m, convs):
            p, w = m.parametrizations.weight, stock.model.decoder.get_submodule(name).weight
            torch.testing.assert_close(m.weight, w, atol=1e-6, rtol=1e-4)
            scale = p.original1.flatten(1).norm(dim=1) / p.original0.flatten()
            torch.testing.assert_close(scale, torch.full_like(scale, 1.0 if name == "model.0" else 10.0))
    z = torch.randn(1, 6, 8)
    with torch.no_grad():
        torch.testing.assert_close(gan.be.decode_train(z), stock.decode_train(z), atol=1e-5, rtol=1e-4)


def test_export_folds_weight_norm(tiny_backend, tmp_path):
    pytest.importorskip("nnAudio")
    gan = fv.VAEDecoderGAN(_cfg(), "cpu", "voxcpm2", 4)
    assert any("parametrizations" in k for k in gan.gen.state_dict())  # trained with weight norm
    with torch.no_grad():
        for name, p in gan.gen.named_parameters():
            if name.endswith("original0"):  # the magnitudes: a weight-norm-only change
                p.mul_(1.5)
    gan.export(tmp_path / "decoder_ft.pt", step=7)
    ck = load_decoder_checkpoint(tmp_path / "decoder_ft.pt")  # weights_only=True
    assert set(ck) == {"decoder", "backend", "target_rate", "sample_rate", "step"}
    assert ck["backend"] == "voxcpm2" and ck["target_rate"] == 48000 and ck["sample_rate"] == 24000
    assert not any("parametrizations" in k or k.endswith(("weight_g", "weight_v")) for k in ck["decoder"])
    z = torch.randn(1, 6, 12)
    with torch.no_grad():
        trained = gan.be.decode_train(z)
    tuned, stock = LatentVocoder("voxcpm2", "cpu", repeat=1, sample_rate=None, decoder=ck), \
        LatentVocoder("voxcpm2", "cpu", repeat=1, sample_rate=None)
    torch.testing.assert_close(tuned(z), trained.clamp(-1, 1), atol=1e-5, rtol=1e-4)
    assert not torch.allclose(stock(z), tuned(z), atol=1e-3)
    with pytest.raises(ValueError, match="dacvae decoder"):
        LatentVocoder("voxcpm2", "cpu", decoder={**ck, "backend": "dacvae"})


def test_finetune_voxcpm2_decoder_cpu(tiny_backend, latent_root, tmp_path):  # noqa: F811
    """extract-latents (fake VAE) -> tiny latent TTS -> GTA fine-tuning of a tiny VoxCPM2 decoder with tiny released-
    style discriminators -> resume -> decoder_ft.pt in LatentVocoder and Synthesizer."""
    pytest.importorskip("nnAudio")
    from drifting_tts.cli import main
    from drifting_tts.data import MelDataset
    from drifting_tts.models.tts import DriftingTTS
    from drifting_tts.synthesize import Synthesizer
    from drifting_tts.utils import save_checkpoint

    root = latent_root / "lat"
    ds = MelDataset(root, min_frames=1)
    cfg = load_config("configs/tts.yaml", [f"data.root={root}", "model.gen.hidden=32", "model.gen.depth=1",
                                           "model.gen.heads=2", "model.gen.noise_coords=2", "model.gen.patch=4",
                                           "model.text.d=16", "model.text.layers=1", "model.text.ffn=32",
                                           "model.text.spk_dim=8", "model.pitch.enabled=true"])
    tts = DriftingTTS(cfg.model, num_speakers=ds.num_speakers, n_mels=ds.dim)  # untrained: GTA needs only shapes
    stats = {**ds.stats.to_dict(), "backend": ds.backend, "latent_repeat": ds.latent_repeat}
    save_checkpoint(tmp_path / "tts.pt", ema=tts.state_dict(), config=cfg.to_dict(), num_speakers=ds.num_speakers,
                    stats=stats, n_mels=ds.dim)
    work = tmp_path / "dec"
    args = ["finetune-vocoder", "--config", "configs/vocoder_voxcpm2_decoder.yaml", "--workdir", str(work),
            f"data.root={root}", f"tts.path={tmp_path / 'tts.pt'}", "vocoder.pretrained_discriminators=false",
            "train.cpu=true", "train.batch_size=2", "train.num_workers=0", "train.log_every=1",
            "train.save_every=2", "train.sample_every=2", "train.segment_frames=4", "train.context_frames=3",
            "train.gta_prob=1.0"]
    main(args + ["train.steps=2"])
    ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and {"generator", "mpd", "mrd", "opt_g", "opt_d"} <= set(ck)
    main(args + ["train.steps=3"])  # resumes from last.pt
    path = work / "decoder_ft.pt"
    assert load_decoder_checkpoint(path)["step"] == 3

    text = "merhaba dünya."
    tuned = Synthesizer(tmp_path / "tts.pt", "cpu", vocoder=str(path))
    stock = Synthesizer(tmp_path / "tts.pt", "cpu")
    assert "fine-tuned" in tuned.vocoder.name and stock.vocoder.name == "voxcpm2"
    (w1, _), (w0, _) = tuned(text, speaker=0), stock(text, speaker=0)
    assert w1.shape == w0.shape and torch.isfinite(w1).all() and not torch.allclose(w1, w0)
    with pytest.raises(ValueError, match="fine-tuned decoder checkpoint"):
        Synthesizer(tmp_path / "tts.pt", "cpu", vocoder="bigvgan-v2")
