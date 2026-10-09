"""GTA fine-tuning of the DAC-VAE decoder (``vocoder.arch: vae_decoder``, issue #32): tiny DAC-VAE-shaped models and
discriminators on the CPU, no downloads. Only the audio path is trained; the watermark stays frozen and is still
added. The decoder is non-causal, so training windows get context on both sides."""

import pytest
import torch

from drifting_tts import finetune_vocoder as fv
from drifting_tts import latents
from drifting_tts.config import load_config
from drifting_tts.latents.dacvae import DACVAE, DACVAEBackend, DecoderBlock
from drifting_tts.latents.vocoder import LatentVocoder, load_decoder_checkpoint
from tests.test_decoder_finetune import DISC
from tests.test_latent_tts import latent_root  # noqa: F401 (module fixture: a latent root made with a fake VAE)

pytest.importorskip("nnAudio")  # the CQT discriminator


def tiny_dacvae(dim: int = 6) -> DACVAE:
    """DAC-VAE's frame layout (48 kHz in and out, 25 Hz frames through 12·10·8·2, a 150 Hz watermark branch through
    8·5·4·2) with a few channels."""
    torch.manual_seed(0)
    model = DACVAE(encoder_dim=2, latent_dim=16, decoder_dim=96, codebook_dim=dim, wm_latent=8)
    with torch.no_grad():  # a watermark well below the audio, as released (a random one is as loud as the audio)
        for p in model.decoder.wm_model.decoder_block.post.parameters():
            p.mul_(1e-3)
    return model


@pytest.fixture
def tiny_backend(monkeypatch):
    """``load_backend("dacvae", ...)`` builds the tiny model (released weights replaced by ``stock``); the
    discriminators get the tiny config."""
    stock = {k: v.clone() for k, v in tiny_dacvae().state_dict().items()}

    def load(name, device="cpu", decoder=None, **kw):
        assert name == "dacvae" and not kw
        m = tiny_dacvae()
        m.load_state_dict(stock)
        return DACVAEBackend("cpu", m, decoder=decoder)

    monkeypatch.setattr(latents, "load_backend", load)
    monkeypatch.setattr("drifting_tts.latents.vocoder.load_backend", load)
    monkeypatch.setattr(fv, "discriminator_hparams", lambda repo, overrides=None: {**DISC, **(overrides or {})})
    return load


def _cfg(*overrides):
    return load_config("configs/vocoder_dacvae_decoder.yaml", ["vocoder.pretrained_discriminators=false",
                                                              "train.segment_frames=3", "train.context_frames=4",
                                                              "train.right_context_frames=4", *overrides])


def _watermark_keys(decoder: torch.nn.Module) -> set[str]:
    keys = {f"wm_model.{k}" for k in decoder.wm_model.state_dict()}
    for i in range(1, len(decoder.model)):
        for j in DecoderBlock.WM_DOWN + DecoderBlock.WM_UP:
            keys |= {f"model.{i}.block.{j}.{k}" for k in decoder.model[i].block[j].state_dict()}
    return keys


def test_trainable_decoder_freezes_the_watermark(tiny_backend):
    be = tiny_backend("dacvae")
    dec = be.trainable_decoder()
    trained = {k for k, p in dec.named_parameters() if p.requires_grad}
    wm = _watermark_keys(dec)
    assert trained and wm and trained.isdisjoint(wm) and trained | wm == set(dec.state_dict())
    assert all(k.startswith("model.") and (k.startswith("model.0.") or int(k.split(".")[3]) in DecoderBlock.MAIN)
               for k in trained)
    assert not any(p.requires_grad for p in [*be.model.encoder.parameters(), *be.model.quantizer.parameters()])
    z = torch.randn(2, 6, 5)
    y, w = be.watermark_components(z)
    with torch.no_grad():
        out = be.decode_train(z)
    assert out.shape == y.shape == (2, 5 * 1920) and w.abs().max() > 0
    torch.testing.assert_close(out, y + w)  # the watermark is added in training as at inference
    torch.testing.assert_close(be.decode(z), (y + w).clamp(-1, 1))


def test_latent_windows_two_sided_context():
    torch.manual_seed(0)
    B, dim, repeat, hop, seg, left, right = 6, 3, 4, 960, 5, 4, 3
    native = torch.randn(B, dim, 30)
    frames = native.repeat_interleave(repeat, -1)
    n = torch.tensor([30, 30, 12, 13, 25, 14])
    audio = torch.arange(30 * hop).float().repeat(B, 1)
    width, seen = left + seg + right, set()
    for _ in range(30):
        z, offsets, y = fv.latent_windows(frames, audio, n * repeat, repeat, seg, left, hop, right)
        assert z.shape == (B, dim, width) and y.shape == (B, seg * hop)
        for i in range(B):
            s = int(y[i, 0]) // hop  # the target audio starts on a VAE frame
            assert y[i, 0] == s * hop and s + seg + 1 <= n[i]
            a = s - int(offsets[i])
            assert a == min(max(0, s - left), int(n[i]) - width)  # full context, or the utterance start / end
            assert s - a >= min(left, s) and a + width - (s + seg) >= min(right, int(n[i]) - s - seg)
            torch.testing.assert_close(z[i], native[i, :, a: a + width])  # repeats averaged back
            seen.add("start" if a == 0 else "end" if a + width == n[i] else "inside")
    assert seen == {"start", "end", "inside"}
    with pytest.raises(ValueError, match="shorter than a window"):
        fv.latent_windows(frames, audio, torch.full((B,), 11 * repeat), repeat, seg, left, hop, right)


def test_windowed_decode_matches_whole_utterance(tiny_backend):
    """A segment decoded in its window (context on both sides, cropped after resampling to 24 kHz) is the same audio
    as the whole utterance through LatentVocoder: within the decoder's receptive field with 8 frames on each side,
    and far from it without right context."""
    gan = fv.VAEDecoderGAN(_cfg("train.context_frames=8", "train.right_context_frames=8",
                                "vocoder.weight_norm=false"), "cpu", "dacvae", 4)
    assert gan.right == 8 and gan.min_frames == 4 * (8 + 3 + 8)
    voc = LatentVocoder("dacvae", "cpu", repeat=4)
    torch.manual_seed(1)
    frames = torch.randn(1, 6, 30).repeat_interleave(4, -1)
    whole = voc(frames)[0]
    assert whole.shape == (30 * 960,) and gan.audio_samples(120) == whole.numel()
    audio = torch.arange(30 * 960).float()[None]  # sample indices: tell where each segment starts

    def worst_snr(right: int) -> float:
        gan.right, snr = right, []
        torch.manual_seed(2)
        for _ in range(8):
            x, y = gan.segments(frames, audio, torch.tensor([120]))
            start = int(y[0, 0])
            with torch.no_grad():
                seg = gan.forward(x)[0, 0].clamp(-1, 1)
            ref = whole[start: start + seg.numel()]
            snr.append(10 * torch.log10(ref.pow(2).sum() / (ref - seg).pow(2).sum()).item())
        return min(snr)

    assert worst_snr(8) > 90 and worst_snr(1) < 30  # one frame on the right: about 18 dB


def test_training_keeps_the_watermark(tiny_backend, tmp_path):
    """A few large optimizer steps change the audio path only; the export holds the released watermark weights, its
    output is the trained audio plus the released watermark of that audio, and a changed watermark is refused."""
    gan = fv.VAEDecoderGAN(_cfg("train.lr=1e-2", "train.warmup_steps=0"), "cpu", "dacvae", 4)
    stock = tiny_backend("dacvae")
    assert set(gan.frozen) == _watermark_keys(gan.gen)  # exactly the watermark is frozen
    torch.manual_seed(0)
    frames, audio = torch.randn(2, 6, 4 * 20), 0.1 * torch.randn(2, 20 * 960)
    for step in range(3):
        x, y = gan.segments(frames, audio, torch.tensor([80, 64]))
        gan.step(x, y, step)
    gan.export(tmp_path / "decoder_ft.pt", step=3)
    ck = load_decoder_checkpoint(tmp_path / "decoder_ft.pt")
    released = stock.model.decoder.state_dict()
    wm = _watermark_keys(stock.model.decoder)
    assert set(ck["decoder"]) == set(released)
    assert all(torch.equal(ck["decoder"][k], released[k]) for k in wm)  # bit for bit
    assert not torch.allclose(ck["decoder"]["model.0.weight"], released["model.0.weight"])
    tuned = DACVAEBackend("cpu", tiny_dacvae(), decoder=ck)
    z = torch.randn(1, 6, 12)
    y, w = tuned.watermark_components(z)
    y0, w0 = stock.watermark_components(z)
    assert not torch.allclose(y, y0, atol=1e-4)
    with torch.no_grad():  # the released watermark generator, applied to the fine-tuned audio
        torch.testing.assert_close(w, stock.model.decoder.watermark(y[:, None], stock.message)[:, 0])
    with torch.no_grad():
        gan.gen.wm_model.decoder_block.post[1].bias.add_(1e-3)
    with pytest.raises(RuntimeError, match="frozen decoder weights changed"):
        gan.export(tmp_path / "bad.pt", step=4)


def test_export_and_load(tiny_backend, tmp_path):
    gan = fv.VAEDecoderGAN(_cfg(), "cpu", "dacvae", 4)
    keys = gan.gen.state_dict()
    assert any("parametrizations" in k for k in keys)  # the audio path is trained with weight norm ...
    assert not any("parametrizations" in k for k in keys if k.startswith("wm_model"))  # ... the watermark is not
    with torch.no_grad():
        for name, p in gan.gen.named_parameters():
            if name.endswith("original0"):  # the magnitudes: a weight-norm-only change
                p.mul_(1.5)
    gan.export(tmp_path / "decoder_ft.pt", step=7)
    ck = load_decoder_checkpoint(tmp_path / "decoder_ft.pt")  # weights_only=True
    assert set(ck) == {"decoder", "backend", "target_rate", "sample_rate", "step"}
    assert ck["backend"] == "dacvae" and ck["target_rate"] == 48000 and ck["sample_rate"] == 24000
    assert ck["step"] == 7
    assert not any("parametrizations" in k or k.endswith(("weight_g", "weight_v")) for k in ck["decoder"])
    z = torch.randn(1, 6, 12)
    with torch.no_grad():
        trained = gan.be.decode_train(z)
    tuned = LatentVocoder("dacvae", "cpu", repeat=1, sample_rate=None, decoder=str(tmp_path / "decoder_ft.pt"))
    stock = LatentVocoder("dacvae", "cpu", repeat=1, sample_rate=None)
    assert tuned.name == "dacvae (fine-tuned decoder)" and stock.name == "dacvae"
    torch.testing.assert_close(tuned(z), trained.clamp(-1, 1), atol=1e-5, rtol=1e-4)
    assert not torch.allclose(stock(z), tuned(z), atol=1e-3)
    with pytest.raises(ValueError, match="voxcpm2 decoder"):
        LatentVocoder("dacvae", "cpu", decoder={**ck, "backend": "voxcpm2"})


def test_finetune_dacvae_decoder_cpu(tiny_backend, latent_root, tmp_path):  # noqa: F811
    """extract-latents (fake VAE) -> tiny latent TTS on "dacvae" latents -> GTA fine-tuning of a tiny DAC-VAE decoder
    -> resume -> snapshots and decoder_ft.pt in Synthesizer."""
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
    stats = {**ds.stats.to_dict(), "backend": "dacvae", "latent_repeat": ds.latent_repeat}
    save_checkpoint(tmp_path / "tts.pt", ema=tts.state_dict(), config=cfg.to_dict(), num_speakers=ds.num_speakers,
                    stats=stats, n_mels=ds.dim)
    work = tmp_path / "dec"
    args = ["finetune-vocoder", "--config", "configs/vocoder_dacvae_decoder.yaml", "--workdir", str(work),
            f"data.root={root}", f"tts.path={tmp_path / 'tts.pt'}", "vocoder.pretrained_discriminators=false",
            "train.cpu=true", "train.batch_size=2", "train.num_workers=0", "train.log_every=1",
            "train.save_every=2", "train.sample_every=2", "train.segment_frames=4", "train.context_frames=3",
            "train.right_context_frames=3", "train.gta_prob=1.0"]
    main(args + ["train.steps=2"])
    ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and {"generator", "mpd", "mrd", "opt_g", "opt_d"} <= set(ck)
    main(args + ["train.steps=3"])  # resumes from last.pt
    path = work / "decoder_ft.pt"
    assert load_decoder_checkpoint(path)["step"] == 3
    assert load_decoder_checkpoint(work / "decoder_ft_2.pt")["step"] == 2  # keep_snapshots

    text = "merhaba dünya."
    tuned = Synthesizer(tmp_path / "tts.pt", "cpu", vocoder=str(path))
    stock = Synthesizer(tmp_path / "tts.pt", "cpu")
    assert tuned.vocoder.name == "dacvae (fine-tuned decoder)" and stock.vocoder.name == "dacvae"
    (w1, _), (w0, _) = tuned(text, speaker=0), stock(text, speaker=0)
    assert w1.shape == w0.shape and torch.isfinite(w1).all() and not torch.allclose(w1, w0)
