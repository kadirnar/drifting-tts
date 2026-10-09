"""SLM adversary (drifting_tts/slm.py): shapes, update separation, gradient paths, and the disabled default."""

import pytest
import torch

from drifting_tts.cli import main
from drifting_tts.config import load_config
from drifting_tts.data import MelStats
from drifting_tts.models.mae import MelMAE
from drifting_tts.models.tts import DriftingTTS
from drifting_tts.slm import SLMAdversary, WavLMDiscriminator, audio_segments, needs_audio, stack_layers
from drifting_tts.train import CropBank, build_loader, training_step
from drifting_tts.utils import save_checkpoint
from drifting_tts.vocoder import Vocoder, build_vocos
from tests.test_data import _fake_parquet
from tests.test_train import TINY, TINY_VOCOS

TINY_VOCOS_SAME = TINY_VOCOS.replace("hop_length: 256, padding: center}\n", "hop_length: 256, padding: same}\n", 1)
STATS = MelStats(-5.0, 2.0)
F = 64  # crop frames (TINY)


def tiny_wavlm():
    from transformers import WavLMConfig, WavLMModel

    torch.manual_seed(0)
    return WavLMModel(WavLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                                  conv_dim=(8, 8), conv_stride=(5, 4), conv_kernel=(10, 4), num_conv_pos_embeddings=8,
                                  num_conv_pos_embedding_groups=2, num_buckets=16, max_bucket_distance=40))


def tiny_vocoder(tmp_path) -> Vocoder:
    (tmp_path / "tiny_vocos_same.yaml").write_text(TINY_VOCOS_SAME)
    model = build_vocos(str(tmp_path / "tiny_vocos_same.yaml"))
    assert model.head.istft.padding == "same"
    return Vocoder.wrap(model.eval(), "vocos", "cpu", "bigvgan", "tiny")


def slm_cfg(*overrides):
    return load_config("configs/tts.yaml", TINY + ["slm.enabled=true", "slm.trim_frames=4", "slm.disc_warmup=0",
                                                   "slm.dtype=fp32", *overrides])


def adversary(tmp_path, cfg) -> SLMAdversary:
    return SLMAdversary(cfg.slm, tiny_vocoder(tmp_path), tiny_wavlm(), STATS, "cpu")


def test_discriminator_and_layer_stacking_shapes():
    hidden = tuple(torch.randn(3, 20, 16) for _ in range(5))
    x = stack_layers(hidden)
    assert x.shape == (3, 5 * 16, 20) and torch.equal(x[:, 16:32], hidden[1].transpose(1, 2))
    d = WavLMDiscriminator(dim=16, layers=5, channels=4)
    assert d(x).shape == (3, 20)


def test_audio_segments_follow_bigvgan_framing():
    audio = torch.arange(2 * 4096, dtype=torch.float32).view(2, 4096)
    seg = audio_segments(audio, torch.tensor([0, 3]), 2)
    assert seg.shape == (2, 512) and seg[0, 0] == 0 and seg[1, 0] == 4096 + 3 * 256 and seg[1, -1] == 4096 + 5 * 256 - 1


def test_vocoder_differentiable_matches_inference_path(tmp_path):
    voc = tiny_vocoder(tmp_path)
    mel = torch.randn(2, 100, 12, requires_grad=True)
    wav = voc.differentiable(mel)
    assert wav.shape == (2, 12 * 256) and torch.allclose(wav.detach().clamp(-1, 1), voc(mel))
    wav.square().sum().backward()
    assert mel.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="Vocos vocoder on BigVGAN mels"):
        Vocoder.wrap(voc.model, "vocos", "cpu", "vocos").differentiable(mel)


def test_step_updates_the_discriminator_only_and_returns_generator_terms(tmp_path):
    torch.manual_seed(0)
    slm = adversary(tmp_path, slm_cfg("slm.fm_weight=0.5", "slm.chunk=4"))
    slm.separate_terms = True
    B, c = 2, 3
    fake = torch.randn(B * c, 100, F, requires_grad=True)
    real_mel, real_audio = torch.randn(B, 100, F), 0.1 * torch.randn(B, F * 256)
    before = [p.detach().clone() for p in slm.disc.parameters()]
    loss, metrics, terms = slm.step(fake, real_mel, real_audio)
    assert slm.steps == 1 and loss.requires_grad and loss.ndim == 0
    assert {"slm_disc", "slm_d_real", "slm_d_fake", "slm_adv", "slm_fm"} <= set(metrics)
    assert torch.allclose(loss, terms["adv"] + 0.5 * terms["fm"]) and terms["fm"] > 0
    assert any(not torch.equal(a, p) for a, p in zip(before, slm.disc.parameters()))  # the discriminator stepped
    g_terms = [torch.autograd.grad(terms[k], fake, retain_graph=True)[0] for k in ("adv", "fm")]
    assert torch.allclose(torch.autograd.grad(loss, fake, retain_graph=True)[0], g_terms[0] + 0.5 * g_terms[1])
    # the generator terms reach the generated mels, not the discriminator, vocoder or WavLM
    disc_grads = [p.grad.clone() for p in slm.disc.parameters()]
    assert all(g is None for g in torch.autograd.grad(terms["adv"], list(slm.disc.parameters()), retain_graph=True,
                                                      allow_unused=True))
    loss.backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0 and torch.isfinite(fake.grad).all()
    assert all(torch.equal(g, p.grad) for g, p in zip(disc_grads, slm.disc.parameters()))
    frozen = [*slm.wavlm.parameters(), *slm.vocoder.model.parameters()]
    assert all(p.grad is None and not p.requires_grad for p in frozen)
    assert all(p.requires_grad for p in slm.disc.parameters())  # unfrozen again for its next update


def test_disc_warmup_trains_the_discriminator_alone(tmp_path):
    slm = adversary(tmp_path, slm_cfg("slm.disc_warmup=2", "slm.real=vocoded"))
    fake = torch.randn(2, 100, F, requires_grad=True)
    for step in range(3):
        loss, metrics, terms = slm.step(fake, torch.randn(2, 100, F))
        assert (loss.requires_grad, "slm_adv" in metrics) == ((step >= 2),) * 2
    assert slm.steps == 3
    before = [p.detach().clone() for p in slm.disc.parameters()]
    with torch.no_grad():  # a discriminator-only step from a no-grad caller still updates it
        loss, _, _ = slm.step(fake, torch.randn(2, 100, F))
    assert not loss.requires_grad and any(not torch.equal(a, p) for a, p in zip(before, slm.disc.parameters()))
    state = slm.state_dict()
    again = adversary(tmp_path, slm_cfg("slm.disc_warmup=2"))
    again.load_state_dict(state)
    assert again.steps == 4 and all(torch.equal(a, b) for a, b in zip(again.disc.parameters(), slm.disc.parameters()))


def test_chunked_gradients_match_one_pass(tmp_path):
    """``slm.chunk`` only bounds memory: the generator gradient and the discriminator update do not depend on it."""
    fake0, real_mel, real_audio = torch.randn(6, 100, F), torch.randn(2, 100, F), 0.1 * torch.randn(2, F * 256)
    out = []
    for chunk in (1, 4, 6):
        torch.manual_seed(0)
        slm = adversary(tmp_path, slm_cfg("slm.fm_weight=0.3", f"slm.chunk={chunk}"))
        torch.manual_seed(1)
        slm.disc = WavLMDiscriminator(16, 3, 64)  # identical initial discriminators
        slm.opt = torch.optim.SGD(slm.disc.parameters(), lr=1e-2)  # Adam would amplify float noise of ~0 grads
        fake = fake0.clone().requires_grad_(True)
        loss, metrics, _ = slm.step(fake, real_mel, real_audio)
        loss.backward()
        out.append((float(loss.detach()), fake.grad, [p.detach().clone() for p in slm.disc.parameters()]))
    for value, grad, disc in out[1:]:
        assert abs(value - out[0][0]) < 1e-5 and torch.allclose(grad, out[0][1], atol=1e-6, rtol=1e-4)
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(disc, out[0][2]))


def tiny_batch(audio: bool = True) -> dict:
    torch.manual_seed(1)
    mel_len = torch.tensor([80, 75, 70])
    batch = {"text": torch.randint(2, 39, (3, 20)), "text_len": torch.tensor([20, 18, 15]),
             "mel": torch.randn(3, 100, 80), "mel_len": mel_len, "spk": torch.tensor([0, 1, 0])}
    if audio:
        batch["audio"] = 0.1 * torch.randn(3, 80 * 256)
    return batch


@pytest.mark.parametrize("mode", ["subset", "extra", "vocoded"])
def test_training_step_gradient_paths(tmp_path, mode):
    over = {"subset": ["slm.crops_per_cond=2"], "extra": ["slm.crops_per_cond=2", "slm.alpha=1.0"],
            "vocoded": ["slm.real=vocoded"]}[mode]
    cfg = slm_cfg("slm.fm_weight=1.0", *over)
    torch.manual_seed(0)
    model = DriftingTTS(cfg.model, num_speakers=2)
    mae = MelMAE(n_mels=100, base_channels=8).eval().requires_grad_(False)
    slm = adversary(tmp_path, cfg)
    slm.separate_terms = True
    sizes, generate = [], model.generate
    model.generate = lambda z, *a, **kw: sizes.append(z.shape[0]) or generate(z, *a, **kw)
    gen_before = [p.detach().clone() for p in model.generator.parameters()]
    disc_before = [p.detach().clone() for p in slm.disc.parameters()]
    batch = tiny_batch(audio=mode != "vocoded")
    loss, metrics, info = training_step(model, mae, batch, CropBank(16, 100, F, "cpu"), cfg, "cpu", slm=slm)
    B, G = 3, cfg.drift.gen_per_cond
    assert sizes == [B * (G + 2 if mode == "extra" else G)]
    assert torch.isfinite(loss) and {"slm_disc", "slm_adv", "slm_fm"} <= set(metrics)
    # the discriminator was updated inside the step, the generator was not
    assert any(not torch.equal(a, p) for a, p in zip(disc_before, slm.disc.parameters()))
    assert all(torch.equal(a, p) for a, p in zip(gen_before, model.generator.parameters()))
    # the adversarial term trains the generator, and only through the generated crops
    gen = [p for p in model.generator.parameters() if p.requires_grad]
    g_adv = torch.autograd.grad(info["slm"]["adv"], gen, retain_graph=True, allow_unused=True)
    assert sum(float(g.abs().sum()) for g in g_adv if g is not None) > 0
    assert all(g is None for g in torch.autograd.grad(info["slm"]["adv"], list(slm.disc.parameters()),
                                                      retain_graph=True, allow_unused=True))
    disc_grads = [p.grad.clone() for p in slm.disc.parameters()]
    loss.backward()
    assert all(torch.equal(g, p.grad) for g, p in zip(disc_grads, slm.disc.parameters()))
    assert all(torch.isfinite(p.grad).all() for p in gen if p.grad is not None)


def test_disabled_slm_leaves_the_step_unchanged():
    """``slm.enabled: false`` (the default) runs the plain step: no audio loaded, no SLM metrics, no extra RNG."""
    base = load_config("configs/tts.yaml", TINY)
    assert base.slm.enabled is False and not needs_audio(base)
    outs = []
    for cfg in (base, load_config("configs/tts.yaml", TINY + ["slm=null"])):
        torch.manual_seed(0)
        model = DriftingTTS(cfg.model, num_speakers=2)
        mae = MelMAE(n_mels=100, base_channels=8).eval().requires_grad_(False)
        loss, metrics, _ = training_step(model, mae, tiny_batch(audio=False), CropBank(16, 100, F, "cpu"), cfg, "cpu")
        loss.backward()
        outs.append((loss.detach(), sorted(metrics), [p.grad.clone() for p in model.generator.parameters()],
                     torch.get_rng_state()))
    (l0, k0, g0, r0), (l1, k1, g1, r1) = outs
    assert torch.equal(l0, l1) and k0 == k1 and torch.equal(r0, r1) and all(torch.equal(a, b) for a, b in zip(g0, g1))
    assert not any(k.startswith("slm") for k in k0)


def _prepare_bigvgan(tmp_path):
    _fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "0", "--no-trim", "--backend", "bigvgan", "--save-audio"])
    mae_cfg = load_config("configs/mae.yaml", ["model.base_channels=8"])
    save_checkpoint(tmp_path / "mae.pt", ema=MelMAE(n_mels=100, base_channels=8).state_dict(),
                    config=mae_cfg.to_dict(), num_classes=0)
    return data


def test_train_with_slm_end_to_end_cpu(tmp_path):
    data = _prepare_bigvgan(tmp_path)
    voc = tiny_vocoder(tmp_path)
    save_checkpoint(tmp_path / "vocos_tiny.pt", vocos=voc.model.state_dict(),
                    init=str(tmp_path / "tiny_vocos_same.yaml"), mel="bigvgan", head_padding="same")
    tiny_wavlm().save_pretrained(tmp_path / "wavlm")
    work = tmp_path / "run"
    args = ["train", "--workdir", str(work), f"data.root={data}", "data.min_quality=0",
            f"mae.path={tmp_path / 'mae.pt'}", "train.cpu=true", "train.batch_size=3", "train.num_workers=0",
            "train.log_every=1", "train.save_every=1", "train.sample_every=0", "train.warmup=1", *TINY,
            "slm.enabled=true", f"slm.vocoder={tmp_path / 'vocos_tiny.pt'}", f"slm.wavlm={tmp_path / 'wavlm'}",
            "slm.trim_frames=4", "slm.disc_warmup=1", "slm.crops_per_cond=2", "slm.alpha=1.0", "slm.fm_weight=0.1"]
    cfg = load_config("configs/tts.yaml", [a for a in args[3:] if "=" in a])
    ds, _ = build_loader(cfg)
    assert ds.with_audio and "audio" in ds[0]
    main(args + ["train.steps=2"])
    ck = torch.load(work / "last.pt", weights_only=False)
    assert ck["slm"]["steps"] == 2 and "disc" in ck["slm"] and "opt" in ck["slm"]
    main(args + ["train.steps=3"])  # resumes the discriminator and its optimiser
    assert torch.load(work / "last.pt", weights_only=False)["slm"]["steps"] == 3
    # disabled (the default): no adversary state in the checkpoint
    main(["train", "--workdir", str(tmp_path / "plain")] + args[3:args.index("slm.enabled=true")] + ["train.steps=1"])
    assert "slm" not in torch.load(tmp_path / "plain" / "last.pt", weights_only=False)
