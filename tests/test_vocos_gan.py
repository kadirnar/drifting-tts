"""VocosGAN's opt-in extensions on CPU: BigVGAN discriminators, cosine learning rate, ``train.init_from``."""

import pytest
import torch

from drifting_tts.config import Config
from drifting_tts.finetune_vocoder import VocosGAN
from drifting_tts.utils import save_checkpoint
from tests.test_drift_vocoder import TINY_CQT
from tests.test_train import TINY_VOCOS

FRAMES = 16


def _cfg(tmp_path, train: dict | None = None, **vocoder) -> Config:
    (tmp_path / "tiny_vocos.yaml").write_text(TINY_VOCOS)
    return Config({"vocoder": {"init": str(tmp_path / "tiny_vocos.yaml"), **vocoder},
                   "train": {"lr": 1e-3, "grad_clip": 10.0, "mel_loss_coeff": 45.0, "mrd_loss_coeff": 0.1,
                             "disc_warmup_steps": 1, "steps": 4, "segment_frames": FRAMES, **(train or {})}})


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 100, FRAMES, generator=g) - 5, 0.1 * torch.randn(2, FRAMES * 256, generator=g)


def test_default_recipe_keeps_its_layout(tmp_path):
    gan = VocosGAN(_cfg(tmp_path), "cpu", mel="bigvgan")
    assert gan.bigvgan == {} and gan.mpd is not None and gan.mrd is not None
    mel, audio = _batch()
    assert set(gan.step(mel, audio, 0)) == {"disc"}  # discriminator warm-up
    m = gan.step(mel, audio, 1)
    assert {"disc", "gen", "mel", "fm"} <= set(m) and "lr" not in m
    assert gan.opt_g.param_groups[0]["lr"] == 1e-3  # constant
    assert set(gan.state_dict()) == {"vocos", "mpd", "mrd", "opt_g", "opt_d"}  # what earlier runs saved


def test_bigvgan_discriminators_cosine_and_resume(tmp_path):
    cfg = _cfg(tmp_path, {"lr_schedule": "cosine", "lr_min_ratio": 0.1}, discriminators=["mrd"],
               bigvgan_discriminators=["mpd", "cqtd"], bigvgan_pretrained=False, bigvgan_hparams=TINY_CQT)
    torch.manual_seed(0)
    gan = VocosGAN(cfg, "cpu", mel="bigvgan")
    assert gan.mpd is None and list(gan.bigvgan) == ["cqtd", "mpd"] and len(gan.scheds) == 3
    mel, audio = _batch()
    m0 = gan.step(mel, audio, 0)
    assert {"disc", "disc_bigvgan_mpd", "disc_bigvgan_cqtd"} <= set(m0) and "gen" not in m0
    m1 = gan.step(mel, audio, 1)
    assert {"adv_bigvgan_mpd", "fm_bigvgan_cqtd", "gen", "mel"} <= set(m1)
    assert all(torch.isfinite(torch.tensor(v)) for v in m1.values())
    assert 1e-4 < m1["lr"] < 1e-3  # cosine from 1e-3 to 1e-4 over 4 steps
    state = gan.state_dict()
    assert {"bigvgan_mpd", "bigvgan_cqtd", "opt_db", "scheds", "mrd"} <= set(state) and "mpd" not in state
    other = VocosGAN(cfg, "cpu", mel="bigvgan")
    other.load_state_dict(state)
    assert other.scheds[0].get_last_lr() == gan.scheds[0].get_last_lr()
    for a, b in zip(other.bigvgan["cqtd"].parameters(), gan.bigvgan["cqtd"].parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("discriminators", [["mpd", "mrd"], ["mrd"]])
def test_init_from_a_finished_run(tmp_path, discriminators):
    torch.manual_seed(0)
    old = VocosGAN(_cfg(tmp_path), "cpu", mel="bigvgan")
    mel, audio = _batch()
    old.step(mel, audio, 1)
    save_checkpoint(tmp_path / "last.pt", step=40000, epoch=3, **old.state_dict())
    cfg = _cfg(tmp_path, {"lr": 5e-4}, discriminators=discriminators, bigvgan_discriminators=["mpd"],
               bigvgan_pretrained=False, bigvgan_hparams=TINY_CQT)
    new = VocosGAN(cfg, "cpu", mel="bigvgan")
    new.init_from(str(tmp_path / "last.pt"))
    for a, b in zip(new.vocos.parameters(), old.vocos.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(new.mrd.parameters(), old.mrd.parameters()):
        assert torch.equal(a, b)
    assert new.opt_g.param_groups[0]["lr"] == 5e-4 and len(new.opt_g.state) > 0  # moments loaded, our LR kept
    assert (len(new.opt_d.state) > 0) == (discriminators == ["mpd", "mrd"])  # only for the same discriminators
    assert torch.isfinite(torch.tensor(new.step(mel, audio, 1)["gen"]))


def test_bigvgan_recipe_losses_and_warmup(tmp_path):
    """BigVGAN's balance: summed LSGAN terms, feature matching x 2, multi-scale mel; separate D rate, G warm-up."""
    cfg = _cfg(tmp_path, {"mel_loss": "multiscale", "mel_loss_coeff": 15.0, "bigvgan_reduce": "sum",
                          "lr_disc": 2e-3, "warmup_steps": 2, "disc_warmup_steps": 0},
               discriminators=[], bigvgan_discriminators=["mpd", "cqtd"], bigvgan_pretrained=False,
               bigvgan_hparams=TINY_CQT)
    torch.manual_seed(0)
    gan = VocosGAN(cfg, "cpu", mel="bigvgan")
    assert gan.opt_d is None and gan.opt_g.param_groups[0]["lr"] == pytest.approx(5e-4)  # warm-up: 1 / 2 of 1e-3
    assert gan.opt_db.param_groups[0]["lr"] == pytest.approx(2e-3)  # no warm-up for the discriminators
    mel, audio = _batch()
    m = gan.step(mel, audio, 0)
    assert {"gen", "mel", "adv_bigvgan_mpd", "grad_gen", "lr"} <= set(m) and "disc" not in m
    assert m["mel"] > 1.0  # the sum of 7 scales
    assert gan.opt_g.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert "scheds" in gan.state_dict()


def test_unknown_discriminator(tmp_path):
    with pytest.raises(ValueError):
        VocosGAN(_cfg(tmp_path, discriminators=["msd"]), "cpu")
    with pytest.raises(ValueError):
        VocosGAN(_cfg(tmp_path, discriminators=[]), "cpu")


def test_phase_derivative_loss_ignores_shifts_and_sees_jitter():
    from drifting_tts.finetune_vocoder import phase_derivative_loss

    t = torch.arange(4 * 4096) / 24000
    tone = lambda f: sum(torch.sin(2 * torch.pi * k * f * t) / k for k in (1, 2, 3))[None]  # noqa: E731
    y = tone(120.0)
    assert phase_derivative_loss(y, y) < 1e-6
    shifted = torch.roll(y, 37, -1)  # a constant delay: the phase advance per frame does not change
    jitter = tone(120.0 * (1 + 0.03 * torch.sin(2 * torch.pi * 9 * t)))  # +-3% vibrato at 9 Hz
    assert phase_derivative_loss(shifted, y) < 0.1 * phase_derivative_loss(jitter, y)
    y_hat = y.clone().requires_grad_()
    phase_derivative_loss(y_hat + 0.01 * torch.randn_like(y), y).backward()
    assert torch.isfinite(y_hat.grad).all()


def test_iaf_term_on_recorded_batches_only(tmp_path):
    gan = VocosGAN(_cfg(tmp_path, {"iaf_loss_coeff": 5.0}), "cpu", mel="bigvgan")
    mel, audio = _batch()
    gan.batch_is_gta = True
    assert "iaf" not in gan.step(mel, audio, 1)
    gan.batch_is_gta = False
    assert "iaf" in gan.step(mel, audio, 1)
