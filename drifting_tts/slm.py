"""SLM adversary for fine-tuning the acoustic model (opt-in ``slm:`` config block, ``slm.enabled``).

StyleTTS 2 (arXiv 2306.07691) trains its generator against a discriminator on the hidden states of a frozen WavLM
and loses 0.32 CMOS without it. Here, each training step:

1. takes ``crops_per_cond`` generated mel crops per condition: a random subset of the ``G`` drift samples
   (``alpha: null``) or extra samples drawn at a fixed CFG scale ``alpha`` with noise ``temperature``;
2. denormalises them and vocodes them with the **frozen** Vocos fine-tune through its differentiable ISTFT head
   (:meth:`drifting_tts.vocoder.Vocoder.differentiable`), drops ``trim_frames`` at both crop ends (crop-edge
   artefacts of the vocoder reach ~12-16 frames), resamples 24 -> 16 kHz and runs the **frozen** WavLM;
3. scores every WavLM frame with a small trainable discriminator over the stacked hidden states (StyleTTS 2's
   ``WavLMDiscriminator``: 1-D convolutions over ``[layers x dim]`` channels), LSGAN. The real inputs are the
   recorded audio of the same crops (``real: audio``, BigVGAN framing: ``F`` frames <-> ``F * 256`` samples from
   ``start * 256``) or the real mel crops through the same vocoder (``real: vocoded``);
4. computes the generator terms through the discriminator with its parameters frozen: ``weight`` x LSGAN, plus
   ``fm_weight`` x the L1 distance between the WavLM hidden states of each generated segment and its real one
   (the crops are generated under the ground-truth alignment and pitch, so they line up), then updates the
   discriminator on detached inputs with its own AdamW (:meth:`SLMAdversary.step`). For the first ``disc_warmup``
   steps only the discriminator trains.

Memory: the gradient of the generator terms w.r.t. the generated mels is computed inside :meth:`SLMAdversary.step`,
``chunk`` crops at a time, and handed back as a surrogate loss with that gradient; WavLM's activations (~0.12 GB per
crop, mostly its convolutional front end) are freed before the caller's drift graph is built.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .audio import HOP_LENGTH, SAMPLE_RATE
from .data import MelStats

SLM_RATE = 16_000
LRELU_SLOPE = 0.1
REAL_SOURCES = ("audio", "vocoded")


def slm_enabled(cfg) -> bool:
    return bool((cfg.get("slm") or {}).get("enabled", False))


def needs_audio(cfg) -> bool:
    """Whether the dataset must load the recordings (``prepare --save-audio``)."""
    return slm_enabled(cfg) and cfg.slm.get("real", "audio") == "audio"


class WavLMDiscriminator(nn.Module):
    """StyleTTS 2's SLM discriminator: stacked hidden states ``[B, layers * dim, T]`` -> frame scores ``[B, T]``."""

    def __init__(self, dim: int = 768, layers: int = 13, channels: int = 64):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.pre = wn(nn.Conv1d(dim * layers, channels, 1))
        self.convs = nn.ModuleList([wn(nn.Conv1d(channels, 2 * channels, 5, padding=2)),
                                    wn(nn.Conv1d(2 * channels, 4 * channels, 5, padding=2)),
                                    wn(nn.Conv1d(4 * channels, 4 * channels, 5, padding=2))])
        self.post = wn(nn.Conv1d(4 * channels, 1, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
        return self.post(x).flatten(1)


def stack_layers(hidden: tuple[Tensor, ...]) -> Tensor:
    """WavLM hidden states, ``layers`` x ``[B, T, dim]`` -> ``[B, layers * dim, T]`` (layer-major channels)."""
    return torch.stack(hidden, 1).transpose(-1, -2).flatten(1, 2)


def audio_segments(audio: Tensor, starts: Tensor, frames: int) -> Tensor:
    """Recorded samples of mel crops ``[start, start + frames)`` (BigVGAN framing): ``[B, frames * 256]``."""
    idx = starts[:, None] * HOP_LENGTH + torch.arange(frames * HOP_LENGTH, device=audio.device)[None]
    return torch.gather(audio, 1, idx)


class SLMAdversary:
    """Frozen vocoder + frozen WavLM + trainable :class:`WavLMDiscriminator` with its own optimiser.

    ``cfg``: the ``slm`` config block; ``vocoder``: a :class:`drifting_tts.vocoder.Vocoder` (Vocos on BigVGAN mels);
    ``wavlm``: a ``transformers`` WavLM (or any model returning ``hidden_states``); ``stats``: the mel normalisation.
    Not an ``nn.Module``: only the discriminator, its optimiser and the step count are checkpointed."""

    def __init__(self, cfg, vocoder, wavlm: nn.Module, stats: MelStats, device):
        import torchaudio

        self.cfg, self.vocoder, self.stats, self.device = cfg, vocoder, stats, torch.device(device)
        self.real = cfg.get("real", "audio")
        if self.real not in REAL_SOURCES:
            raise ValueError(f"slm.real must be one of {REAL_SOURCES}, got {self.real!r}")
        self.weight, self.fm_weight = float(cfg.get("weight", 1.0)), float(cfg.get("fm_weight", 0.0))
        self.crops_per_cond = int(cfg.get("crops_per_cond", 1))
        self.alpha = cfg.get("alpha")  # None: a subset of the drift samples; a number: extra samples at that scale
        self.temperature = float(cfg.get("temperature", 1.0))
        self.trim = int(cfg.get("trim_frames", 16)) * HOP_LENGTH
        self.disc_warmup = int(cfg.get("disc_warmup", 0))
        self.chunk = int(cfg.get("chunk", 8))
        self.grad_clip = float(cfg.get("grad_clip") or 0.0)  # max norm of the weighted gradient w.r.t. the crops
        self.separate_terms = False  # also return each term's own surrogate (gradient-norm probes; costs a backward)
        self.amp = cfg.get("dtype", "bf16") == "bf16" and self.device.type == "cuda"
        for m in (vocoder.model, wavlm):
            m.eval().requires_grad_(False)
        self.wavlm = wavlm.to(self.device)
        self.resample = torchaudio.transforms.Resample(SAMPLE_RATE, SLM_RATE).to(self.device)
        c = wavlm.config
        self.disc = WavLMDiscriminator(c.hidden_size, c.num_hidden_layers + 1,
                                       int(cfg.get("disc_channels", 64))).to(self.device)
        self.opt = torch.optim.AdamW(self.disc.parameters(), lr=float(cfg.get("disc_lr", 1e-4)),
                                     betas=tuple(cfg.get("disc_betas", (0.0, 0.99))),
                                     weight_decay=float(cfg.get("disc_weight_decay", 1e-4)))
        self.steps = 0

    @property
    def extra_per_cond(self) -> int:
        """Extra generator samples per condition that the training step must draw (0: reuse the drift samples)."""
        return 0 if self.alpha is None else self.crops_per_cond

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

    def waveform(self, mel: Tensor) -> Tensor:
        """Normalised mels ``[B, C, F]`` -> 24 kHz waveform ``[B, F * 256]`` (fp32, differentiable)."""
        with torch.autocast(self.device.type, enabled=False):
            return self.vocoder.differentiable(self.stats.denormalize(mel.float()))

    def hidden_states(self, wav: Tensor) -> tuple[Tensor, ...]:
        """24 kHz waveform -> WavLM hidden states (all layers, ``[B, T, dim]`` each) of its trimmed middle."""
        if self.trim:
            wav = wav[:, self.trim: wav.shape[-1] - self.trim]
        wav = self.resample(wav.float())
        with self._autocast():
            return self.wavlm(input_values=wav, output_hidden_states=True).hidden_states

    def step(self, fake: Tensor, real_mel: Tensor, real_audio: Tensor | None = None) -> tuple[Tensor, dict, dict]:
        """Generator terms and one discriminator update.

        ``fake``: generated normalised mel crops ``[B * c, C, F]`` (with their graph), grouped by condition;
        ``real_mel`` ``[B, C, F]`` and ``real_audio`` ``[B, F * 256]``: the real crops of the same conditions.
        Returns the weighted generator loss as a surrogate whose value is the loss and whose gradient w.r.t. ``fake``
        is the loss's, its norm capped at ``grad_clip`` (zero during ``disc_warmup``), detached metrics, and the
        unweighted terms (``adv``, ``fm``; surrogates with their own, uncapped gradients when ``separate_terms``)."""
        B, n = real_mel.shape[0], fake.shape[0]
        c = n // B
        train_g = (self.steps >= self.disc_warmup and (self.weight > 0 or self.fm_weight > 0)
                   and torch.is_grad_enabled())  # a no-grad caller gets a discriminator-only step
        with torch.no_grad():
            real_wav = real_audio.float() if self.real == "audio" else self.waveform(real_mel)
            h_real = self.hidden_states(real_wav)
        leaf = fake.detach().requires_grad_(train_g)
        x_fake, adv, fm = [], fake.new_zeros(()), fake.new_zeros(())
        grads = {"total": torch.zeros_like(leaf), "adv": torch.zeros_like(leaf), "fm": torch.zeros_like(leaf)}
        self.disc.requires_grad_(False)  # the generator terms train only the generated path
        try:
            for i in range(0, n, self.chunk):
                j = min(n, i + self.chunk)
                with torch.set_grad_enabled(train_g):
                    h = self.hidden_states(self.waveform(leaf[i:j]))
                    x = stack_layers(h)
                    x_fake.append(x.detach())
                    if not train_g:
                        continue
                    with self._autocast():
                        g = self.disc(x).float()
                    a = (1 - g).pow(2).mean() * (j - i) / n
                    cond = torch.arange(i, j, device=leaf.device) // c  # the real segment of each crop
                    f = torch.stack([(hf.float() - hr[cond].float()).abs().mean() for hf, hr in zip(h, h_real)])
                    f = f.mean() * (j - i) / n
                terms = {"total": self.weight * a + self.fm_weight * f}
                if self.separate_terms:
                    terms.update(adv=a, fm=f)
                for k, t in terms.items():
                    grads[k] += torch.autograd.grad(t, leaf, retain_graph=k != list(terms)[-1])[0]
                adv, fm = adv + a.detach(), fm + f.detach()
        finally:
            self.disc.requires_grad_(True)

        # discriminator: LSGAN on detached inputs, its own optimiser (also when called under no_grad)
        with torch.enable_grad(), self._autocast():
            d_real, d_fake = self.disc(stack_layers(h_real)).float(), self.disc(torch.cat(x_fake)).float()
            l_disc = (1 - d_real).pow(2).mean() + d_fake.pow(2).mean()
        self.opt.zero_grad(set_to_none=True)
        l_disc.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.disc.parameters(), float(self.cfg.get("disc_grad_clip", 1e9)))
        self.opt.step()
        self.steps += 1
        metrics = {"slm_disc": l_disc.detach(), "slm_d_real": d_real.mean().detach(),
                   "slm_d_fake": d_fake.mean().detach(), "slm_grad_norm_disc": gnorm}
        if not train_g:
            return fake.new_zeros(()), metrics, {}
        gnorm_x = grads["total"].norm()
        metrics.update(slm_adv=adv, slm_fm=fm, slm_grad_norm_x=gnorm_x)
        if self.grad_clip:  # cap the push on the generated mels (the discriminator's sharpness varies a lot)
            grads["total"] = grads["total"] * (self.grad_clip / (gnorm_x + 1e-12)).clamp(max=1.0)
            metrics["slm_capped"] = (gnorm_x > self.grad_clip).float()  # its mean over a log window: the binding rate

        def surrogate(grad: Tensor, value: Tensor) -> Tensor:  # value ``value``, gradient ``grad`` w.r.t. ``fake``
            s = (fake * grad).sum()
            return s - s.detach() + value

        loss = surrogate(grads["total"], self.weight * adv + self.fm_weight * fm)
        if not self.separate_terms:
            return loss, metrics, {"adv": adv, "fm": fm}
        return loss, metrics, {"adv": surrogate(grads["adv"], adv), "fm": surrogate(grads["fm"], fm)}

    def state_dict(self) -> dict:
        return {"disc": self.disc.state_dict(), "opt": self.opt.state_dict(), "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        self.disc.load_state_dict(state["disc"])
        self.opt.load_state_dict(state["opt"])
        self.steps = int(state["steps"])


def load_wavlm(name: str) -> nn.Module:
    """A Hugging Face WavLM (``microsoft/wavlm-base-plus``) or a local directory saved with ``save_pretrained``."""
    from transformers import WavLMModel

    return WavLMModel.from_pretrained(name)


def build_slm(cfg, stats: MelStats, backend: str, device) -> SLMAdversary:
    """The adversary of ``cfg.slm`` for a model trained on ``backend`` mels (BigVGAN-style only)."""
    from .vocoder import load_vocoder

    if backend != "bigvgan":
        raise ValueError(f"slm: needs BigVGAN-style mels (the Vocos fine-tune's framing), the data has {backend!r}")
    sc = cfg.slm
    vocoder = load_vocoder(sc.get("vocoder", "vocos-ft"), str(device))
    return SLMAdversary(sc, vocoder, load_wavlm(sc.get("wavlm", "microsoft/wavlm-base-plus")), stats, device)
