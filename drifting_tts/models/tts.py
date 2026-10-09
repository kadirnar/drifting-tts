"""Full drifting TTS model: text encoder (+ durations) and the one-step DriftDiT generator."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..alignment import sequence_mask
from ..text import SYMBOLS
from .generator import DriftDiT
from .text_encoder import DurationPredictor, TextEncoder, durations_to_alignment, expand, frames_to_alignment


class DriftingTTS(nn.Module):
    def __init__(self, cfg, num_speakers: int, n_mels: int = 100):
        super().__init__()
        t, g = cfg.text, cfg.gen
        self.encoder = TextEncoder(len(SYMBOLS), n_mels=n_mels, d=t.d, heads=t.heads, layers=t.layers, ffn=t.ffn,
                                   dropout=t.dropout, num_speakers=num_speakers, spk_dim=t.spk_dim)
        self.generator = DriftDiT(n_mels=n_mels, cond_channels=n_mels + t.d, hidden=g.hidden, depth=g.depth,
                                  heads=g.heads, patch=g.patch, mlp_ratio=g.mlp_ratio, n_registers=g.n_registers,
                                  num_speakers=num_speakers, noise_classes=g.noise_classes,
                                  noise_coords=g.noise_coords, num_steps=int(g.get("num_steps", 1)))
        # predict a residual over the aligned prior mean: a non-constant, well-conditioned output at init
        self.residual_prior = bool(g.get("residual_prior", True))
        self.n_mels = n_mels
        # FastPitch-style pitch conditioning: token-level normalised log-F0 added to the text features
        self.pitch_enabled = bool(cfg.get("pitch", {}).get("enabled", False))
        if self.pitch_enabled:
            self.pitch_predictor = DurationPredictor(t.d + t.spk_dim, dropout=t.dropout)
            self.pitch_emb = nn.Conv1d(1, t.d, 3, padding=1)
            self.register_buffer("lf0_stats", torch.tensor([5.0, 0.35]))  # log-F0 mean / std, set from the data

    def pitch_condition(self, h: Tensor, x_mask: Tensor, spk: Tensor, pitch: Tensor | None = None,
                        pitch_shift: float = 0.0) -> tuple[Tensor, Tensor]:
        """Add a pitch embedding to the token features.

        Uses the given (ground-truth) token pitch during training, the predicted one otherwise.
        ``pitch_shift`` is in semitones. Returns the conditioned features and the predicted pitch.
        """
        s = self.encoder.spk(spk)[:, :, None].expand(-1, -1, h.shape[-1])
        pred = self.pitch_predictor(torch.cat([h.detach(), s], 1), x_mask)
        p = pred if pitch is None else pitch
        if pitch_shift:  # semitones -> normalised log-F0 units
            p = p + pitch_shift / 12 * 0.6931471805599453 / self.lf0_stats[1]
        return h + self.pitch_emb(p) * x_mask, pred

    def generate(self, z: Tensor, cond: Tensor, spk: Tensor, cfg_scale: Tensor, mask: Tensor | None = None,
                 noise_labels: Tensor | None = None, attn_window: int | None = None, step: int = 0) -> Tensor:
        """One generator evaluation on an aligned condition (see :meth:`frame_condition`)."""
        x = self.generator(z, cond, spk, cfg_scale, mask=mask, noise_labels=noise_labels, attn_window=attn_window,
                           step=step)
        return x + cond[:, : self.n_mels] if self.residual_prior else x

    def rollout(self, z: Tensor, cond: Tensor, spk: Tensor, cfg_scale: Tensor, steps: int,
                generator: torch.Generator | None = None, **kw) -> Tensor:
        """Multi-step drifting: ``x_0 = z``, ``x_{k+1} = G(x_k, k)`` for ``k < steps`` (``steps = 1``: one-step).

        Unless given, one style code per trajectory is drawn from ``generator`` (the RNG of ``z``), so a seeded
        generator fully determines the output."""
        if kw.get("noise_labels") is None:
            net = self.generator
            kw["noise_labels"] = torch.randint(0, net.noise_classes, (spk.shape[0], max(1, net.noise_coords)),
                                               device=spk.device, generator=generator)
        x = z
        for k in range(steps):
            x = self.generate(x, cond, spk, cfg_scale, step=k, **kw)
        return x

    @staticmethod
    def frame_condition(h: Tensor, mu: Tensor, attn: Tensor) -> Tensor:
        """Aligned generator condition ``[B, n_mels + d, T]``: prior mean and text features."""
        return torch.cat([expand(mu, attn), expand(h, attn)], 1)

    @torch.no_grad()
    def synthesize(self, text: Tensor, text_len: Tensor, spk: Tensor, cfg_scale: float = 1.0,
                   temperature: float = 1.0, length_scale: float = 1.0, noise_labels: Tensor | None = None,
                   generator: torch.Generator | None = None, attn_window: int | None = None,
                   pitch_shift: float = 0.0, steps: int | None = None, durations: Tensor | None = None,
                   pitch: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Generate mels with ``steps`` generator evaluations (default: the trained number, 1 = one-step).

        ``z`` and the style codes both come from ``generator``. Returns normalised mels and lengths.
        Prosody overrides (:mod:`drifting_tts.prosody`): ``durations`` (frames per token ``[B, N]``, used as they are,
        without ``length_scale``) and ``pitch`` (normalised token log-F0 ``[B, 1, N]``) replace the predicted ones."""
        h, mu, logw, x_mask = self.encoder(text, text_len, spk)
        if self.pitch_enabled:
            h, _ = self.pitch_condition(h, x_mask, spk, pitch, pitch_shift=pitch_shift)
        if durations is None:
            attn, y_len = durations_to_alignment(logw, x_mask, length_scale)
        else:
            attn, y_len = frames_to_alignment(durations, x_mask)
        cond = self.frame_condition(h, mu, attn)
        z = torch.randn(cond.shape[0], mu.shape[1], cond.shape[-1], device=cond.device, generator=generator)
        mask = sequence_mask(y_len, cond.shape[-1])
        alpha = torch.full((text.shape[0],), float(cfg_scale), device=text.device)
        steps = self.generator.num_steps if steps is None else steps
        mel = self.rollout(z * temperature, cond, spk, alpha, steps, generator=generator, mask=mask,
                           noise_labels=noise_labels, attn_window=attn_window)
        return mel * mask[:, None], y_len
