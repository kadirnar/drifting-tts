"""Text encoder with prior projection and duration predictor (Grad-TTS / Matcha-TTS style).

It provides the frame-aligned conditioning of the one-step generator: monotonic alignment search
between the prior mean ``mu`` and the target mel gives hard durations during training, and the
duration predictor replaces them at inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..alignment import gaussian_log_prior, generate_path, maximum_path, sequence_mask


def rope(x: Tensor, offset: int = 0) -> Tensor:
    """Rotary position embedding on ``[B, H, N, Dh]``."""
    N, D = x.shape[-2], x.shape[-1]
    half = D // 2
    freqs = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    t = torch.arange(offset, offset + N, device=x.device, dtype=torch.float32)
    ang = torch.outer(t, freqs)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ConvFFN(nn.Module):
    def __init__(self, d: int, hidden: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, d, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # x: [B, N, d], mask [B, N, 1]
        h = self.conv1((x * mask).transpose(1, 2))
        h = self.drop(F.gelu(h))
        return (self.conv2(h * mask.transpose(1, 2))).transpose(1, 2) * mask


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = ConvFFN(d, ffn, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, N, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        attn_mask = mask[:, None, None, :, 0].bool()  # [B, 1, 1, N] keys; a float mask would be an additive bias
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.out(h.transpose(1, 2).reshape(B, N, d)))
        x = x + self.drop(self.ffn(self.norm2(x), mask))
        return x * mask


class DurationPredictor(nn.Module):
    def __init__(self, d: int, hidden: int = 256, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d, hidden, kernel, padding=kernel // 2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.norm2 = nn.LayerNorm(hidden)
        self.proj = nn.Conv1d(hidden, 1, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # x [B, d, N], mask [B, 1, N]
        x = self.conv1(x * mask)
        x = self.drop(self.norm1(F.relu(x).transpose(1, 2)).transpose(1, 2))
        x = self.conv2(x * mask)
        x = self.drop(self.norm2(F.relu(x).transpose(1, 2)).transpose(1, 2))
        return self.proj(x * mask) * mask  # log-durations [B, 1, N]


class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int, n_mels: int = 100, d: int = 192, heads: int = 2, layers: int = 6,
                 ffn: int = 768, dropout: float = 0.1, num_speakers: int = 1, spk_dim: int = 64):
        super().__init__()
        self.d, self.n_mels = d, n_mels
        self.emb = nn.Embedding(n_vocab, d)
        nn.init.normal_(self.emb.weight, 0.0, d**-0.5)
        self.spk = nn.Embedding(num_speakers, spk_dim)
        self.spk_proj = nn.Linear(spk_dim, d)
        self.prenet = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2) for _ in range(3)])
        self.prenet_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ffn, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.proj_mu = nn.Conv1d(d, n_mels, 1)
        self.duration = DurationPredictor(d + spk_dim, dropout=dropout)

    def forward(self, text: Tensor, text_len: Tensor, spk: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns hidden ``h`` [B, d, N], prior mean ``mu`` [B, n_mels, N], log-durations [B, 1, N], mask [B, 1, N]."""
        mask = sequence_mask(text_len, text.shape[1])[:, :, None].float()  # [B, N, 1]
        s = self.spk(spk)
        x = self.emb(text) * math.sqrt(self.d) + self.spk_proj(s)[:, None]
        for conv, norm in zip(self.prenet, self.prenet_norms):  # residual conv prenet
            x = x + norm(F.relu(conv((x * mask).transpose(1, 2)).transpose(1, 2)))
        x = x * mask
        for layer in self.layers:
            x = layer(x, mask)
        h = (self.norm(x) * mask).transpose(1, 2)
        m = mask.transpose(1, 2)
        mu = self.proj_mu(h) * m
        dur_in = torch.cat([h.detach(), s[:, :, None].expand(-1, -1, h.shape[-1])], 1)
        logw = self.duration(dur_in, m)
        return h, mu, logw, m


def align(mu: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor) -> tuple[Tensor, Tensor]:
    """MAS between the prior ``mu`` [B, C, N] and the target mel ``y`` [B, C, T].

    Returns the hard alignment ``[B, N, T]`` and the target log-durations ``[B, 1, N]``.
    """
    attn_mask = x_mask.transpose(1, 2) * y_mask  # [B, N, T]
    with torch.no_grad():
        attn = maximum_path(gaussian_log_prior(mu.float(), y.float()), attn_mask)
    logw_target = torch.log(1e-8 + attn.sum(-1, keepdim=True)).transpose(1, 2) * x_mask
    return attn, logw_target


def expand(x: Tensor, attn: Tensor) -> Tensor:
    """Token features ``[B, C, N]`` -> frame features ``[B, C, T]`` with a hard alignment ``[B, N, T]``."""
    return torch.bmm(x, attn)


def prior_loss(mu_y: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
    """Gaussian NLL of the target under the aligned prior (per element)."""
    nll = 0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask
    return nll.sum() / (y_mask.sum() * y.shape[1])


def duration_loss(logw: Tensor, logw_target: Tensor, x_mask: Tensor) -> Tensor:
    return ((logw - logw_target) ** 2 * x_mask).sum() / x_mask.sum()


def durations_to_alignment(logw: Tensor, x_mask: Tensor, length_scale: float = 1.0) -> tuple[Tensor, Tensor]:
    """Predicted log-durations -> hard alignment ``[B, N, T]`` and mel lengths ``[B]``."""
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w).clamp_min(0)[:, 0] * x_mask[:, 0]
    y_len = w_ceil.sum(1).clamp_min(1).long()
    y_mask = sequence_mask(y_len)[:, None].to(x_mask.dtype)
    attn_mask = x_mask.transpose(1, 2) * y_mask
    return generate_path(w_ceil, attn_mask), y_len


def token_pitch(f0: Tensor, attn: Tensor, lf0_mean: float | Tensor, lf0_std: float | Tensor) -> Tensor:
    """Average normalised log-F0 of the voiced frames of every token (FastPitch); 0 for unvoiced tokens.

    Args:
        f0: ``[B, T]`` F0 in Hz (0 = unvoiced). attn: ``[B, N, T]`` hard alignment.
        lf0_mean / lf0_std: log-F0 statistics (floats or 0-dim tensors on the device of ``f0``).
    Returns:
        ``[B, 1, N]`` token pitch targets.
    """
    voiced = (f0 > 0).float()
    lf0 = torch.where(f0 > 0, (torch.log(f0.clamp_min(1.0)) - lf0_mean) / lf0_std, torch.zeros_like(f0))
    num = torch.bmm(attn, (lf0 * voiced)[:, :, None])[:, :, 0]
    den = torch.bmm(attn, voiced[:, :, None])[:, :, 0]
    return torch.where(den > 0, num / den.clamp_min(1.0), torch.zeros_like(num))[:, None]
