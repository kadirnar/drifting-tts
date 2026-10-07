"""DriftDiT: the one-step generator, a 1-D port of the official ``DitGen`` / ``LightningDiT``.

``mel = G(z, cond)`` maps Gaussian noise ``z`` (same shape as the mel) plus a frame-aligned text
condition to a mel-spectrogram in a single forward pass. Design choices follow the official
generator: adaLN-zero blocks with RMSNorm, QK-norm, RoPE and SwiGLU; ``n_registers`` in-context
condition tokens; random *style* embeddings (``noise_coords`` categorical codes with
``noise_classes`` values each, an extra source of randomness besides ``z``); and an embedding of
the CFG scale ``alpha`` so that guidance learned at training time is selectable at inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .text_encoder import rope


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x.to(dtype)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        hidden = (int(2 / 3 * dim * mlp_ratio) + 31) // 32 * 32
        self.mlp = SwiGLU(dim, hidden)
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: Tensor, c: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(F.silu(c)).chunk(6, dim=-1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, sc1), attn_mask)
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, sc2))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ada = nn.Linear(dim, 2 * dim)
        self.linear = nn.Linear(dim, out)
        for m in (self.ada, self.linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.ada(F.silu(c)).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class ScalarEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar (the CFG scale) followed by an MLP (``TimestepEmbedder``)."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], -1))


class DriftDiT(nn.Module):
    def __init__(
        self,
        n_mels: int = 100,
        cond_channels: int = 292,
        hidden: int = 384,
        depth: int = 8,
        heads: int = 6,
        patch: int = 2,
        mlp_ratio: float = 4.0,
        n_registers: int = 16,
        num_speakers: int = 1,
        noise_classes: int = 64,
        noise_coords: int = 32,
        num_steps: int = 1,
    ):
        super().__init__()
        self.n_mels, self.patch, self.n_registers = n_mels, patch, n_registers
        self.noise_classes, self.noise_coords = noise_classes, noise_coords
        self.in_proj = nn.Linear((n_mels + cond_channels) * patch, hidden)
        self.spk = nn.Embedding(num_speakers, hidden)
        nn.init.normal_(self.spk.weight, std=0.02)
        self.noise_embeds = nn.ModuleList([nn.Embedding(noise_classes, hidden) for _ in range(noise_coords)])
        for e in self.noise_embeds:
            nn.init.normal_(e.weight, std=0.02)
        self.cfg_embed = ScalarEmbedder(hidden)
        self.cfg_norm = RMSNorm(hidden)
        # multi-step drifting (DriftTTS): a step embedding tells the generator which state it refines
        self.num_steps = num_steps
        if num_steps > 1:
            self.step_embed = nn.Embedding(num_steps, hidden)
            nn.init.zeros_(self.step_embed.weight)
        if n_registers > 0:
            self.reg_proj = nn.Linear(hidden, hidden)
            self.registers = nn.Parameter(torch.randn(1, n_registers, hidden) * 0.02)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(hidden, n_mels * patch)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

    def condition(self, spk: Tensor, cfg_scale: Tensor, noise_labels: Tensor | None = None, step: int = 0) -> Tensor:
        """Global conditioning vector: speaker + style codes + 0.02 * RMSNorm(cfg embedding)."""
        c = self.spk(spk)
        if self.noise_coords > 0:
            if noise_labels is None:
                shape = (spk.shape[0], self.noise_coords)
                noise_labels = torch.randint(0, self.noise_classes, shape, device=spk.device)
            c = c + sum(e(noise_labels[:, i]) for i, e in enumerate(self.noise_embeds))
        if self.num_steps > 1:
            c = c + self.step_embed(torch.full_like(spk, step))
        return c + 0.02 * self.cfg_norm(self.cfg_embed(cfg_scale))

    def window_mask(self, n_tokens: int, window: int, device) -> Tensor:
        """``[1, 1, N, N]`` boolean band mask; register tokens attend / are attended globally."""
        pos = torch.arange(n_tokens, device=device)
        frame = pos >= self.n_registers
        band = (pos[:, None] - pos[None, :]).abs() <= window
        return (band | ~frame[:, None] | ~frame[None, :])[None, None]

    def forward(self, z: Tensor, cond: Tensor, spk: Tensor, cfg_scale: Tensor, mask: Tensor | None = None,
                noise_labels: Tensor | None = None, attn_window: int | None = None, step: int = 0) -> Tensor:
        """
        Args:
            z: ``[B, n_mels, T]`` Gaussian noise.
            cond: ``[B, cond_channels, T]`` frame-aligned condition (prior mean and text features).
            spk: ``[B]`` speaker ids. cfg_scale: ``[B]`` guidance scale ``alpha`` (1 = none).
            mask: optional ``[B, T]`` valid-frame mask (for padded batches).
            attn_window: if set, frame tokens attend only to frame tokens within +-``attn_window``
                tokens (registers stay global). Matches the training receptive field on long inputs.
            step: refinement step ``k`` (multi-step drifting): ``z`` is then the state ``x_k`` (noise for k = 0).
        Returns:
            ``[B, n_mels, T]`` mel-spectrogram (normalised).
        """
        B, _, T = z.shape
        p = self.patch
        pad = (-T) % p
        x = torch.cat([z, cond], 1)
        if pad:
            x = F.pad(x, (0, pad))
        n = x.shape[-1] // p
        x = self.in_proj(x.view(B, x.shape[1], n, p).permute(0, 2, 1, 3).reshape(B, n, -1))
        c = self.condition(spk, cfg_scale, noise_labels, step)

        key_mask = None
        if mask is not None:
            m = F.pad(mask, (0, pad)).view(B, n, p).any(-1)
            key_mask = m
        if self.n_registers > 0:
            regs = self.reg_proj(c)[:, None] + self.registers
            x = torch.cat([regs, x], 1)
            if key_mask is not None:
                key_mask = torch.cat([key_mask.new_ones(B, self.n_registers), key_mask], 1)
        attn_mask = None if key_mask is None else key_mask[:, None, None, :]
        if attn_window is not None and n > attn_window:
            band = self.window_mask(x.shape[1], attn_window, x.device)
            attn_mask = band if attn_mask is None else attn_mask & band
        for block in self.blocks:
            x = block(x, c, attn_mask)
        x = self.final(x, c)[:, self.n_registers:]
        x = x.view(B, n, self.n_mels, p).permute(0, 2, 1, 3).reshape(B, self.n_mels, n * p)
        return x[..., :T]
