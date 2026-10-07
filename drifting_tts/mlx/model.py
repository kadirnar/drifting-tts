"""The acoustic model in MLX: text encoder, duration and pitch predictors, and the one-step DriftDiT generator.

A port of ``drifting_tts.models`` for inference on one utterance at a time (no padding masks). Tensors are
channels-last (``[B, N, C]``) as MLX convolutions expect; parameter names follow the PyTorch modules, so a converted
state dict loads directly (see :mod:`drifting_tts.mlx.convert`).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def rope(x: mx.array) -> mx.array:
    """Rotary position embedding on ``[B, H, N, Dh]`` (halves layout, base 10000), as ``models.text_encoder.rope``."""
    n, d = x.shape[-2], x.shape[-1]
    half = d // 2
    freqs = 1.0 / (10000 ** (mx.arange(half, dtype=mx.float32) / half))
    ang = mx.arange(n, dtype=mx.float32)[:, None] * freqs[None]
    cos, sin = mx.cos(ang).astype(x.dtype), mx.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def attention(qkv: mx.array, heads: int, q_norm=None, k_norm=None) -> mx.array:
    """Self-attention from a fused ``[B, N, 3 * D]`` projection; returns ``[B, N, D]``."""
    b, n, d3 = qkv.shape
    d = d3 // 3
    qkv = qkv.reshape(b, n, 3, heads, d // heads).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if q_norm is not None:
        q, k = q_norm(q), k_norm(k)
    q, k = rope(q), rope(k)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=(d // heads) ** -0.5)
    return x.transpose(0, 2, 1, 3).reshape(b, n, d)


class ConvFFN(nn.Module):
    def __init__(self, d: int, hidden: int, kernel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(d, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, d, kernel, padding=kernel // 2)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2(nn.gelu(self.conv1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = ConvFFN(d, ffn)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.out(attention(self.qkv(self.norm1(x)), self.heads))
        return x + self.ffn(self.norm2(x))


class DurationPredictor(nn.Module):
    def __init__(self, d: int, hidden: int = 256, kernel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(d, hidden, kernel, padding=kernel // 2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.norm2 = nn.LayerNorm(hidden)
        self.proj = nn.Conv1d(hidden, 1, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm1(nn.relu(self.conv1(x)))
        x = self.norm2(nn.relu(self.conv2(x)))
        return self.proj(x)  # [B, N, 1]


class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int, n_mels: int, d: int, heads: int, layers: int, ffn: int, num_speakers: int,
                 spk_dim: int):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(n_vocab, d)
        self.spk = nn.Embedding(num_speakers, spk_dim)
        self.spk_proj = nn.Linear(spk_dim, d)
        self.prenet = [nn.Conv1d(d, d, 5, padding=2) for _ in range(3)]
        self.prenet_norms = [nn.LayerNorm(d) for _ in range(3)]
        self.layers = [EncoderLayer(d, heads, ffn) for _ in range(layers)]
        self.norm = nn.LayerNorm(d)
        self.proj_mu = nn.Conv1d(d, n_mels, 1)
        self.duration = DurationPredictor(d + spk_dim)

    def __call__(self, text: mx.array, spk: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Token ids ``[B, N]`` -> features ``h`` ``[B, N, d]``, prior mean ``[B, N, n_mels]``, log-durations
        ``[B, N, 1]``."""
        s = self.spk(spk)
        x = self.emb(text) * math.sqrt(self.d) + self.spk_proj(s)[:, None]
        for conv, norm in zip(self.prenet, self.prenet_norms):
            x = x + norm(nn.relu(conv(x)))
        for layer in self.layers:
            x = layer(x)
        h = self.norm(x)
        mu = self.proj_mu(h)
        s = mx.broadcast_to(s[:, None], (*h.shape[:2], s.shape[-1]))
        return h, mu, self.duration(mx.concatenate([h, s], axis=-1))


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale[:, None]) + shift[:, None]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(attention(self.qkv(x), self.heads, self.q_norm, self.k_norm))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, (int(2 / 3 * dim * mlp_ratio) + 31) // 32 * 32)
        self.ada = nn.Linear(dim, 6 * dim)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        s1, sc1, g1, s2, sc2, g2 = mx.split(self.ada(nn.silu(c)), 6, axis=-1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, sc1))
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, sc2))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ada = nn.Linear(dim, 2 * dim)
        self.linear = nn.Linear(dim, out)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        shift, scale = mx.split(self.ada(nn.silu(c)), 2, axis=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class ScalarEmbedder(nn.Module):
    """Sinusoidal embedding of the CFG scale followed by an MLP (``mlp.0`` / ``mlp.2`` in PyTorch)."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = [nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)]

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None]
        x = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        for layer in self.mlp:
            x = layer(x)
        return x


class DriftDiT(nn.Module):
    def __init__(self, n_mels: int, cond_channels: int, hidden: int, depth: int, heads: int, patch: int,
                 mlp_ratio: float, n_registers: int, num_speakers: int, noise_classes: int, noise_coords: int):
        super().__init__()
        self.n_mels, self.patch, self.n_registers = n_mels, patch, n_registers
        self.noise_classes, self.noise_coords = noise_classes, noise_coords
        self.in_proj = nn.Linear((n_mels + cond_channels) * patch, hidden)
        self.spk = nn.Embedding(num_speakers, hidden)
        self.noise_embeds = [nn.Embedding(noise_classes, hidden) for _ in range(noise_coords)]
        self.cfg_embed = ScalarEmbedder(hidden)
        self.cfg_norm = RMSNorm(hidden)
        if n_registers > 0:
            self.reg_proj = nn.Linear(hidden, hidden)
            self.registers = mx.zeros((1, n_registers, hidden))
        self.blocks = [DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)]
        self.final = FinalLayer(hidden, n_mels * patch)

    def condition(self, spk: mx.array, cfg_scale: mx.array, noise_labels: mx.array) -> mx.array:
        c = self.spk(spk)
        for i, e in enumerate(self.noise_embeds):
            c = c + e(noise_labels[:, i])
        return c + 0.02 * self.cfg_norm(self.cfg_embed(cfg_scale))

    def __call__(self, z: mx.array, cond: mx.array, spk: mx.array, cfg_scale: mx.array,
                 noise_labels: mx.array) -> mx.array:
        """Noise ``[B, T, n_mels]`` and frame condition ``[B, T, C]`` -> normalised mel ``[B, T, n_mels]``."""
        b, t, _ = z.shape
        p = self.patch
        x = mx.concatenate([z, cond], axis=-1)
        pad = (-t) % p
        if pad:
            x = mx.pad(x, [(0, 0), (0, pad), (0, 0)])
        n = x.shape[1] // p
        # patches as in PyTorch: features ordered channel-major, then position within the patch
        x = self.in_proj(x.reshape(b, n, p, -1).transpose(0, 1, 3, 2).reshape(b, n, -1))
        c = self.condition(spk, cfg_scale, noise_labels)
        if self.n_registers > 0:
            regs = self.reg_proj(c)[:, None] + self.registers
            x = mx.concatenate([mx.broadcast_to(regs, (b, *regs.shape[1:])), x], axis=1)
        for block in self.blocks:
            x = block(x, c)
        x = self.final(x, c)[:, self.n_registers:]
        x = x.reshape(b, n, self.n_mels, p).transpose(0, 1, 3, 2).reshape(b, n * p, self.n_mels)
        return x[:, :t]


class DriftingTTS(nn.Module):
    """Text encoder + pitch conditioning + DriftDiT, built from the ``model`` section of the training config."""

    def __init__(self, cfg: dict, num_speakers: int, n_vocab: int, n_mels: int = 100):
        super().__init__()
        t, g = cfg["text"], cfg["gen"]
        self.n_mels = n_mels
        self.encoder = TextEncoder(n_vocab, n_mels, t["d"], t["heads"], t["layers"], t["ffn"], num_speakers,
                                   t["spk_dim"])
        self.generator = DriftDiT(n_mels, n_mels + t["d"], g["hidden"], g["depth"], g["heads"], g["patch"],
                                  g["mlp_ratio"], g["n_registers"], num_speakers, g["noise_classes"],
                                  g["noise_coords"])
        self.residual_prior = bool(g.get("residual_prior", True))
        self.pitch_enabled = bool(cfg.get("pitch", {}).get("enabled", False))
        if self.pitch_enabled:
            self.pitch_predictor = DurationPredictor(t["d"] + t["spk_dim"])
            self.pitch_emb = nn.Conv1d(1, t["d"], 3, padding=1)
            self.lf0_stats = mx.array([5.0, 0.35])

    def encode(self, text: mx.array, spk: mx.array, pitch_shift: float = 0.0) -> tuple[mx.array, mx.array]:
        """Token ids ``[1, N]`` -> token condition ``[1, N, n_mels + d]`` (prior mean, text features) and log-durations
        ``[1, N]``. ``pitch_shift`` is in semitones."""
        h, mu, logw = self.encoder(text, spk)
        if self.pitch_enabled:
            s = mx.broadcast_to(self.encoder.spk(spk)[:, None], (*h.shape[:2], self.encoder.spk.weight.shape[1]))
            p = self.pitch_predictor(mx.concatenate([h, s], axis=-1))
            if pitch_shift:
                p = p + pitch_shift / 12 * math.log(2) / self.lf0_stats[1]
            h = h + self.pitch_emb(p)
        return mx.concatenate([mu, h], axis=-1), logw[..., 0]

    def generate(self, z: mx.array, cond: mx.array, spk: mx.array, cfg_scale: mx.array,
                 noise_labels: mx.array) -> mx.array:
        x = self.generator(z, cond, spk, cfg_scale, noise_labels)
        return x + cond[..., : self.n_mels] if self.residual_prior else x


def durations(logw: np.ndarray, scale: float) -> np.ndarray:
    """``ceil(exp(logw) * scale)`` in float32, as ``durations_to_alignment``."""
    w = np.exp(logw.astype(np.float32)) * np.float32(scale)
    return np.maximum(np.ceil(w), 0).astype(np.int64)
