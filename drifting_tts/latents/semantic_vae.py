# Encoder and projection code adapted from https://github.com/ZhikangNiu/Semantic-VAE (MIT; its encoder derives
# from Descript's DAC, MIT). Changes: inference only, one module, weight norm folded at load time, the posterior
# mean, the decoder built from NVIDIA's BigVGAN code.
"""Semantic-VAE (Niu et al., https://arxiv.org/abs/2509.22167; weights ``zkniu/Semantic-VAE``): 16 kHz in and out,
64-d latents at 40 Hz.

A VAE whose latent is pulled towards WavLM features by a cosine loss while it is trained to reconstruct: the paper
reports the reconstruction unchanged (PESQ 3.74 against 3.75) and F5-TTS on these latents at 2.10% WER / 0.64 SIM on
LibriSpeech-PC against 2.65% / 0.60 on the plain VAE. Here it tests one hypothesis (#93): the drifting kernel is an
L2 distance, so it needs a space in which L2 means something. A 16 kHz codec is not the final audio band.

* Encoder: DAC's convolution stack (strides 4, 4, 5, 5: 400 samples per frame, 1,024 channels), a causal
  attention projection to 64 dimensions, then the posterior's mean and log-variance.
* Decoder: BigVGAN-v2 (snake-beta, anti-aliased; upsampling 5, 5, 2, 2, 2, 2) from the 64-d latent.
* ``encode`` returns the posterior mean; ``posterior`` also the standard deviation, for training on samples (#92).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import AudioBackend
from .layers import Snake1d

REPO, REVISION = "zkniu/Semantic-VAE", None
CHECKPOINTS = ("semantic_vae", "semantic_vae_1000k", "acoustic_vae_dim64", "acoustic_vae_dim16")
DECODER = {"resblock": "1", "upsample_rates": [5, 5, 2, 2, 2, 2], "upsample_kernel_sizes": [9, 9, 4, 4, 4, 4],
           "upsample_initial_channel": 1024, "resblock_kernel_sizes": [3, 7, 11],
           "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "use_tanh_at_final": False,
           "use_bias_at_final": False, "activation": "snakebeta", "snake_logscale": True}


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(Snake1d(dim), nn.Conv1d(dim, dim, 7, dilation=dilation, padding=3 * dilation),
                                   Snake1d(dim), nn.Conv1d(dim, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(ResidualUnit(dim // 2, 1), ResidualUnit(dim // 2, 3), ResidualUnit(dim // 2, 9),
                                   Snake1d(dim // 2),
                                   nn.Conv1d(dim // 2, dim, 2 * stride, stride, padding=math.ceil(stride / 2)))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, dim: int, strides: list[int]):
        super().__init__()
        block: list[nn.Module] = [nn.Conv1d(1, dim, 7, padding=3)]
        for stride in strides:
            dim *= 2
            block.append(EncoderBlock(dim, stride))
        block += [Snake1d(dim), nn.Conv1d(dim, dim, 3, padding=1)]
        self.block, self.dim = nn.Sequential(*block), dim

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w0, self.w1, self.w2 = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.w2(F.gelu(self.w0(x), approximate="tanh") * self.w1(x))


class ProjAttention(nn.Module):
    """Causal attention that narrows ``in_dim`` to ``out_dim``: the heads' outputs are averaged (and pooled when a
    head is wider than ``out_dim``)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.qkv = nn.Linear(in_dim, 3 * in_dim, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(in_dim)), nn.Parameter(torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        bias = torch.cat([self.q_bias, torch.zeros_like(self.q_bias), self.v_bias])
        q, k, v = F.linear(x, self.qkv.weight, bias).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).mean(1)  # [B, N, C / heads]
        if y.shape[-1] != self.out_dim:
            y = F.adaptive_avg_pool1d(y, self.out_dim)
        return self.proj(y)


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 8):
        super().__init__()
        self.norm1, self.norm3, self.norm2 = nn.LayerNorm(in_dim), nn.LayerNorm(in_dim), nn.LayerNorm(out_dim)
        self.attn, self.proj = ProjAttention(in_dim, out_dim, heads), nn.Linear(in_dim, out_dim)
        self.mlp = GeGluMlp(out_dim, 2 * out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def fold_parametrized(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """``parametrizations.weight.original0`` / ``original1`` (``torch.nn.utils.parametrizations.weight_norm``: the
    magnitude and the direction) -> ``weight``."""
    out = {}
    for k, v in state.items():
        if k.endswith("parametrizations.weight.original0"):
            continue
        if k.endswith("parametrizations.weight.original1"):
            g = state[k[:-1] + "0"]
            k = k.removesuffix("parametrizations.weight.original1") + "weight"
            v = v * (g / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.ndim - 1)))
        out[k] = v
    return out


class SemanticVAEBackend(AudioBackend):
    name = "semantic-vae"
    input_rate = output_rate = 16000
    frame_rate = 40.0
    dim = 64

    def __init__(self, device: str = "cuda", checkpoint: str = "semantic_vae"):
        from huggingface_hub import hf_hub_download

        from ..vocoder import BIGVGAN_REPO, bigvgan_snapshot

        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown Semantic-VAE checkpoint {checkpoint!r}; choose from {', '.join(CHECKPOINTS)}")
        self.device, self.checkpoint = device, checkpoint
        path = hf_hub_download(REPO, f"{checkpoint}/dac/ema_state_dict.pth", revision=REVISION)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = fold_parametrized({k.removeprefix("ema_model."): v for k, v in state.items()
                                   if k.startswith("ema_model.")})
        self.dim = state["fc_mu.weight"].shape[0]
        self.encoder = Encoder(64, [4, 4, 5, 5])
        self.pre_block = AttnProjection(self.encoder.dim, self.dim)
        self.fc_mu, self.fc_var = nn.Linear(self.dim, self.dim), nn.Linear(self.dim, self.dim)
        bigvgan, _ = bigvgan_snapshot(BIGVGAN_REPO, weights=False)  # NVIDIA's code; the weights are the VAE's
        self.decoder = bigvgan.BigVGAN(bigvgan.AttrDict({**DECODER, "num_mels": self.dim}), use_cuda_kernel=False)
        self.decoder.remove_weight_norm()
        for name, module in (("encoder", self.encoder), ("pre_block", self.pre_block), ("fc_mu", self.fc_mu),
                             ("fc_var", self.fc_var), ("decoder", self.decoder)):
            part = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
            part.pop("attn.zero_k_bias", None)
            module.load_state_dict(part, strict=True)
            module.to(device).eval().requires_grad_(False)

    def _posterior(self, wav: Tensor, sr: int) -> tuple[Tensor, Tensor]:
        x = self._input(wav, sr)
        x = F.pad(x, (0, (-x.shape[-1]) % self.hop_in))
        h = self.pre_block(self.encoder(x).transpose(1, 2))
        return self.fc_mu(h).transpose(1, 2), self.fc_var(h).clamp(-12, 12).transpose(1, 2)

    @torch.no_grad()
    def encode(self, wav: Tensor, sr: int) -> Tensor:
        return self._posterior(wav, sr)[0]

    @torch.no_grad()
    def posterior(self, wav: Tensor, sr: int) -> tuple[Tensor, Tensor]:
        """The posterior's mean and standard deviation ``[B, dim, T]`` (the model was trained on its samples)."""
        mu, log_var = self._posterior(wav, sr)
        return mu, (0.5 * log_var).exp()

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent.to(self.device).float())[:, 0].clamp(-1, 1)
