# Model code adapted from https://github.com/OpenBMB/VoxCPM (src/voxcpm/modules/audiovae, commit f0c787f).
# Copyright OpenBMB. Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Changes: inference only, v1 and v2 in one module, weight norm folded at load time, no pydantic config.
"""The AudioVAE of VoxCPM (OpenBMB, Apache-2.0): 64-d latents at 25 Hz, causal encoder and decoder.

* ``voxcpm2`` (``openbmb/VoxCPM2``): 16 kHz in (encoder rates 2·5·8·8 = 640), 48 kHz out (decoder rates
  8·6·5·2·2·2 = 1920). Before every upsampling block the decoder scales and shifts its features with embeddings of a
  sample-rate bucket, ``bucketize(rate, [20000, 30000, 40000])``: the bandwidth to generate. VoxCPM2 always decodes
  with 48000 (bucket 3, full band), and so does this backend by default (``target_rate``).
* ``voxcpm1.5`` (``openbmb/VoxCPM1.5``): 44.1 kHz in and out (rates 2·3·6·7·7 = 1764 both ways), no conditioning.

The model code is a minimal inference port of ``voxcpm/modules/audiovae`` (https://github.com/OpenBMB/VoxCPM,
Apache-2.0), vendored because the ``voxcpm`` package pulls in its whole TTS stack. Only ``audiovae.pth`` and
``config.json`` are downloaded; weight norm is folded into the weights. ``encode`` returns the posterior mean ``mu``
(as VoxCPM does); ``posterior`` also returns ``exp(logvar / 2)``.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import AudioBackend
from .layers import Snake1d, load_vae_checkpoint

MODELS = {  # name -> (HF repo, pinned revision)
    "voxcpm2": ("openbmb/VoxCPM2", "32279effe8c19989596f05d353d1447f51d9e915"),
    "voxcpm1.5": ("openbmb/VoxCPM1.5", "8cdc403854cda0e3af12252d27da038fda5982ac"),
}
DEFAULTS = {  # AudioVAEConfig defaults of audio_vae.py (v1) and audio_vae_v2.py; config.json overrides them
    "v1": dict(encoder_dim=128, encoder_rates=[2, 5, 8, 8], latent_dim=64, decoder_dim=1536,
               decoder_rates=[8, 8, 5, 2], depthwise=True, sample_rate=16000, use_noise_block=False),
    "v2": dict(encoder_dim=128, encoder_rates=[2, 5, 8, 8], latent_dim=64, decoder_dim=2048,
               decoder_rates=[8, 6, 5, 2, 2, 2], depthwise=True, sample_rate=16000, out_sample_rate=48000,
               use_noise_block=False, sr_bin_boundaries=[20000, 30000, 40000], cond_type="scale_bias"),
}


class CausalConv(nn.Conv1d):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, dilation: int = 1, groups: int = 1,
                 left: int = 0):
        super().__init__(cin, cout, kernel, stride, dilation=dilation, groups=groups)
        self.left = left

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left, 0)))


class CausalConvT(nn.ConvTranspose1d):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__(cin, cout, 2 * stride, stride)

    def forward(self, x: Tensor) -> Tensor:
        y = super().forward(x)
        return y[..., : y.shape[-1] - self.stride[0]]


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, groups: int):
        super().__init__()
        self.block = nn.Sequential(Snake1d(dim), CausalConv(dim, dim, 7, dilation=dilation, groups=groups,
                                                           left=6 * dilation),
                                   Snake1d(dim), CausalConv(dim, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, groups: int, v2: bool):
        super().__init__()
        left = 2 * math.ceil(stride / 2) - (stride % 2 if v2 else 0)  # v1 pads one more sample for odd strides
        self.block = nn.Sequential(*(ResidualUnit(dim // 2, d, groups) for d in (1, 3, 9)), Snake1d(dim // 2),
                                   CausalConv(dim // 2, dim, 2 * stride, stride, left=left))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, strides: list[int], depthwise: bool, v2: bool):
        super().__init__()
        layers: list[nn.Module] = [CausalConv(1, d_model, 7, left=6)]
        for s in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, s, d_model // 2 if depthwise else 1, v2))
        self.block = nn.Sequential(*layers)
        self.fc_mu = CausalConv(d_model, latent_dim, 3, left=2)
        self.fc_logvar = CausalConv(d_model, latent_dim, 3, left=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.block(x)
        return self.fc_mu(h), self.fc_logvar(h)


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, groups: int):
        super().__init__()
        self.input_channels = cin
        self.block = nn.Sequential(Snake1d(cin), CausalConvT(cin, cout, stride),
                                   *(ResidualUnit(cout, d, groups) for d in (1, 3, 9)))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class RateCondition(nn.Module):
    """``cond_type="scale_bias"``: ``x * scale[bucket] + bias[bucket]`` per channel."""

    def __init__(self, dim: int, buckets: int):
        super().__init__()
        self.scale_embed, self.bias_embed = nn.Embedding(buckets, dim), nn.Embedding(buckets, dim)

    def forward(self, x: Tensor, bucket: Tensor) -> Tensor:
        return x * self.scale_embed(bucket)[..., None] + self.bias_embed(bucket)[..., None]


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, channels: int, rates: list[int], depthwise: bool,
                 sr_bin_boundaries: list[int] | None = None):
        super().__init__()
        layers: list[nn.Module] = ([CausalConv(latent_dim, latent_dim, 7, groups=latent_dim, left=6),
                                    CausalConv(latent_dim, channels, 1)] if depthwise
                                   else [CausalConv(latent_dim, channels, 7, left=6)])
        for i, s in enumerate(rates):
            cout = channels // 2 ** (i + 1)
            layers.append(DecoderBlock(channels // 2**i, cout, s, cout if depthwise else 1))
        layers += [Snake1d(cout), CausalConv(cout, 1, 7, left=6), nn.Tanh()]
        self.model = nn.ModuleList(layers)
        self.sr_cond_model = None
        if sr_bin_boundaries is not None:
            self.register_buffer("sr_bin_boundaries", torch.tensor(sr_bin_boundaries, dtype=torch.int32))
            n = len(sr_bin_boundaries) + 1
            self.sr_cond_model = nn.ModuleList([RateCondition(m.input_channels, n)
                                                if isinstance(m, DecoderBlock) else None for m in layers])

    def forward(self, x: Tensor, rate: int | None = None) -> Tensor:
        if self.sr_cond_model is None:
            for layer in self.model:
                x = layer(x)
            return x
        bucket = torch.bucketize(torch.full((x.shape[0],), rate, dtype=torch.int32, device=x.device),
                                 self.sr_bin_boundaries)
        for layer, c in zip(self.model, self.sr_cond_model):
            x = layer(x if c is None else c(x, bucket))
        return x


class AudioVAE(nn.Module):
    def __init__(self, config: dict, v2: bool = True):
        super().__init__()
        c = {**DEFAULTS["v2" if v2 else "v1"], **config}
        if c.get("use_noise_block") or c.get("cond_type", "scale_bias") != "scale_bias":
            raise NotImplementedError("noise blocks and other sample-rate conditionings are not ported")
        self.sample_rate = c["sample_rate"]
        self.out_sample_rate = c.get("out_sample_rate", self.sample_rate)
        self.hop_length = math.prod(c["encoder_rates"])
        self.latent_dim = c["latent_dim"]
        self.encoder = Encoder(c["encoder_dim"], c["latent_dim"], c["encoder_rates"], c["depthwise"], v2)
        self.decoder = Decoder(c["latent_dim"], c["decoder_dim"], c["decoder_rates"], c["depthwise"],
                               c.get("sr_bin_boundaries") if v2 else None)

    def posterior(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        """``[B, 1, S]`` -> mean and std ``[B, latent_dim, ceil(S / hop)]`` (zero-padded to whole frames)."""
        mu, logvar = self.encoder(F.pad(audio, (0, -audio.shape[-1] % self.hop_length)))
        return mu, (0.5 * logvar).exp()

    def decode(self, z: Tensor, rate: int | None = None) -> Tensor:
        """``[B, latent_dim, T]`` -> ``[B, 1, T * prod(decoder_rates)]``; ``rate``: the sample-rate bucket (v2)."""
        return self.decoder(z, rate or self.out_sample_rate)


class VoxCPMBackend(AudioBackend):
    frame_rate, causal = 25.0, True

    def __init__(self, name: str = "voxcpm2", device: str = "cuda", model: AudioVAE | None = None,
                 target_rate: int | None = None):
        """``name``: ``voxcpm2`` or ``voxcpm1.5``; ``model``: an ``AudioVAE`` (default: the released weights);
        ``target_rate``: VoxCPM2's output-bandwidth condition (default 48000, as VoxCPM2 decodes)."""
        if name not in MODELS:
            raise ValueError(f"unknown VoxCPM model {name!r}; choose from {', '.join(MODELS)}")
        if model is None:
            from huggingface_hub import hf_hub_download

            repo, revision = MODELS[name]
            config = json.loads(open(hf_hub_download(repo, "config.json", revision=revision)).read())
            model = AudioVAE(config["audio_vae_config"], v2=config.get("architecture") == "voxcpm2")
            model.load_state_dict(load_vae_checkpoint(hf_hub_download(repo, "audiovae.pth", revision=revision))[0])
        self.name, self.model, self.device = name, model.to(device).eval(), device
        self.input_rate, self.output_rate = model.sample_rate, model.out_sample_rate
        self.frame_rate = model.sample_rate / model.hop_length
        self.dim, self.target_rate = model.latent_dim, target_rate

    @torch.no_grad()
    def posterior(self, wav: Tensor, sr: int) -> tuple[Tensor, Tensor]:
        return self.model.posterior(self._input(wav, sr))

    def encode(self, wav: Tensor, sr: int) -> Tensor:
        return self.posterior(wav, sr)[0]

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        return self.model.decode(latent.float().to(self.device), self.target_rate)[:, 0].clamp(-1, 1)
