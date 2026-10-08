# Model code adapted from https://github.com/facebookresearch/dacvae (commit 414c207).
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Changes: inference only, one module, weight norm folded at load time, posterior mean, fixed watermark message,
# hooks for fine-tuning the decoder's audio path with the watermark frozen.
"""DAC-VAE (``facebook/dacvae-watermarked``, Apache-2.0): 48 kHz in and out, 128-d latents at 25 Hz.

The model code is a minimal inference port of https://github.com/facebookresearch/dacvae (Apache-2.0, Copyright (c)
Meta Platforms, Inc. and affiliates; derived from Descript's DAC, MIT). It is vendored because the package's
``descript-audiotools`` dependency pins ``protobuf<3.20``. Weight norm is folded into the weights at load time.

* ``encode`` returns the posterior mean (the upstream ``DACVAE.encode`` returns a sample); ``posterior`` also returns
  the standard deviation ``softplus(scale) + 1e-4``.
* The decoder adds an AudioSeal-style watermark through a 150 Hz branch with two LSTMs. Upstream draws a random
  16-bit message per call; here the message is fixed (``message``), so decoding is deterministic.
* Encoder and decoder are non-causal convolution stacks (symmetric padding); the watermark LSTMs are causal.
* Decoder fine-tuning (``finetune-vocoder`` with ``vocoder.arch: vae_decoder``) trains the audio path only
  (:meth:`Decoder.audio_path`). The watermark (``wm_model`` and the watermark layers of every ``DecoderBlock``) is
  frozen and still added to the output, at training and at inference time, exactly as by the released decoder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import AudioBackend
from .layers import Snake1d, load_vae_checkpoint

REPO, REVISION = "facebook/dacvae-watermarked", "8680102d141858a21bd533543966a2eb2e569f92"


class Conv(nn.Conv1d):
    """Upstream ``NormConv1d``: ``pad_mode="none"`` (static symmetric padding) or ``"auto"`` (causal: left padding
    plus what completes the last stride)."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, dilation: int = 1, causal: bool = False):
        super().__init__(cin, cout, kernel, stride, 0 if causal else (kernel - stride) * dilation // 2, dilation)
        self.causal = causal

    def forward(self, x: Tensor) -> Tensor:
        if self.causal:
            k, s, d = self.kernel_size[0], self.stride[0], self.dilation[0]
            total = (k - 1) * d + 1 - s
            frames = math.ceil((x.shape[-1] - (k - 1) * d - 1 + total) / s + 1)
            x = F.pad(x, (total, (frames - 1) * s + k - total - x.shape[-1]))
        return super().forward(x)


class ConvT(nn.ConvTranspose1d):
    """Upstream ``NormConvTranspose1d``: static padding, or causal (trim ``kernel - stride`` samples at the end)."""

    def __init__(self, cin: int, cout: int, stride: int, causal: bool = False):
        super().__init__(cin, cout, 2 * stride, stride, 0 if causal else (stride + 1) // 2,
                         0 if causal else stride % 2)
        self.causal = causal

    def forward(self, x: Tensor) -> Tensor:
        y = super().forward(x)
        return y[..., : y.shape[-1] - self.stride[0]] if self.causal else y


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, kernel: int = 7, dilation: int = 1, elu: bool = False):
        super().__init__()
        hidden = dim // 2 if elu else dim
        act = (lambda c: nn.ELU()) if elu else Snake1d
        self.block = nn.Sequential(act(dim), Conv(dim, hidden, kernel, dilation=dilation, causal=elu),
                                   act(hidden), Conv(hidden, dim, 1, causal=elu))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(*(ResidualUnit(dim // 2, dilation=d) for d in (1, 3, 9)), Snake1d(dim // 2),
                                   Conv(dim // 2, dim, 2 * stride, stride))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, strides: list[int], d_latent: int):
        super().__init__()
        layers: list[nn.Module] = [Conv(1, d_model, 7)]
        for s in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, s))
        self.block = nn.Sequential(*layers, Snake1d(d_model), Conv(d_model, d_latent, 3))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VAEBottleneck(nn.Module):
    def __init__(self, input_dim: int, codebook_dim: int):
        super().__init__()
        self.in_proj, self.out_proj = Conv(input_dim, 2 * codebook_dim, 1), Conv(codebook_dim, input_dim, 1)


class DecoderBlock(nn.Module):
    """Upsampling block whose layer list interleaves the audio path (``MAIN``) with the watermark's down / up paths
    (``WM_DOWN``, ``WM_UP``), in the upstream order (so the state dict loads as is)."""

    MAIN, WM_DOWN, WM_UP = (0, 1, 4, 5, 8, 9), (7, 10, 11), (2, 3, 6)

    def __init__(self, cin: int, cout: int, stride: int, wm_stride: int):
        super().__init__()
        self.block = nn.ModuleList([
            Snake1d(cin), ConvT(cin, cout, stride),
            nn.ELU(), ConvT(cin // 3, cout // 3, wm_stride, causal=True),
            ResidualUnit(cout, dilation=1), ResidualUnit(cout, dilation=3),
            ResidualUnit(cout // 3, kernel=3, elu=True), ResidualUnit(cout // 3, kernel=3, elu=True),
            ResidualUnit(cout, dilation=9), nn.Identity(),
            nn.ELU(), Conv(cout // 3, cin // 3, 2 * wm_stride, wm_stride, causal=True),
        ])

    def run(self, x: Tensor, layers: tuple[int, ...]) -> Tensor:
        for i in layers:
            x = self.block[i](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.run(x, self.MAIN)


class LSTMBlock(nn.Module):
    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(2, 0, 1)
        return (self.lstm(x)[0] + x).permute(1, 2, 0)


class MsgProcessor(nn.Module):
    def __init__(self, nbits: int, hidden: int):
        super().__init__()
        self.msg_processor = nn.Embedding(2 * nbits, hidden)

    def forward(self, h: Tensor, msg: Tensor) -> Tensor:
        idx = 2 * torch.arange(msg.shape[-1], device=h.device) + msg.long()
        return h + self.msg_processor(idx).sum(-2)[..., None]


class Watermarker(nn.Module):
    def __init__(self, dim: int, channels: int, d_latent: int = 128, hidden: int = 512, nbits: int = 16):
        super().__init__()
        enc, dec = nn.Module(), nn.Module()
        enc.pre = nn.Sequential(Snake1d(dim), Conv(dim, 1, 7), nn.Tanh(), Conv(1, channels, 7, causal=True))
        enc.post = nn.Sequential(LSTMBlock(hidden, 2), nn.ELU(), Conv(hidden, d_latent, 7, causal=True))
        dec.pre = nn.Sequential(Conv(d_latent, hidden, 7, causal=True), LSTMBlock(hidden, 2))
        dec.post = nn.Sequential(nn.ELU(), Conv(channels, 1, 7, causal=True))
        self.encoder_block, self.decoder_block = enc, dec
        self.msg_processor = MsgProcessor(nbits, d_latent)
        self.nbits, self.alpha = nbits, channels / d_latent


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, channels: int, rates: list[int], wm_rates: list[int], wm_latent: int = 128):
        super().__init__()
        blocks = [DecoderBlock(channels // 2**i, channels // 2 ** (i + 1), s, w)
                  for i, (s, w) in enumerate(zip(rates, wm_rates))]
        self.model = nn.ModuleList([Conv(latent_dim, channels, 7), *blocks])
        out = channels // 2 ** len(rates)
        self.wm_model = Watermarker(out, out // 3, wm_latent, channels // 3)

    def audio_path(self) -> list[nn.Module]:
        """The modules that make the audio before the watermark: the input convolution and the ``MAIN`` layers of
        every block. The output layer (Snake, the convolution to one channel, tanh) sits inside
        ``wm_model.encoder_block.pre`` upstream and is left out with the rest of ``wm_model``."""
        return [self.model[0]] + [b.block[i] for b in self.model[1:] for i in DecoderBlock.MAIN]

    def components(self, x: Tensor, message: Tensor) -> tuple[Tensor, Tensor]:
        """``(y, w)``: the audio before the watermark and the watermark ``alpha * post(h)`` that the decoder adds to
        it, each ``[B, 1, samples]``. The watermark is computed from ``y`` and the message only."""
        for layer in self.model:
            x = layer(x)
        y = self.wm_model.encoder_block.pre[:3](x)  # the audio before the watermark
        return y, self.watermark(y, message)

    def watermark(self, y: Tensor, message: Tensor) -> Tensor:
        """The watermark ``alpha * post(h)`` that the decoder adds to the audio ``y`` ``[B, 1, samples]``."""
        wm, blocks = self.wm_model, self.model[1:]
        h = wm.encoder_block.pre[3](y)
        for b in reversed(blocks):
            h = b.run(h, DecoderBlock.WM_DOWN)
        h = wm.decoder_block.pre(wm.msg_processor(wm.encoder_block.post(h), message.to(h.device)))
        for b in blocks:
            h = b.run(h, DecoderBlock.WM_UP)
        return wm.alpha * wm.decoder_block.post(h)

    def forward(self, x: Tensor, message: Tensor) -> Tensor:
        y, w = self.components(x, message)
        return y + w


class DACVAE(nn.Module):
    def __init__(self, encoder_dim: int = 64, encoder_rates: Sequence[int] = (2, 8, 10, 12), latent_dim: int = 1024,
                 decoder_dim: int = 1536, decoder_rates: Sequence[int] = (12, 10, 8, 2), codebook_dim: int = 128,
                 wm_rates: Sequence[int] = (8, 5, 4, 2), wm_latent: int = 128, sample_rate: int = 48_000, **_):
        """The released model's ``metadata.kwargs`` (codebook count / size are unused by the VAE bottleneck)."""
        super().__init__()
        self.sample_rate, self.hop_length = sample_rate, math.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, list(encoder_rates), latent_dim)
        self.quantizer = VAEBottleneck(latent_dim, codebook_dim)
        self.decoder = Decoder(latent_dim, decoder_dim, list(decoder_rates), list(wm_rates), wm_latent)

    def posterior(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        """``[B, 1, S]`` -> mean and std ``[B, codebook_dim, ceil(S / hop)]`` (reflect-padded to whole frames)."""
        pad = -audio.shape[-1] % self.hop_length
        if pad:
            audio = F.pad(audio, (0, pad), "reflect" if pad < audio.shape[-1] else "constant")
        mean, scale = self.quantizer.in_proj(self.encoder(audio)).chunk(2, dim=1)
        return mean, F.softplus(scale) + 1e-4

    def decode(self, z: Tensor, message: Tensor) -> Tensor:
        """``[B, codebook_dim, T]`` -> ``[B, 1, T * hop]``."""
        return self.decoder(self.quantizer.out_proj(z), message)


class DACVAEBackend(AudioBackend):
    name, input_rate, output_rate, frame_rate, dim = "dacvae", 48_000, 48_000, 25.0, 128

    def __init__(self, device: str = "cuda", model: DACVAE | None = None, message: Tensor | None = None,
                 decoder: dict | None = None):
        """``model``: a ``DACVAE`` (default: the released weights); ``message``: the watermark's 16 bits (zeros);
        ``decoder``: a fine-tuned decoder (``decoder_ft.pt`` of ``finetune-vocoder``, loaded with
        :func:`drifting_tts.latents.vocoder.load_decoder_checkpoint`) that replaces the released decoder weights. Its
        watermark weights are the released ones (they are frozen in fine-tuning)."""
        self.released = model is None and decoder is None  # the released weights, unchanged
        if model is None:
            from huggingface_hub import hf_hub_download

            state, kwargs = load_vae_checkpoint(hf_hub_download(REPO, "weights.pth", revision=REVISION))
            model = DACVAE(**kwargs)
            model.load_state_dict(state)
        if decoder is not None:
            model.decoder.load_state_dict(decoder["decoder"])
        self.model, self.device = model.to(device).eval(), device
        self.input_rate = self.output_rate = model.sample_rate
        self.frame_rate = model.sample_rate / model.hop_length
        self.dim = model.quantizer.out_proj.in_channels
        nbits = model.decoder.wm_model.nbits
        self.message = (torch.zeros(1, nbits) if message is None else message.reshape(1, nbits).float()).to(device)

    @torch.no_grad()
    def posterior(self, wav: Tensor, sr: int) -> tuple[Tensor, Tensor]:
        return self.model.posterior(self._input(wav, sr))

    def encode(self, wav: Tensor, sr: int) -> Tensor:
        return self.posterior(wav, sr)[0]

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        return self.decode_train(latent.float().to(self.device)).clamp(-1, 1)

    def trainable_decoder(self) -> nn.Module:
        """The decoder, with only its audio path trainable (:meth:`Decoder.audio_path`). Everything else is frozen:
        the encoder, the bottleneck projections and the whole watermark (``wm_model``, including the audio's output
        layer, and the watermark layers of every block). :meth:`decode_train` still adds the watermark."""
        self.model.requires_grad_(False)
        for m in self.model.decoder.audio_path():
            m.requires_grad_(True)
        return self.model.decoder

    def decoder_weight_norm(self) -> dict[str, tuple[Tensor, Tensor]]:
        """The released decoder's weight norm before folding, ``{module path in the decoder: (weight_g, weight_v)}``
        (``dim=0``): the audio path's convolutions and the output layer's (the watermark's own convolutions have
        none). Empty unless the backend holds the released weights."""
        if not self.released:
            return {}
        from huggingface_hub import hf_hub_download

        state = torch.load(hf_hub_download(REPO, "weights.pth", revision=REVISION), map_location="cpu",
                           weights_only=True)["state_dict"]
        return {k[len("decoder."): -len(".weight_g")]: (v, state[k[:-1] + "v"]) for k, v in state.items()
                if k.startswith("decoder.") and k.endswith(".weight_g")}

    def decode_train(self, latent: Tensor) -> Tensor:
        """:meth:`decode` with gradients and without the clamp; the watermark is added as at release."""
        return self.model.decode(latent, self.message.expand(latent.shape[0], -1))[:, 0]

    @torch.no_grad()
    def watermark_components(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """``(y, w)`` ``[B, samples]``: the audio before the watermark and the watermark that ``decode`` adds to it
        (before the clamp)."""
        z = self.model.quantizer.out_proj(latent.float().to(self.device))
        y, w = self.model.decoder.components(z, self.message.expand(z.shape[0], -1))
        return y[:, 0], w[:, 0]
