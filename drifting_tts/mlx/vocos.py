"""Vocos (charactr/vocos-mel-24khz architecture, MIT) in MLX for inference: a ConvNeXt backbone and an ISTFT head.

Only the ``same``-padded head of the fine-tune on BigVGAN-style mels is supported: frame ``i`` is centred on sample
``i * hop + hop / 2``, so ``T`` frames give ``T * hop`` samples (``drifting_tts.vocoder.ola_istft``). Channels-last
internally; parameter names follow the PyTorch state dict.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .bigvgan import to_numpy


def vocos_context_frames(h: dict) -> int:
    """Mel frames of context on each side that make a decoded chunk equal the whole-utterance output: the receptive
    field of the 7-tap embedding and depthwise convolutions, plus the frames whose ISTFT windows overlap the chunk
    (29 for the 8-layer, n_fft 1024, hop 256 model)."""
    return 3 + 3 * h["num_layers"] + (h["n_fft"] // 2 + h["hop_length"] // 2 - 1) // h["hop_length"]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        # zero padding is applied explicitly: MLX's Metal depthwise kernel only handles unpadded convolutions
        self.dwconv = nn.Conv1d(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.dwconv(mx.pad(x, [(0, 0), (3, 3), (0, 0)]))
        return x + self.gamma * self.pwconv2(nn.gelu(self.pwconv1(self.norm(y))))


class VocosBackbone(nn.Module):
    def __init__(self, input_channels: int, dim: int, intermediate_dim: int, num_layers: int):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = [ConvNeXtBlock(dim, intermediate_dim) for _ in range(num_layers)]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(self.embed(x))
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x)


def overlap_add(frames: mx.array, hop: int) -> mx.array:
    """``[B, T, n_fft]`` -> ``[B, (T - 1) * hop + n_fft]``: the frames as ``n_fft / hop`` blocks of ``hop`` samples,
    each block shifted into place and summed (no scatter)."""
    b, t, n = frames.shape
    r = n // hop
    blocks = frames.reshape(b, t, r, hop)
    y = sum(mx.pad(blocks[:, :, k], [(0, 0), (k, r - 1 - k), (0, 0)]) for k in range(r))
    return y.reshape(b, (t + r - 1) * hop)


class ISTFTHead(nn.Module):
    """Linear -> log-magnitude and phase -> inverse real FFT -> windowed overlap-add divided by the squared-window
    envelope, cropped by ``(n_fft - hop) / 2`` on each side (Vocos's ``ISTFT(padding="same")``)."""

    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        if n_fft % hop_length:
            raise ValueError("n_fft must be a multiple of hop_length")
        self.out = nn.Linear(dim, n_fft + 2)
        self.n_fft, self.hop_length = n_fft, hop_length
        n = np.arange(n_fft)  # torch.hann_window (periodic)
        self._window = mx.array((0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)).astype(np.float32))

    def __call__(self, x: mx.array) -> mx.array:
        mag, p = mx.split(self.out(x).astype(mx.float32), 2, axis=-1)
        spec = mx.minimum(mx.exp(mag), 1e2) * (mx.cos(p) + 1j * mx.sin(p))
        frames = mx.fft.irfft(spec, n=self.n_fft, axis=-1) * self._window
        t, pad = frames.shape[1], (self.n_fft - self.hop_length) // 2
        env = overlap_add(mx.broadcast_to(mx.square(self._window), (1, t, self.n_fft)), self.hop_length)
        return (overlap_add(frames, self.hop_length) / env)[:, pad: pad + t * self.hop_length]


class Vocos(nn.Module):
    """Log-mel ``[B, input_channels, T]`` -> waveform ``[B, T * hop_length]`` (unclamped). ``h``: ``input_channels``,
    ``dim``, ``intermediate_dim``, ``num_layers``, ``n_fft``, ``hop_length``."""

    def __init__(self, h: dict):
        super().__init__()
        if h.get("padding", "same") != "same":
            raise NotImplementedError("only the 'same'-padded ISTFT head (BigVGAN framing) is supported")
        self.backbone = VocosBackbone(h["input_channels"], h["dim"], h["intermediate_dim"], h["num_layers"])
        self.head = ISTFTHead(h["dim"], h["n_fft"], h["hop_length"])
        self.hop_length = h["hop_length"]
        self.context_frames = vocos_context_frames(h)

    def prepare_for_inference(self) -> Vocos:
        mx.eval(self.head._window)
        return self

    def __call__(self, mel: mx.array) -> mx.array:
        return self.head(self.backbone(mel.transpose(0, 2, 1).astype(self.backbone.embed.weight.dtype)))


def vocos_hparams(model) -> dict:
    """The MLX hyper-parameters of a PyTorch ``vocos.Vocos`` with a ``same``-padded ISTFT head."""
    bb, head = model.backbone, model.head
    return {"input_channels": bb.embed.in_channels, "dim": bb.embed.out_channels,
            "intermediate_dim": bb.convnext[0].pwconv1.out_features, "num_layers": len(bb.convnext),
            "n_fft": head.istft.n_fft, "hop_length": head.istft.hop_length, "padding": head.istft.padding}


def convert_vocos(state_dict: dict) -> dict[str, np.ndarray]:
    """PyTorch Vocos state dict -> float32 MLX weights keyed as :class:`Vocos`'s parameters (the mel front end and
    the ISTFT window are dropped)."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("feature_extractor.") or k.endswith("istft.window"):
            continue
        a = to_numpy(v)
        if a.ndim == 3:  # Conv1d [out, in / groups, k] -> [out, k, in / groups]
            a = a.transpose(0, 2, 1)
        out[k] = np.ascontiguousarray(a, dtype=np.float32)
    return out
