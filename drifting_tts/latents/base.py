"""The audio-backend interface: waveform <-> a sequence of ``dim``-d frames at ``frame_rate`` frames per second.

Frames and samples, for every backend:

* ``encode(wav, sr)`` resamples to ``input_rate`` (``n = ceil(S * input_rate / sr)`` samples) and returns
  ``T = num_frames(S, sr)`` frames: ``ceil(n / hop_in)`` for the VAEs (the input is right-padded to whole frames),
  ``n // hop_in`` for the BigVGAN mel (as ``prepare`` computes it).
* ``decode(latent)`` returns exactly ``T * hop_out`` samples at ``output_rate``. Frame ``t`` covers output samples
  ``[t * hop_out, (t + 1) * hop_out)``, i.e. the same time span as input samples ``[t * hop_in, (t + 1) * hop_in)``:
  there is no delay, so a decoded utterance lines up with its input sample by sample (cut it to
  ``ceil(S * output_rate / sr)`` samples to drop the padded tail).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import torch
from torch import Tensor


def resample(wav: Tensor, sr: int, target: int) -> Tensor:
    if sr == target:
        return wav
    import torchaudio

    return torchaudio.functional.resample(wav, sr, target)


class AudioBackend(ABC):
    """Encoder / decoder pair between waveforms and ``[B, dim, T]`` frame sequences (see the module docstring)."""

    name: str
    input_rate: int  # sample rate the encoder takes
    output_rate: int  # sample rate the decoder returns
    frame_rate: float  # frames per second
    dim: int  # channels per frame
    causal: bool = False  # the decoder needs no right context (streaming)
    device: str = "cpu"

    @property
    def hop_in(self) -> int:
        """Input samples per frame."""
        return round(self.input_rate / self.frame_rate)

    @property
    def hop_out(self) -> int:
        """Output samples per frame."""
        return round(self.output_rate / self.frame_rate)

    def num_frames(self, samples: int, sr: int) -> int:
        """Frames that ``encode`` returns for ``samples`` samples at ``sr``."""
        return math.ceil(-(-samples * self.input_rate // sr) / self.hop_in)

    def _input(self, wav: Tensor, sr: int) -> Tensor:
        """``[S]`` or ``[B, S]`` at ``sr`` -> ``[B, 1, S']`` at ``input_rate`` on the device."""
        wav = wav[None] if wav.ndim == 1 else wav
        return resample(wav.float().to(self.device), sr, self.input_rate)[:, None]

    @abstractmethod
    def encode(self, wav: Tensor, sr: int) -> Tensor:
        """Waveform ``[S]`` or ``[B, S]`` at ``sr`` -> frames ``[B, dim, T]`` (VAEs: the posterior mean)."""

    @abstractmethod
    def decode(self, latent: Tensor) -> Tensor:
        """Frames ``[B, dim, T]`` -> waveform ``[B, T * hop_out]`` at ``output_rate``, in ``[-1, 1]``."""

    def trainable_decoder(self) -> torch.nn.Module:
        """The decoder module that ``finetune-vocoder`` trains (``vocoder.arch: vae_decoder``); its forward pass is
        :meth:`decode_train`."""
        raise NotImplementedError(f"decoder fine-tuning is not implemented for the {self.name} backend")

    def decoder_weight_norm(self) -> dict[str, tuple[Tensor, Tensor]]:
        """``{module path in the trainable decoder: (weight_g, weight_v)}`` of the released (unfolded) weight norm,
        where known: fine-tuning re-parametrizes those convolutions as they were trained."""
        return {}

    def decode_train(self, latent: Tensor) -> Tensor:
        """:meth:`decode` with gradients and without the final clamp: ``[B, dim, T]`` on the device ->
        ``[B, T * hop_out]``."""
        raise NotImplementedError(f"decoder fine-tuning is not implemented for the {self.name} backend")


def stream_decode(backend: AudioBackend, latent: Tensor, first: int = 8, chunk: int = 64,
                  context: int = 8) -> Iterator[Tensor]:
    """Decode ``latent`` ``[1, dim, T]`` window by window; yields the waveform in pieces (``[samples]``).

    The first ``first`` frames, then ``chunk`` frames at a time, each decoded with ``context`` frames of left context
    and, unless the decoder is causal, of right context; only the window's own samples are kept."""
    t, hop, s = latent.shape[-1], backend.hop_out, 0
    right = 0 if backend.causal else context
    while s < t:
        e = min(t, s + (first if s == 0 else chunk))
        a, b = max(0, s - context), min(t, e + right)
        yield backend.decode(latent[..., a:b])[0, (s - a) * hop: (e - a) * hop]
        s = e


class LatentStats:
    """Per-channel normalisation ``(z - mean) / std`` of ``[..., dim, T]`` frames."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.mean, self.std = mean.float(), std.float()

    @classmethod
    def from_latents(cls, latents: Iterable[Tensor]) -> LatentStats:
        """Statistics over all frames of ``[dim, T]`` / ``[B, dim, T]`` tensors."""
        s = ss = None
        n = 0
        for z in latents:
            z = z.detach().double().cpu().reshape(-1, z.shape[-2], z.shape[-1]).transpose(0, 1).flatten(1)
            s = z.sum(1) if s is None else s + z.sum(1)
            ss = (z**2).sum(1) if ss is None else ss + (z**2).sum(1)
            n += z.shape[1]
        mean = s / n
        return cls(mean, (ss / n - mean**2).clamp_min(1e-12).sqrt())

    def _view(self, x: Tensor, z: Tensor) -> Tensor:
        return x.to(z.device)[:, None]

    def normalize(self, z: Tensor) -> Tensor:
        return (z - self._view(self.mean, z)) / self._view(self.std, z)

    def denormalize(self, z: Tensor) -> Tensor:
        return z * self._view(self.std, z) + self._view(self.mean, z)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> LatentStats:
        return cls(torch.tensor(d["mean"]), torch.tensor(d["std"]))
