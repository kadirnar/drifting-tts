"""The "vocoder" of a model trained on VAE latents: the VAE decoder, behind the interface of :class:`Vocoder`."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..audio import SAMPLE_RATE
from . import load_backend, resample


class LatentVocoder:
    """Denormalised latent frames ``[B, dim, T]`` at ``repeat`` x the VAE frame rate -> waveform at ``sample_rate``.

    ``extract-latents`` repeats every VAE frame ``repeat`` times (so that character-level alignment has enough frames
    per token); the repeats are averaged back before decoding. The output is resampled to ``sample_rate`` (24 kHz by
    default, as the rest of the pipeline), or left at the VAE's own rate with ``sample_rate=None``."""

    # the attributes of drifting_tts.vocoder.Vocoder that Synthesizer reads: no streaming windows, no CUDA graphs
    kind, context, graphs = "vae", None, False

    def __init__(self, backend: str, device: str = "cuda", repeat: int = 4, sample_rate: int | None = SAMPLE_RATE):
        self.backend = load_backend(backend, device)
        self.name = self.mel = backend
        self.device, self.repeat = device, repeat
        self.sample_rate = sample_rate or self.backend.output_rate

    @torch.no_grad()
    def __call__(self, frames: Tensor) -> Tensor:
        z = frames.to(self.device).float()
        if self.repeat > 1:
            t = z.shape[-1] // self.repeat * self.repeat
            z = F.avg_pool1d(z[..., :t], self.repeat)
        wav = self.backend.decode(z)
        if self.sample_rate != self.backend.output_rate:
            wav = resample(wav, self.backend.output_rate, self.sample_rate)
        return wav.clamp(-1, 1)
