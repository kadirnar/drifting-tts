"""The "vocoder" of a model trained on VAE latents: the VAE decoder, behind the interface of :class:`Vocoder`."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ..audio import SAMPLE_RATE
from . import load_backend, resample


def load_decoder_checkpoint(spec: str | Path | dict) -> dict:
    """A fine-tuned VAE decoder written by ``finetune-vocoder`` (``vocoder.arch: vae_decoder``): a path or the
    loaded dict ``{"decoder": state dict, "backend": "voxcpm2", "target_rate", "sample_rate", "step"}``. Loaded as
    weights only."""
    ck = spec if isinstance(spec, dict) else torch.load(spec, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "decoder" not in ck or "backend" not in ck:
        raise ValueError(f"{spec if not isinstance(spec, dict) else 'checkpoint'} is not a fine-tuned VAE decoder "
                         "(decoder_ft.pt of `finetune-vocoder` with vocoder.arch: vae_decoder)")
    return ck


def native_frames(frames: Tensor, repeat: int) -> Tensor:
    """``[..., dim, T]`` frames stored ``repeat`` times each -> ``[..., dim, ceil(T / repeat)]``: the mean of every
    group of ``repeat``, the last one over the frames it has (a generated length need not be a multiple)."""
    if repeat <= 1:
        return frames
    t = frames.shape[-1] // repeat * repeat
    whole = frames[..., :t].unflatten(-1, (-1, repeat)).mean(-1)
    return whole if t == frames.shape[-1] else torch.cat([whole, frames[..., t:].mean(-1, keepdim=True)], -1)


class LatentVocoder:
    """Denormalised latent frames ``[B, dim, T]`` at ``repeat`` x the VAE frame rate -> waveform at ``sample_rate``.

    ``extract-latents`` repeats every VAE frame ``repeat`` times (so that character-level alignment has enough frames
    per token); the repeats are averaged back before decoding (:func:`native_frames`). The waveform has the length
    of the ``T`` frames: a last group of fewer than ``repeat`` frames is decoded as one VAE frame and cut to its
    share. The output is resampled to ``sample_rate`` (24 kHz by default, as the rest of the pipeline), or left at
    the VAE's own rate with ``sample_rate=None``.

    ``decoder``: a fine-tuned decoder (``decoder_ft.pt`` of ``finetune-vocoder``, a path or the loaded dict) in place
    of the released one; it must come from the same backend. It was trained on 24 kHz losses, so it is meant for the
    default ``sample_rate``, and its output is resampled with the filter it was trained through: a decoder
    fine-tuned before #94 learned to make up for the roll-off of torchaudio's default filter above 8 kHz, and the
    sharp one (:func:`drifting_tts.audio.resample`) would leave that boost in the audio."""

    # the attributes of drifting_tts.vocoder.Vocoder that Synthesizer reads: no streaming windows, no CUDA graphs
    kind, context, graphs = "vae", None, False

    def __init__(self, backend: str, device: str = "cuda", repeat: int = 4, sample_rate: int | None = SAMPLE_RATE,
                 decoder: str | Path | dict | None = None):
        kwargs = {}
        if decoder is not None:
            ck = load_decoder_checkpoint(decoder)
            if ck["backend"] != backend:
                raise ValueError(f"the decoder checkpoint is a {ck['backend']} decoder, but the model produces "
                                 f"{backend} latents")
            kwargs["decoder"] = ck
        self.backend = load_backend(backend, device, **kwargs)
        self.mel = backend
        self.name = f"{backend} (fine-tuned decoder)" if decoder is not None else backend
        self.device, self.repeat = device, repeat
        self.sample_rate = sample_rate or self.backend.output_rate
        self.sharp = decoder is None or ck.get("resample") == "sharp"

    @torch.no_grad()
    def __call__(self, frames: Tensor) -> Tensor:
        z = native_frames(frames.to(self.device).float(), self.repeat)
        wav = self.backend.decode(z)
        if frames.shape[-1] % max(self.repeat, 1):  # the last VAE frame stands for fewer than ``repeat`` frames
            wav = wav[..., : round(frames.shape[-1] * self.backend.hop_out / self.repeat)]
        wav = resample(wav, self.backend.output_rate, self.sample_rate, sharp=self.sharp)
        return wav.clamp(-1, 1)
