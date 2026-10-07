"""The BigVGAN-v2 log-mel (24 kHz, 100 bins, hop 256, 93.75 frames / s) as an audio backend."""

from __future__ import annotations

import torch
from torch import Tensor

from ..audio import HOP_LENGTH, N_MELS, SAMPLE_RATE, BigVGANLogMel
from .base import AudioBackend


class BigVGANMel(AudioBackend):
    """``encode``: :class:`~drifting_tts.audio.BigVGANLogMel` (``S // 256`` frames, as ``prepare`` stores them);
    ``decode``: the BigVGAN-v2 vocoder (loaded on first use)."""

    name, input_rate, output_rate, dim = "bigvgan", SAMPLE_RATE, SAMPLE_RATE, N_MELS
    frame_rate = SAMPLE_RATE / HOP_LENGTH

    def __init__(self, device: str = "cuda", finetuned: str | None = None, cuda_kernel: bool = False):
        """``finetuned``: a ``bigvgan_ft.pt`` (``drifting-tts finetune-vocoder``); ``cuda_kernel``: fused activation."""
        self.device, self.finetuned, self.cuda_kernel = device, finetuned, cuda_kernel
        self.mel = BigVGANLogMel().to(device)
        self._vocoder = None

    def num_frames(self, samples: int, sr: int) -> int:
        return -(-samples * self.input_rate // sr) // self.hop_in

    def encode(self, wav: Tensor, sr: int) -> Tensor:
        return self.mel(self._input(wav, sr)[:, 0])

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        if self._vocoder is None:
            from ..vocoder import Vocoder

            self._vocoder = Vocoder(self.device, finetuned=self.finetuned, backend="bigvgan",
                                    cuda_kernel=self.cuda_kernel)
        return self._vocoder(latent)
