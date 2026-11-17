"""Audio backends: the BigVGAN mel and pretrained audio-VAE latent spaces behind one encode / decode interface.

==============  ==========  ===========  ============  =====  ==================================================
name            input rate  output rate  frame rate    dim    weights
==============  ==========  ===========  ============  =====  ==================================================
``bigvgan``     24 kHz      24 kHz       93.75 Hz      100    BigVGAN-v2 (``pip install "drifting-tts[bigvgan]"``)
``dacvae``      48 kHz      48 kHz       25 Hz         128    ``facebook/dacvae-watermarked`` (``[dacvae]``)
``voxcpm2``     16 kHz      48 kHz       25 Hz         64     ``openbmb/VoxCPM2`` AudioVAE (``[voxcpm]``)
``voxcpm1.5``   44.1 kHz    44.1 kHz     25 Hz         64     ``openbmb/VoxCPM1.5`` AudioVAE (``[voxcpm]``)
``mimi``        24 kHz      24 kHz       12.5 Hz       32     Kyutai's Pocket TTS Mimi (``[mimi]``, gated weights)
``semantic-vae`` 16 kHz     16 kHz       40 Hz         64     ``zkniu/Semantic-VAE`` (WavLM-aligned latents, #93)
==============  ==========  ===========  ============  =====  ==================================================
"""

from __future__ import annotations

from .base import AudioBackend, LatentStats, resample, stream_decode

BACKENDS = ("bigvgan", "dacvae", "voxcpm2", "voxcpm1.5", "mimi", "semantic-vae")
# the TTS model can be trained on these (`extract-latents`)
VAE_BACKENDS = ("dacvae", "voxcpm2", "voxcpm1.5", "mimi", "semantic-vae")

__all__ = ["BACKENDS", "VAE_BACKENDS", "AudioBackend", "LatentStats", "load_backend", "resample", "stream_decode"]


def load_backend(name: str, device: str = "cuda", **kwargs) -> AudioBackend:
    """Build backend ``name`` (see :data:`BACKENDS`) on ``device``; ``kwargs`` go to its constructor (e.g.
    ``finetuned=`` for ``bigvgan``, ``target_rate=`` for ``voxcpm2``)."""
    if name == "bigvgan":
        from .mel import BigVGANMel

        return BigVGANMel(device, **kwargs)
    if name == "dacvae":
        from .dacvae import DACVAEBackend

        return DACVAEBackend(device, **kwargs)
    if name in ("voxcpm2", "voxcpm1.5"):
        from .voxcpm import VoxCPMBackend

        return VoxCPMBackend(name, device, **kwargs)
    if name == "mimi":
        from .mimi import MimiBackend

        return MimiBackend(device, **kwargs)
    if name == "semantic-vae":
        from .semantic_vae import SemanticVAEBackend

        return SemanticVAEBackend(device, **kwargs)
    raise ValueError(f"unknown audio backend {name!r}; choose from {', '.join(BACKENDS)}")
