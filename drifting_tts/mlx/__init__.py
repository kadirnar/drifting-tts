"""MLX inference (Apple silicon): ``from drifting_tts.mlx import Synthesizer``. Needs only ``mlx``, ``numpy`` and
``huggingface_hub``; converting the PyTorch checkpoints (:mod:`drifting_tts.mlx.convert`) also needs torch."""

from .synthesize import Synthesizer, write_wav
from .vocoder import VOCODERS, load_vocoder

__all__ = ["VOCODERS", "Synthesizer", "load_vocoder", "write_wav"]
