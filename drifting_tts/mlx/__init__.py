"""MLX inference (Apple silicon): ``from drifting_tts.mlx import Synthesizer``. Needs only ``mlx``, ``numpy`` and
``huggingface_hub``; converting the PyTorch checkpoints (:mod:`drifting_tts.mlx.convert`) also needs torch."""

from .synthesize import Synthesizer, write_wav

__all__ = ["Synthesizer", "write_wav"]
