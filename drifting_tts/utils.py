"""Training utilities shared by the MAE and TTS trainers."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict:
    """Python, NumPy, torch and (current-device) CUDA RNG states, so that a resumed run continues the streams."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


class EMA:
    """Exponential moving average of a module's parameters (buffers are copied)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        torch._foreach_lerp_(list(self.model.parameters()), [p.detach() for p in model.parameters()], 1.0 - self.decay)
        buffers = list(self.model.buffers())
        if buffers:
            torch._foreach_copy_(buffers, list(model.buffers()))


def lr_lambda(warmup: int, total: int, schedule: str = "const", final_frac: float = 0.05):
    """Linear warm-up followed by a constant or cosine schedule (multiplier of the base LR)."""

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "cosine":
            t = min(1.0, (step - warmup) / max(1, total - warmup))
            return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t))
        return 1.0

    return f


def random_crop(
    mel: torch.Tensor, lengths: torch.Tensor, frames: int, generator=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop ``[B, C, T]`` to ``[B, C, frames]`` at random valid starts. Returns crops and starts."""
    B = mel.shape[0]
    max_start = (lengths - frames).clamp_min(0)
    starts = (torch.rand(B, generator=generator) * (max_start + 1).float()).long()
    if mel.shape[-1] < frames:
        mel = nn.functional.pad(mel, (0, frames - mel.shape[-1]))
    idx = starts[:, None] + torch.arange(frames)[None]
    out = torch.gather(mel, 2, idx[:, None, :].expand(-1, mel.shape[1], -1).to(mel.device))
    return out, starts


def save_checkpoint(path: str | Path, **state) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def infinite(loader):
    while True:
        yield from loader
