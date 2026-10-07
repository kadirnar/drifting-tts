"""Pieces shared by the vendored VAE models: the Snake activation and checkpoint loading with weight norm folded."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def snake(x: Tensor, alpha: Tensor) -> Tensor:
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return snake(x, self.alpha)


def fold_weight_norm(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Replace every ``weight_g`` / ``weight_v`` pair (``torch.nn.utils.weight_norm``, ``dim=0``) by ``weight``."""
    out = {}
    for k, v in state.items():
        leaf = k.rsplit(".", 1)[-1]
        if leaf == "weight_g":
            continue
        if leaf == "weight_v":
            g = state[k[:-1] + "g"]
            k, v = k[:-2], v * (g / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.ndim - 1)))
        out[k] = v
    return out


def load_vae_checkpoint(path: str) -> tuple[dict[str, Tensor], dict]:
    """A ``{"state_dict", "metadata": {"kwargs"}}`` checkpoint, loaded as weights only -> folded state, kwargs."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return fold_weight_norm(ck["state_dict"]), ck.get("metadata", {}).get("kwargs", {})
