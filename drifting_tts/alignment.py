"""Monotonic alignment search (Glow-TTS / Grad-TTS) and duration helpers."""

from __future__ import annotations

import math

import numba
import numpy as np
import torch
from torch import Tensor

# numba threads for the batched MAS DP. A few threads already make it ~1 ms per batch; more threads mostly add
# contention, and in CPU-quota-limited containers (cgroup ``cpu.max``) they stall the whole training process.
MAS_THREADS = 4


def sequence_mask(lengths: Tensor, max_len: int | None = None) -> Tensor:
    max_len = int(lengths.max()) if max_len is None else max_len
    return torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]


@numba.njit(cache=True, boundscheck=False)
def _maximum_path_single(value: np.ndarray, t_x: int, t_y: int, out: np.ndarray) -> None:
    """Glow-TTS ``maximum_path_each`` on a ``[T_mel, N_text]`` layout: the inner loop over tokens is unit-stride."""
    neg_inf = -1e9
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = neg_inf if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else neg_inf
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        out[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


@numba.njit(parallel=True, cache=True)
def _maximum_path_batch(values: np.ndarray, t_xs: np.ndarray, t_ys: np.ndarray, out: np.ndarray) -> None:
    for b in numba.prange(values.shape[0]):
        _maximum_path_single(values[b], t_xs[b], t_ys[b], out[b])


def _to_host(x: Tensor) -> Tensor:
    """Start a device-to-host copy into page-locked memory (the caller synchronises before reading)."""
    if x.device.type != "cuda":
        return x
    return torch.empty(x.shape, dtype=x.dtype, pin_memory=True).copy_(x, non_blocking=True)


@torch.no_grad()
def maximum_path(log_prior: Tensor, mask: Tensor) -> Tensor:
    """Most likely monotonic alignment.

    The DP runs on the CPU (numba) over a transposed ``[T_mel, N_text]`` copy; on CUDA both copies go
    through pinned buffers, so the only host sync is waiting for the scores.

    Args:
        log_prior: ``[B, N_text, T_mel]`` log-likelihood of every frame under every token.
        mask: ``[B, N_text, T_mel]`` valid (token, frame) pairs.
    Returns:
        hard alignment ``[B, N_text, T_mel]`` (each frame assigned to exactly one token).
    """
    device, dtype = log_prior.device, log_prior.dtype
    values = (log_prior * mask).transpose(1, 2).float().contiguous()  # [B, T, N]
    lens = torch.stack([mask[:, :, 0].sum(1), mask[:, 0].sum(1)]).int()  # text / mel lengths
    values, lens = _to_host(values), _to_host(lens)
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    out = torch.zeros(values.shape, dtype=torch.int8, pin_memory=device.type == "cuda")
    threads = numba.get_num_threads()
    numba.set_num_threads(max(1, min(MAS_THREADS, threads, values.shape[0])))
    try:
        _maximum_path_batch(values.numpy(), lens[0].numpy(), lens[1].numpy(), out.numpy())
    finally:
        numba.set_num_threads(threads)
    return out.to(device, non_blocking=True).transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)


def gaussian_log_prior(mu: Tensor, y: Tensor) -> Tensor:
    """``log N(y_t; mu_n, I)`` for all token / frame pairs: ``[B, C, N] x [B, C, T] -> [B, N, T]``."""
    C = mu.shape[1]
    y_sq = -0.5 * (y**2).sum(1, keepdim=True)  # [B, 1, T]
    mu_sq = -0.5 * (mu**2).sum(1)[..., None]  # [B, N, 1]
    cross = torch.einsum("bcn,bct->bnt", mu, y)
    return y_sq + mu_sq + cross - 0.5 * C * math.log(2 * math.pi)


def generate_path(durations: Tensor, mask: Tensor) -> Tensor:
    """Hard alignment from integer durations ``[B, N]`` and a ``[B, N, T]`` mask."""
    B, N, T = mask.shape
    cum = torch.cumsum(durations, 1)
    path = sequence_mask(cum.flatten(), T).view(B, N, T).to(mask.dtype)
    path = path - torch.nn.functional.pad(path, (0, 0, 1, 0))[:, :-1]
    return path * mask
