"""Streaming the acoustic model: the DiT on frame windows of each sentence (opt-in; docs/LATENCY.md).

Without it, :meth:`Synthesizer.stream` generates a sentence's whole mel before its first audio piece, so time to first
audio grows with the sentence (and, in a batch, with the longest first sentence). Here the text encoder and the
prosody predictor still run once over the sentence (durations, token pitch and the aligned condition are known up
front), and the noise ``z`` and the style codes are drawn once, in the single-request order. The DiT then runs on
windows of the aligned condition and of that noise:

* the first window covers frames ``[0, head + right)`` and commits ``[0, head)``: ``head`` is what the vocoder's
  first piece needs (``first`` frames plus its ``context``), ``right`` is the DiT's lookahead;
* window ``k`` commits the next ``chunk`` frames ``[c, c + chunk)``, seeing ``left`` frames before and ``right`` after;
  its first ``crossfade`` frames are blended (in log-mel space, linearly) with the previous window's lookahead there;
* a window that reaches the sentence's end commits it all, so a sentence of at most ``head + right`` frames is one
  window: the whole-sentence generation, bit for bit.

The DiT was trained on 256-frame crops (``drift.crop_frames``) with the condition cropped alike, so windows of up to
256 frames are what it saw in training. With ``chunk`` equal to the streaming vocoder's chunk, every vocoder window
(:func:`drifting_tts.batched.stream_windows`) ends where a DiT window's committed frames end.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Chunking:
    """DiT windows of a sentence (frames, 93.75 per second): see the module docstring."""

    right: int = 64
    left: int = 64
    chunk: int = 256
    crossfade: int = 32

    def __post_init__(self):
        if self.crossfade > self.right:
            raise ValueError("the crossfade needs the previous window's lookahead: crossfade <= right")
        if min(self.right, self.left, self.crossfade) < 0 or self.chunk <= 0:
            raise ValueError("right, left and crossfade must be >= 0 and chunk > 0")


def dit_windows(t: int, head: int, chunk: int, left: int, right: int) -> list[tuple[int, int, int, int]]:
    """The DiT windows of a sentence of ``t`` frames: ``(a, b, c0, c1)``, run frames ``[a, b)`` and commit ``[c0,
    c1)``. The first window commits ``[0, head)``, every later one ``chunk`` frames; one that reaches ``t`` commits the
    rest."""
    b = min(t, head + right)
    out = [(0, b, 0, t if b == t else head)]
    c = out[0][3]
    while c < t:
        a, b = max(0, c - left), min(t, c + chunk + right)
        out.append((a, b, c, t if b == t else c + chunk))
        c = out[-1][3]
    return out


DTYPES = (None, "fp32", "bf16", "fp16")


def autocast(dtype: str | None, device: str = "cuda"):
    """``torch.autocast`` in ``"bf16"`` / ``"fp16"`` (a no-op for ``None`` / ``"fp32"``)."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    if dtype in (None, "fp32"):
        return contextlib.nullcontext()
    return torch.autocast(device, dtype=torch.bfloat16 if dtype == "bf16" else torch.float16)


def _take(x: Tensor, starts: Tensor, n: int) -> Tensor:
    """``x[b, ..., starts[b]: starts[b] + n]`` for every row (indices past the end repeat the last element)."""
    idx = (starts[:, None] + torch.arange(n, device=x.device)[None]).clamp_max(x.shape[-1] - 1)
    return x.gather(-1, idx.view(idx.shape[0], *[1] * (x.dim() - 2), n).expand(*x.shape[:-1], n))


class WindowedMel:
    """A batch of sentences whose mels are generated window by window (:func:`dit_windows`).

    ``generate(z, cond, spk, alpha, mask, labels)`` runs the DiT on a batch of windows (``[B, n_mels, W]`` noise,
    ``[B, C, W]`` condition, ``[B, W]`` valid-frame mask); ``z`` ``[B, n_mels, T]`` is the sentences' noise (already
    scaled by the temperature), ``cond`` ``[B, C, T]`` their aligned condition, ``lengths`` their frame counts (host
    ints). :meth:`step` runs the next window of some rows; :attr:`mel` ``[B, n_mels, T]`` holds the output and
    :attr:`committed` how many frames of each row are final."""

    def __init__(self, generate: Callable, z: Tensor, cond: Tensor, spk: Tensor, alpha: Tensor, labels: Tensor,
                 lengths: list[int], chunking: Chunking, head: int):
        self.generate, self.chunking = generate, chunking
        self.z, self.cond, self.spk, self.alpha, self.labels = z, cond, spk, alpha, labels
        self.lengths = list(lengths)
        c = chunking
        self.windows = [dit_windows(t, head, c.chunk, c.left, c.right) for t in self.lengths]
        self.k = [0] * len(self.lengths)
        self.committed = [0] * len(self.lengths)
        self.mel = torch.zeros(z.shape[0], z.shape[1], z.shape[-1], device=z.device)
        self.dit_frames = 0  # frames the DiT has run over (padding included): the cost of the windows

    def pending(self, row: int) -> bool:
        return self.k[row] < len(self.windows[row])

    @torch.no_grad()
    def step(self, rows: list[int] | None = None) -> None:
        """Run the next window of every row in ``rows`` (default: all unfinished rows) as one DiT batch."""
        rows = [r for r in (range(len(self.lengths)) if rows is None else rows) if self.pending(r)]
        if not rows:
            return
        win = [self.windows[r][self.k[r]] for r in rows]
        dev = self.z.device
        W = max(b - a for a, b, _, _ in win)
        idx = torch.tensor(rows, device=dev)
        a = torch.tensor([w[0] for w in win], device=dev)
        n = torch.tensor([w[1] - w[0] for w in win], device=dev)
        mask = torch.arange(W, device=dev)[None] < n[:, None]
        z, cond = _take(self.z[idx], a, W), _take(self.cond[idx], a, W)
        if len(rows) > 1 or W != win[0][1] - win[0][0]:  # rows of different lengths: zeros after each one's end
            z, cond = z * mask[:, None], cond * mask[:, None]
        out = self.generate(z, cond, self.spk[idx], self.alpha[idx], mask, self.labels[idx]).float()
        self.dit_frames += W * len(rows)
        # blend weights of the new window: 0 before the frames it commits (they are final), a ramp over the first
        # ``crossfade`` of them (the previous window's lookahead), 1 after; the first window has no previous one
        pos = a[:, None] + torch.arange(W, device=dev)[None]
        c0 = torch.tensor([w[2] for w in win], device=dev)[:, None]
        X = self.chunking.crossfade
        first = torch.tensor([self.k[r] == 0 for r in rows], device=dev)[:, None]
        w = ((pos - c0).float() + 0.5) / X if X else (pos >= c0).float()
        w = torch.where(first, torch.ones_like(w), w.clamp(0, 1) * (pos >= c0)) * mask
        T = self.mel.shape[-1]
        full = torch.nn.functional.pad(self.mel[idx], (0, W))  # no clamped (duplicate) indices in the scatter
        prev = _take(full, a, W)
        new = torch.where(w[:, None] == 1, out, prev * (1 - w[:, None]) + out * w[:, None])
        full.scatter_(2, pos[:, None].expand(-1, full.shape[1], -1), new)
        self.mel[idx] = full[..., :T]
        for r, (_, _, _, c1) in zip(rows, win):
            self.k[r] += 1
            self.committed[r] = c1
