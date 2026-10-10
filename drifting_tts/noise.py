"""Counter-based noise for batched serving: every request's noise from ``(seed, sentence, stream, element)``.

With ``torch.Generator`` each request draws its noise call by call (four calls per sentence), so a batch of 256
requests costs a thousand small kernel launches. Here one launch fills a whole batch, and an element's value depends
only on its request's seed, the sentence index, the stream and the element's index: not on the batch, the padding or
the other requests. A window of a sentence's DiT noise can be drawn without the rest of it (the index is
frame-major). The single-request path uses the same function, so a request sounds the same alone or in a batch.

Scheme (Philox4x32-10, the generator of cuRAND and Triton):

* key ``(seed mod 2^32, seed // 2^32)``; counter ``(i // 4, 0, stream, sentence)``; element ``i`` takes output word
  ``i % 4``; words ``(0, 1)`` and ``(2, 3)`` are pairs of uniforms ``(w >> 8 + 0.5) / 2^24`` turned into normals by the
  Box-Muller transform (``sqrt(-2 ln u0) cos / sin (2 pi u1)``); integers in ``[0, n)`` are ``w mod n``.
* streams (:data:`STREAMS`): the prosody predictor's token noise ``[noise_tok, N]`` (element ``token * noise_tok +
  channel``) and global noise, the DiT's noise ``[n_mels, T]`` (element ``frame * n_mels + channel``) and the style
  codes.

:func:`philox_normal` / :func:`philox_randint` run a Triton kernel on CUDA when Triton is installed, else the same
arithmetic in PyTorch (int64; the CPU, CI). They agree up to the float rounding of ``log`` / ``cos`` / ``sin``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

STREAMS = {"prosody_tok": 0, "prosody_glob": 1, "dit": 2, "style": 3}
_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF

try:  # pragma: no cover - needs a GPU
    import triton
    import triton.language as tl
    from triton.language.random import philox as _tl_philox

    @triton.jit
    def _normal_kernel(out_ptr, seed_ptr, sent_ptr, start_ptr, len_ptr, stride_b, stride_c, stride_l, C, L,
                       stream, scale, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        e = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)  # element of this row, channel-major
        c, ll = e // L, e % L
        seed = tl.load(seed_ptr + b)
        i = (tl.load(start_ptr + b) + ll) * C + c  # frame-major index in the sentence
        z = tl.zeros_like(i).to(tl.uint32)
        w0, w1, w2, w3 = _tl_philox(seed, (i // 4).to(tl.uint32), z, z + stream,
                                    z + tl.load(sent_ptr + b).to(tl.uint32))
        lane = i % 4
        a = tl.where(lane < 2, w0, w2)
        bb = tl.where(lane < 2, w1, w3)
        u0 = ((a >> 8).to(tl.float32) + 0.5) * 5.9604644775390625e-08
        u1 = ((bb >> 8).to(tl.float32) + 0.5) * 5.9604644775390625e-08
        r = tl.sqrt(-2.0 * tl.log(u0))
        theta = 6.283185307179586 * u1
        v = tl.where(lane % 2 == 0, r * tl.cos(theta), r * tl.sin(theta)) * scale
        ok = (e < C * L) & (ll < tl.load(len_ptr + b))
        tl.store(out_ptr + b * stride_b + c * stride_c + ll * stride_l, tl.where(ok, v, 0.0), mask=e < C * L)

    @triton.jit
    def _randint_kernel(out_ptr, seed_ptr, sent_ptr, n, high, stream, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        j = tl.arange(0, BLOCK)
        z = tl.zeros_like(j).to(tl.uint32)
        w0, w1, w2, w3 = _tl_philox(tl.load(seed_ptr + b), (j // 4).to(tl.uint32), z, z + stream,
                                    z + tl.load(sent_ptr + b).to(tl.uint32))
        lane = j % 4
        w = tl.where(lane == 0, w0, tl.where(lane == 1, w1, tl.where(lane == 2, w2, w3)))
        tl.store(out_ptr + b * n + j, (w % high.to(tl.uint32)).to(tl.int64), mask=j < n)

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


def _mulhilo(a: Tensor, m: int) -> tuple[Tensor, Tensor]:
    """``(a * m) >> 32`` and ``(a * m) mod 2^32`` for int64 ``a`` in ``[0, 2^32)`` (16-bit halves: no overflow)."""
    p_l, p_h = a * (m & 0xFFFF), a * (m >> 16)
    s = p_l + ((p_h & 0xFFFF) << 16)
    return ((p_h >> 16) + (s >> 32)) & _MASK, s & _MASK


def _philox4_torch(c: list[Tensor], k0: Tensor, k1: Tensor) -> list[Tensor]:
    for _ in range(10):
        hi0, lo0 = _mulhilo(c[0], _M0)
        hi1, lo1 = _mulhilo(c[2], _M1)
        c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0]
        k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
    return c


def _words(seeds: Tensor, sentences: Tensor, stream: int, i: Tensor) -> tuple[Tensor, Tensor]:
    """Philox output words of elements ``i`` ``[B, ...]`` (int64): the word ``i % 4`` and its Box-Muller partner."""
    shape = (-1,) + (1,) * (i.dim() - 1)
    k0, k1 = (seeds & _MASK).view(shape), ((seeds >> 32) & _MASK).view(shape)
    zero = torch.zeros_like(i)
    w = _philox4_torch([i // 4, zero, zero + stream, zero + sentences.view(shape)], k0, k1)
    lane = i % 4
    a = torch.where(lane < 2, w[0], w[2])
    b = torch.where(lane < 2, w[1], w[3])
    return torch.where(lane % 2 == 0, a, b), (a, b)


def philox_normal(seeds: Tensor, sentences: Tensor, stream: str, channels: int, length: int,
                  starts: Tensor | None = None, lengths: Tensor | None = None, scale: float = 1.0,
                  out: Tensor | None = None) -> Tensor:
    """Standard normal noise ``[B, channels, length]`` (times ``scale``): row ``b`` holds elements ``(starts[b] + l) *
    channels + c`` of stream ``stream`` of sentence ``sentences[b]`` of the request with seed ``seeds[b]``; zero at
    ``l >= lengths[b]``. ``seeds`` / ``sentences`` / ``starts`` / ``lengths``: int64 ``[B]`` on the output's device."""
    dev = seeds.device
    B = seeds.shape[0]
    starts = torch.zeros(B, dtype=torch.long, device=dev) if starts is None else starts
    lengths = torch.full((B,), length, dtype=torch.long, device=dev) if lengths is None else lengths
    if out is None:
        out = torch.empty(B, channels, length, device=dev)
    if B == 0 or channels * length == 0:
        return out
    if HAS_TRITON and out.is_cuda:
        block = 1024
        _normal_kernel[(B, triton.cdiv(channels * length, block))](
            out, seeds, sentences, starts, lengths, out.stride(0), out.stride(1), out.stride(2), channels, length,
            STREAMS[stream], float(scale), BLOCK=block)
        return out
    c = torch.arange(channels, device=dev)[None, :, None]
    ll = torch.arange(length, device=dev)[None, None, :]
    i = (starts.view(-1, 1, 1) + ll) * channels + c
    _, (a, b) = _words(seeds, sentences, STREAMS[stream], i)
    u0 = ((a >> 8).float() + 0.5) * 2.0**-24
    u1 = ((b >> 8).float() + 0.5) * 2.0**-24
    r, theta = torch.sqrt(-2.0 * torch.log(u0)), 2 * math.pi * u1
    v = torch.where(i % 2 == 0, r * torch.cos(theta), r * torch.sin(theta)) * scale
    out.copy_(torch.where(ll < lengths.view(-1, 1, 1), v, torch.zeros_like(v)))
    return out


def philox_randint(seeds: Tensor, sentences: Tensor, stream: str, n: int, high: int) -> Tensor:
    """Integers in ``[0, high)`` ``[B, n]``: element ``j`` is word ``j % 4`` of counter ``j // 4`` mod ``high``
    (unbiased when ``high`` divides 2^32, e.g. the 64 style classes)."""
    if HAS_TRITON and seeds.is_cuda:
        out = torch.empty(seeds.shape[0], n, dtype=torch.long, device=seeds.device)
        if seeds.shape[0]:
            _randint_kernel[(seeds.shape[0],)](out, seeds, sentences, n, high, STREAMS[stream],
                                               BLOCK=triton.next_power_of_2(n))
        return out
    j = torch.arange(n, device=seeds.device)[None].expand(seeds.shape[0], n)
    w, _ = _words(seeds, sentences, STREAMS[stream], j)
    return w % high


class Philox:
    """The keys of a batch of requests under this scheme: each row's seed and sentence index (host ints)."""

    def __init__(self, seeds: list[int], sentences: list[int]):
        if len(seeds) != len(sentences):
            raise ValueError("one sentence index per seed")
        self.seeds, self.sentences = [int(s) for s in seeds], [int(k) for k in sentences]

    def __len__(self) -> int:
        return len(self.seeds)

    def subset(self, rows: list[int]) -> Philox:
        return Philox([self.seeds[r] for r in rows], [self.sentences[r] for r in rows])

    def tensors(self, device) -> tuple[Tensor, Tensor]:
        from .chunked import to_device

        both = to_device([self.seeds, self.sentences], device)
        return both[0], both[1]


class DiTNoise:
    """The DiT's noise of a batch of sentences under this scheme, times ``scale`` (the temperature): drawn window by
    window by :class:`~drifting_tts.chunked.WindowedMel` (``shape``: ``(B, n_mels, T)``, as the tensor it replaces)."""

    def __init__(self, keys: Philox, n_mels: int, lengths: Tensor, scale: float, frames: int | None = None):
        self.seeds, self.sentences = keys.tensors(lengths.device)
        self.n_mels, self.lengths, self.scale = n_mels, lengths, float(scale)
        T = frames if frames is not None else (int(lengths.max()) if len(keys) else 0)  # host lengths: no sync
        self.shape = (len(keys), n_mels, T)
        self.device = lengths.device

    def window(self, rows: Tensor, starts: Tensor, width: int, valid: Tensor) -> Tensor:
        """``[R, n_mels, width]``: frames ``starts[r] + l`` of row ``rows[r]``, zero at ``l >= valid[r]``."""
        return philox_normal(self.seeds[rows], self.sentences[rows], "dit", self.n_mels, width, starts, valid,
                             self.scale)

    def full(self) -> Tensor:
        """The whole sentences' noise ``[B, n_mels, T]`` (zero after each row's length)."""
        return philox_normal(self.seeds, self.sentences, "dit", self.n_mels, self.shape[2], lengths=self.lengths,
                             scale=self.scale)
