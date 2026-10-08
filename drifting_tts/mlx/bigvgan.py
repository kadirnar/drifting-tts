"""BigVGAN generators (NVIDIA, MIT; v2 and the v1 base) in MLX for inference: weight norm removed, channels-last
internally."""

from __future__ import annotations

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> np.ndarray:
    """``alias_free_activation``'s ``kaiser_sinc_filter1d`` (float64 maths, float32 result)."""
    half = kernel_size // 2
    a = 2.285 * (half - 1) * math.pi * 4 * half_width + 7.95
    beta = 0.1102 * (a - 8.7) if a > 50 else 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21) if a >= 21 else 0.0
    t = np.arange(-half, half) + 0.5 if kernel_size % 2 == 0 else np.arange(kernel_size) - half
    f = 2 * cutoff * np.kaiser(kernel_size, beta) * np.sinc(2 * cutoff * t)
    return (f / f.sum()).astype(np.float32)


# Activation1d's fixed setup: ratio 2, 12 taps, the same low-pass (cutoff 0.25, half width 0.3) up and down.
_F = kaiser_sinc_filter(0.25, 0.3, 12)
# UpSample1d (replicate pad 5, transposed conv, crop 15 / 15, times 2): the even / odd output phases are 6-tap filters
# over x edge-padded by 3, starting at offsets 0 / 1
_UP_EVEN, _UP_ODD = [2 * float(v) for v in _F[11::-2]], [2 * float(v) for v in _F[10::-2]]
# LowPassFilter1d (replicate pad 5 / 6, stride 2): 6-tap filters over the even / odd samples of its padded input
_DOWN = [float(v) for v in _F[0::2]] + [float(v) for v in _F[1::2]]


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def vocoder_context_frames(hparams: dict) -> int:
    """Mel context on each side needed to decode a chunk without artificial boundary padding.

    Trace one complete output hop backwards through the network. Each anti-aliased activation has
    radius five; residual branches contribute the largest summed radius. Transposed convolutions
    map an output interval to input indices with integer ceil/floor, preserving all output phases.
    The 24 kHz / 256x BigVGAN-v2 needs 38 mel frames on either side, BigVGAN-base 18.
    """
    rates, kernels = hparams["upsample_rates"], hparams["upsample_kernel_sizes"]
    if len(rates) != len(kernels) or any((k - u) % 2 for u, k in zip(rates, kernels)):
        raise ValueError("chunked BigVGAN requires matched upsampling stages with an exact integer hop")
    if str(hparams.get("resblock", "1")) != "1":
        raise NotImplementedError("only AMPBlock1 (resblock '1') is supported")
    block_radius = max(
        sum(10 + get_padding(k, d) + get_padding(k) for d in dilations)
        for k, dilations in zip(hparams["resblock_kernel_sizes"], hparams["resblock_dilation_sizes"])
    )
    # conv_post (radius 3) and activation_post (radius 5), around a complete output hop.
    lo, hi = -8, math.prod(rates) - 1 + 8
    for stride, kernel in reversed(list(zip(rates, kernels))):
        lo, hi = lo - block_radius, hi + block_radius
        padding = (kernel - stride) // 2
        lo = -(-(lo + padding - kernel + 1) // stride)
        hi = (hi + padding) // stride
    # conv_pre has radius three in mel frames.
    return max(3 - lo, hi + 3)


def _snake(x: mx.array, alpha: mx.array, inv_beta: mx.array) -> mx.array:
    return x + inv_beta * mx.square(mx.sin(x * alpha))


@partial(mx.compile, shapeless=True)
def _upsample_snake(alpha: mx.array, inv_beta: mx.array, *x: mx.array) -> tuple[mx.array, mx.array]:
    """Snake of the even / odd phases of the 2x upsampled signal; ``x``: the padded input at the 7 shifts."""
    even = sum(w * v for w, v in zip(_UP_EVEN, x[:6]))
    odd = sum(w * v for w, v in zip(_UP_ODD, x[1:]))
    return _snake(even, alpha, inv_beta), _snake(odd, alpha, inv_beta)


@partial(mx.compile, shapeless=True)
def _downsample(*z: mx.array) -> mx.array:
    """``z``: the padded even phase at 6 shifts, then the padded odd phase at 6 shifts."""
    return sum(w * v for w, v in zip(_DOWN, z))


def _floats(name: str, values) -> str:
    return f"constant float {name}[{len(values)}] = {{{', '.join(f'{float(v):.9g}f' for v in values)}}};"


# The same activation as one Metal kernel (opt-in, needs validation on a Mac): output t of channel c is
# sum_k F[k] * snake(u[clamp(2t + k - 5)]) over the 2x upsampled signal u, whose even / odd samples are 6-tap filters
# of the edge-padded input. Each thread recomputes the 12 upsampled values it needs instead of materialising the 2x
# signal and its padded phases, so the input is read and the output written once.
_ACTIVATION_HEADER = "\n".join([_floats("UP_EVEN", _UP_EVEN), _floats("UP_ODD", _UP_ODD), _floats("DOWN_F", _F)])
_ACTIVATION_SOURCE = """
    uint c = thread_position_in_grid.x, t = thread_position_in_grid.y, b = thread_position_in_grid.z;
    int n = x_shape[1], ch = x_shape[2];
    if ((int)c >= ch || (int)t >= n) return;
    const device T* xb = x + (size_t)b * n * ch + c;
    float a = alpha[c], ib = inv_beta[c], acc = 0.0f;
    for (int k = 0; k < 12; ++k) {
        int m = clamp(2 * (int)t + k - 5, 0, 2 * n - 1), j = m >> 1;
        float u = 0.0f;
        if (m & 1) {
            for (int r = 0; r < 6; ++r) u += UP_ODD[r] * (float)xb[(size_t)clamp(j + r - 2, 0, n - 1) * ch];
        } else {
            for (int r = 0; r < 6; ++r) u += UP_EVEN[r] * (float)xb[(size_t)clamp(j + r - 3, 0, n - 1) * ch];
        }
        float s = metal::precise::sin(u * a);
        acc += DOWN_F[k] * (u + ib * s * s);
    }
    out[((size_t)b * n + t) * ch + c] = (T)acc;
"""
_activation_kernel = None


def fused_activation_available() -> bool:
    """Whether :func:`fused_activation` can run: a Metal GPU is the default device."""
    return mx.default_device().type == mx.gpu and mx.metal.is_available()


def fused_activation(x: mx.array, alpha: mx.array, inv_beta: mx.array) -> mx.array:
    """:class:`Activation1d` on ``[B, T, C]`` as a single Metal kernel (see ``_ACTIVATION_SOURCE``)."""
    global _activation_kernel
    if _activation_kernel is None:
        _activation_kernel = mx.fast.metal_kernel(name="drifting_aa_snake", input_names=["x", "alpha", "inv_beta"],
                                                  output_names=["out"], source=_ACTIVATION_SOURCE,
                                                  header=_ACTIVATION_HEADER)
    b, t, c = x.shape
    return _activation_kernel(inputs=[x, alpha.astype(mx.float32), inv_beta.astype(mx.float32)],
                              template=[("T", x.dtype)], grid=(c, t, b), threadgroup=(min(c, 32), min(t, 8), 1),
                              output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


class SnakeBeta(nn.Module):
    """``x + 1 / (beta + 1e-9) * sin(alpha * x) ** 2`` per channel (``beta = alpha``: Snake)."""

    def __init__(self, channels: int, logscale: bool = True, beta: bool = True):
        super().__init__()
        self.logscale = logscale
        self.alpha = mx.zeros((channels,)) if logscale else mx.ones((channels,))
        if beta:
            self.beta = mx.zeros((channels,)) if logscale else mx.ones((channels,))

    def coefficients(self) -> tuple[mx.array, mx.array]:
        """``alpha`` and ``1 / (beta + 1e-9)``."""
        alpha, beta = self.alpha, self.get("beta", self.alpha)
        cached = self.get("_inference_coefficients")
        if cached is not None and cached[0] is alpha and cached[1] is beta:
            return cached[2], cached[3]
        if self.logscale:
            alpha, beta = mx.exp(alpha), mx.exp(beta)
        return alpha, 1.0 / (beta + 1e-9)

    def __call__(self, x: mx.array) -> mx.array:
        return _snake(x, *self.coefficients())


def _edge(x: mx.array, n: int) -> mx.array:
    return mx.broadcast_to(x, (x.shape[0], n, x.shape[2]))


class Activation1d(nn.Module):
    """Anti-aliased snake (2x upsample, snake, 2x low-pass downsample) in polyphase form: the 2x signal is kept as
    its even / odd phases and both filters run as fused elementwise kernels over shifted views. ``[B, T, C]``.
    ``fused``: one Metal kernel instead (:func:`fused_activation`) when a Metal GPU is the default device."""

    def __init__(self, channels: int, logscale: bool = True, beta: bool = True):
        super().__init__()
        self.act = SnakeBeta(channels, logscale, beta)
        self.fused = False

    def __call__(self, x: mx.array) -> mx.array:
        if self.fused and fused_activation_available():
            return fused_activation(x, *self.act.coefficients())
        t = x.shape[1]
        x = mx.pad(x, [(0, 0), (3, 3), (0, 0)], mode="edge")
        even, odd = _upsample_snake(*self.act.coefficients(), *(x[:, r: r + t] for r in range(7)))
        # the phases of the upsampled signal replicate-padded by 5 / 6, as LowPassFilter1d pads it
        first, last = even[:, :1], odd[:, -1:]
        pad_even = mx.concatenate([_edge(first, 3), odd, _edge(last, 2)], axis=1)
        pad_odd = mx.concatenate([_edge(first, 2), even, _edge(last, 3)], axis=1)
        return _downsample(*(pad_even[:, i: i + t] for i in range(6)), *(pad_odd[:, i: i + t] for i in range(6)))


class ConvTranspose1d(nn.Module):
    """``torch.nn.ConvTranspose1d`` (weight ``[out, kernel, in]``, as ``mx.conv_transpose1d``) computed as one stride-1
    convolution with ``stride`` phases of output channels: no zero insertion."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(-scale, scale, (out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))
        self.stride, self.padding = stride, padding
        # output t = stride * q + r takes input q - s through kernel tap stride * s + r + padding
        lo = min(-((r + padding) // stride) for r in range(stride))
        hi = max((kernel_size - 1 - r - padding) // stride for r in range(stride))
        self.taps, self.shift, self.offset = hi - lo + 1, hi, -stride * lo - padding

    def _polyphase(self) -> mx.array:
        """``[stride * out, taps, in]`` weight of the equivalent stride-1 convolution."""
        cached = self.get("_inference_weight")
        if cached is not None and cached[0] is self.weight:
            return cached[1]
        o, k, i = self.weight.shape
        u, n = self.stride, self.taps
        w = mx.pad(self.weight, [(0, 0), (self.offset, u * n - k - self.offset), (0, 0)])
        return w.reshape(o, n, u, i)[:, ::-1].transpose(2, 0, 1, 3).reshape(u * o, n, i)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        u, (o, k, _) = self.stride, self.weight.shape
        length = (t - 1) * u - 2 * self.padding + k
        q = -(-length // u)
        left, right = self.shift, q + self.taps - 1 - t - self.shift
        w = self._polyphase()
        if left == right:
            y = mx.conv1d(x, w, padding=left)
        else:
            y = mx.conv1d(mx.pad(x, [(0, 0), (left, max(right, 0)), (0, 0)])[:, : q + self.taps - 1], w)
        return y.reshape(b, q * u, o)[:, :length] + self.bias


class AMPBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int], logscale: bool, beta: bool):
        super().__init__()
        conv = lambda d: nn.Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, d), dilation=d)
        self.convs1 = [conv(d) for d in dilation]
        self.convs2 = [conv(1) for _ in dilation]
        self.activations = [Activation1d(channels, logscale, beta) for _ in range(2 * len(dilation))]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]):
            x = x + c2(a2(c1(a1(x))))
        return x


class BigVGAN(nn.Module):
    """BigVGAN (v2 or v1) generator from its ``config.json`` hparams. Parameters are named as in the PyTorch state dict
    without the anti-aliasing filter buffers; ``use_tanh_at_final=false`` returns the unclamped waveform (the
    reference clamps it to ``[-1, 1]``)."""

    def __init__(self, h: dict):
        super().__init__()
        if str(h.get("resblock", "1")) != "1":
            raise NotImplementedError("only AMPBlock1 (resblock '1') is supported")
        if h["activation"] not in ("snake", "snakebeta"):
            raise ValueError(f"unknown activation {h['activation']!r}")
        self.hop_length = math.prod(h["upsample_rates"])
        self.context_frames = vocoder_context_frames(h)
        logscale, beta = h["snake_logscale"], h["activation"] == "snakebeta"
        self.num_kernels = len(h["resblock_kernel_sizes"])
        ch = h["upsample_initial_channel"]
        self.conv_pre = nn.Conv1d(h["num_mels"], ch, 7, padding=3)
        self.ups, self.resblocks = [], []
        for u, k in zip(h["upsample_rates"], h["upsample_kernel_sizes"]):
            self.ups.append([ConvTranspose1d(ch, ch // 2, k, u, (k - u) // 2)])
            ch //= 2
            self.resblocks.extend(AMPBlock1(ch, rk, d, logscale, beta)
                                  for rk, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"]))
        self.activation_post = Activation1d(ch, logscale, beta)
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3, bias=h.get("use_bias_at_final", True))
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def prepare_for_inference(self) -> BigVGAN:
        """Materialize constant inference transforms without adding checkpoint parameters.

        Cached values are private, and source identity checks invalidate them after weight replacement.
        Call again after loading new weights to regain the optimization.
        """
        constants = []
        for module in self.modules():
            if isinstance(module, ConvTranspose1d):
                weight = module._polyphase()
                module._inference_weight = (module.weight, weight)
                constants.append(weight)
            elif isinstance(module, SnakeBeta):
                alpha, inv_beta = module.coefficients()
                module._inference_coefficients = (module.alpha, module.get("beta", module.alpha), alpha, inv_beta)
                constants.extend((alpha, inv_beta))
        mx.eval(constants)
        return self

    def use_fused_activations(self, enabled: bool = True) -> BigVGAN:
        """Run each anti-aliased activation as one Metal kernel (:func:`fused_activation`) when a Metal GPU is the
        default device; elsewhere this has no effect."""
        for module in self.modules():
            if isinstance(module, Activation1d):
                module.fused = enabled
        return self

    def __call__(self, mel: mx.array) -> mx.array:
        """Log-mel ``[B, num_mels, T]`` -> waveform ``[B, T * prod(upsample_rates)]``."""
        x = self.conv_pre(mel.transpose(0, 2, 1).astype(self.conv_pre.weight.dtype))
        n = self.num_kernels
        for i, (up,) in enumerate(self.ups):
            x = up(x)
            x = sum(block(x) for block in self.resblocks[i * n: (i + 1) * n]) / n
        x = self.conv_post(self.activation_post(x))[..., 0]
        return mx.tanh(x) if self.use_tanh_at_final else x


def to_numpy(v) -> np.ndarray:
    """A torch tensor or array-like as float32 NumPy."""
    return v.detach().cpu().float().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=np.float32)


def convert_bigvgan(state_dict: dict) -> dict[str, np.ndarray]:
    """PyTorch generator state dict (tensors or arrays) -> float32 MLX weights keyed as ``BigVGAN``'s parameters.
    Old-style weight norm (``weight_g`` / ``weight_v``) is folded; the anti-aliasing filters are checked against the
    recomputed one and dropped."""
    sd = {k: to_numpy(v) for k, v in state_dict.items()}
    for k in [k for k in sd if k.endswith(".weight_v")]:
        v, g = sd.pop(k).astype(np.float64), sd.pop(k[:-1] + "g")
        sd[k[:-2]] = g * v / np.sqrt((v.reshape(len(v), -1) ** 2).sum(1)).reshape(g.shape)
    out = {}
    for k, v in sd.items():
        if k.endswith(".filter"):
            if v.size != _F.size or np.abs(v.ravel() - _F).max() > 1e-6:
                raise ValueError(f"{k} is not BigVGAN's anti-aliasing filter")
            continue
        if v.ndim == 3:  # ConvTranspose1d [in, out, k] / Conv1d [out, in, k] -> [out, k, in]
            v = v.transpose(1, 2, 0) if k.startswith("ups.") else v.transpose(0, 2, 1)
        out[k] = np.ascontiguousarray(v, dtype=np.float32)
    return out
