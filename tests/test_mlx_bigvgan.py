import numpy as np
import pytest
import torch

from tests.test_vocoder import _cached

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from drifting_tts.mlx.bigvgan import (  # noqa: E402
    _DOWN,
    _F,
    _UP_EVEN,
    _UP_ODD,
    Activation1d,
    BigVGAN,
    ConvTranspose1d,
    convert_bigvgan,
    fused_activation,
    fused_activation_available,
    vocoder_context_frames,
)
from drifting_tts.mlx.vocoder import load_vocoder  # noqa: E402

# tiny BigVGAN-v2-shaped generator (both upsampling shapes of the 24 kHz model, snakebeta with log-scale parameters)
TINY = {"num_mels": 100, "upsample_rates": [4, 2], "upsample_kernel_sizes": [8, 4], "upsample_initial_channel": 16,
        "resblock": "1", "resblock_kernel_sizes": [3, 5], "resblock_dilation_sizes": [[1, 3], [1, 5]],
        "activation": "snakebeta", "snake_logscale": True, "use_tanh_at_final": False, "use_bias_at_final": False}
# tiny BigVGAN-base-shaped generator (v1: 8x and 2x stages, tanh and bias at the end by default)
TINY_BASE = {"num_mels": 100, "upsample_rates": [8, 2], "upsample_kernel_sizes": [16, 4],
             "upsample_initial_channel": 16, "resblock": "1", "resblock_kernel_sizes": [3, 7],
             "resblock_dilation_sizes": [[1, 3], [1, 3]], "activation": "snakebeta", "snake_logscale": True}


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _random_torch_bigvgan(hparams: dict):
    """Random generator: its weight-norm state dict and the inference model (weight norm removed)."""
    if not _cached("bigvgan.py"):
        pytest.skip("BigVGAN-v2 code (HF repo) not cached")
    pytest.importorskip("librosa")
    from drifting_tts.vocoder import build_bigvgan

    torch.manual_seed(0)
    model = build_bigvgan(hparams=hparams, pretrained=False)
    with torch.no_grad():  # non-trivial snake parameters, gains and biases
        for name, p in model.named_parameters():
            p.copy_(torch.rand_like(p) + 0.5 if name.endswith("weight_g") else
                    torch.randn_like(p) * (0.5 if name.endswith(("alpha", "beta")) else 0.1 if "bias" in name else 1))
    sd_wn = {k: v.clone() for k, v in model.state_dict().items()}
    model.remove_weight_norm()
    return sd_wn, model.eval()


@pytest.fixture(scope="module")
def tiny_torch():
    return _random_torch_bigvgan(TINY)


@pytest.mark.parametrize("frames", [7, 12])
def test_mlx_bigvgan_matches_torch(tiny_torch, tmp_path, frames):
    sd_wn, model = tiny_torch
    weights = convert_bigvgan(model.state_dict())
    folded = convert_bigvgan(sd_wn)
    assert folded.keys() == weights.keys()
    assert all(np.allclose(folded[k], weights[k], rtol=1e-5, atol=1e-6) for k in weights)
    path = str(tmp_path / "bigvgan.safetensors")
    mx.save_safetensors(path, {k: mx.array(v) for k, v in weights.items()})
    mlx_model = load_vocoder(path, TINY)  # strict: every parameter is converted, nothing is left over

    mel = torch.randn(2, 100, frames, generator=torch.Generator().manual_seed(frames)) * 2 - 5
    raw = []  # the reference clamps its output: compare before the clamp
    hook = model.conv_post.register_forward_hook(lambda m, i, o: raw.append(o[:, 0]))
    with torch.no_grad():
        model(mel)
    hook.remove()
    ref = raw[0].numpy()
    out = np.array(mlx_model(mx.array(mel.numpy())))
    assert out.shape == ref.shape == (2, frames * 8)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-4


def test_mlx_activation_matches_torch_at_short_lengths(tiny_torch):
    """The replicate padding of both resamplers at lengths down to a single frame."""
    act = tiny_torch[1].activation_post
    mlx_act = Activation1d(act.act.alpha.numel())
    mlx_act.act.update({k: mx.array(v.detach().numpy()) for k, v in act.act.named_parameters()})
    for t in (1, 2, 5, 16, 33):
        x = torch.randn(2, act.act.alpha.numel(), t) * 2
        with torch.no_grad():
            ref = act(x).numpy().transpose(0, 2, 1)
        out = np.array(mlx_act(mx.array(x.numpy().transpose(0, 2, 1))))
        assert out.shape == ref.shape and np.abs(out - ref).max() < 1e-5


@pytest.mark.parametrize("kernel,stride,padding", [(8, 4, 2), (4, 2, 1), (16, 8, 4), (3, 2, 0), (5, 3, 1), (7, 2, 3)])
def test_polyphase_conv_transpose(kernel, stride, padding):
    torch.manual_seed(0)
    conv = torch.nn.ConvTranspose1d(6, 5, kernel, stride, padding)
    x = torch.randn(2, 6, 9)
    ref = conv(x).detach().numpy().transpose(0, 2, 1)
    m = ConvTranspose1d(6, 5, kernel, stride, padding)
    m.update({"weight": mx.array(conv.weight.detach().numpy().transpose(1, 2, 0)),
              "bias": mx.array(conv.bias.detach().numpy())})
    out = np.array(m(mx.array(x.numpy().transpose(0, 2, 1))))
    assert out.shape == ref.shape and np.abs(out - ref).max() < 1e-5


def test_snake_without_beta():
    h = {**TINY, "activation": "snake", "snake_logscale": False, "use_tanh_at_final": True}
    model = BigVGAN(h)
    assert not any(k.endswith("beta") for k, _ in tree_flatten(model.parameters()))
    out = model(mx.random.normal((1, 100, 5)))
    assert out.shape == (1, 40) and float(mx.abs(out).max()) <= 1


def test_vocoder_context_frames():
    production = {**TINY, "upsample_rates": [4, 4, 2, 2, 2, 2],
                  "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
                  "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5]] * 3}
    assert vocoder_context_frames(production) == 38
    assert vocoder_context_frames(TINY) == 19


def test_chunk_context_matches_full_vocoder():
    mx.random.seed(10)
    model = BigVGAN(TINY).prepare_for_inference()
    mel = mx.random.normal((1, 100, 79)) * 2 - 5
    full = np.array(model(mel))
    context, hop = model.context_frames, model.hop_length
    # Start/end boundaries retain the real model padding; interior chunks need both sides.
    for start, end in ((0, 1), (1, 8), (30, 31), (30, 53), (76, 79)):
        left, right = max(0, start - context), min(mel.shape[-1], end + context)
        decoded = np.array(model(mel[..., left:right]))
        chunk = decoded[:, (start - left) * hop:(end - left) * hop]
        np.testing.assert_allclose(chunk, full[:, start * hop:end * hop], atol=1e-5, rtol=1e-5)


def test_inference_constants_preserve_weights_and_refresh_after_update():
    mx.random.seed(11)
    model = BigVGAN(TINY)
    mel = mx.random.normal((1, 100, 7))
    names = [name for name, _ in tree_flatten(model.parameters())]
    before = np.array(model(mel))
    model.prepare_for_inference()
    assert [name for name, _ in tree_flatten(model.parameters())] == names
    np.testing.assert_array_equal(np.array(model(mel)), before)

    # MLX update/load_weights replace tensors directly: prepared transforms must not stay stale.
    up, snake = model.ups[0][0], model.activation_post.act
    up.update({"weight": up.weight * 0.5})
    snake.update({"alpha": snake.alpha + 0.5, "beta": snake.beta - 0.5})
    changed = np.array(model(mel))
    assert not np.allclose(changed, before)
    model.prepare_for_inference()
    np.testing.assert_array_equal(np.array(model(mel)), changed)
    assert [name for name, _ in tree_flatten(model.parameters())] == names


def test_mlx_bigvgan_base_shape_matches_torch(tmp_path):
    """BigVGAN-base (v1) layout: 8x / 2x upsampling (polyphase 16-tap kernels), tanh and bias at the end."""
    # the base config.json has neither key (both default to true); build_bigvgan starts from the v2 config
    _, model = _random_torch_bigvgan({**TINY_BASE, "use_tanh_at_final": True, "use_bias_at_final": True})
    path = str(tmp_path / "base.safetensors")
    mx.save_safetensors(path, {k: mx.array(v) for k, v in convert_bigvgan(model.state_dict()).items()})
    mlx_model = load_vocoder(path, TINY_BASE)
    mel = torch.randn(1, 100, 11, generator=torch.Generator().manual_seed(3)) * 2 - 5
    with torch.no_grad():
        ref = model(mel)[:, 0].numpy()
    out = np.array(mlx_model(mx.array(mel.numpy())))
    assert out.shape == ref.shape == (1, 11 * 16) and np.abs(out - ref).max() < 5e-5
    assert mlx_model.context_frames == vocoder_context_frames(TINY_BASE)
    real_base = {**TINY_BASE, "upsample_rates": [8, 8, 2, 2], "upsample_kernel_sizes": [16, 16, 4, 4],
                 "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5]] * 3}
    assert vocoder_context_frames(real_base) == 18


def _fused_reference(x: np.ndarray, alpha: np.ndarray, inv_beta: np.ndarray) -> np.ndarray:
    """The Metal kernel's arithmetic in NumPy: output t sums snake(u[clamp(2t + k - 5)]) over the 12 taps."""
    _, n, _ = x.shape
    t = np.arange(n)
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(12):
        m = np.clip(2 * t + k - 5, 0, 2 * n - 1)
        j, odd = m >> 1, (m & 1).astype(bool)
        u = np.zeros_like(out)
        for r in range(6):
            u[:, odd] += _UP_ODD[r] * x[:, np.clip(j[odd] + r - 2, 0, n - 1)]
            u[:, ~odd] += _UP_EVEN[r] * x[:, np.clip(j[~odd] + r - 3, 0, n - 1)]
        out += _F[k] * (u + inv_beta * np.sin(u * alpha) ** 2)
    return out


@pytest.mark.parametrize("t", [1, 2, 3, 7, 40])
def test_fused_activation_arithmetic_matches_polyphase(t):
    """The single-kernel formulation (checked here in NumPy) equals the polyphase activation at every length."""
    rng = np.random.default_rng(t)
    act = Activation1d(5)
    act.act.update({"alpha": mx.array(rng.normal(size=5) * 0.5, dtype=mx.float32),
                    "beta": mx.array(rng.normal(size=5) * 0.5, dtype=mx.float32)})
    x = rng.normal(size=(2, t, 5)).astype(np.float32) * 2
    alpha, inv_beta = (np.array(v) for v in act.act.coefficients())
    np.testing.assert_allclose(_fused_reference(x, alpha, inv_beta), np.array(act(mx.array(x))), atol=1e-5)
    assert _DOWN == [float(v) for v in _F[0::2]] + [float(v) for v in _F[1::2]]


def test_fused_activations_flag_is_a_no_op_without_metal():
    model = BigVGAN(TINY).prepare_for_inference()
    mel = mx.random.normal((1, 100, 6), key=mx.random.key(4))
    before = np.array(model(mel))
    model.use_fused_activations()
    assert all(m.fused for m in model.modules() if isinstance(m, Activation1d))
    if not fused_activation_available():
        np.testing.assert_array_equal(np.array(model(mel)), before)


@pytest.mark.skipif(not fused_activation_available(), reason="needs a Metal GPU (Apple silicon)")
@pytest.mark.parametrize("t", [1, 2, 7, 300])
def test_fused_activation_kernel_matches_polyphase(t):
    rng = np.random.default_rng(t)
    act = Activation1d(48)
    act.act.update({"alpha": mx.array(rng.normal(size=48) * 0.5, dtype=mx.float32),
                    "beta": mx.array(rng.normal(size=48) * 0.5, dtype=mx.float32)})
    x = mx.array(rng.normal(size=(2, t, 48)).astype(np.float32) * 2)
    ref = np.array(act(x))
    np.testing.assert_allclose(np.array(fused_activation(x, *act.act.coefficients())), ref, atol=1e-5)
    mel = mx.random.normal((1, 100, 9), key=mx.random.key(5))
    model = BigVGAN(TINY).prepare_for_inference()
    plain = np.array(model(mel))
    np.testing.assert_allclose(np.array(model.use_fused_activations()(mel)), plain, atol=1e-4 * np.abs(plain).max())
