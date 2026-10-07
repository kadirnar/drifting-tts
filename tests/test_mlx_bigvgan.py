import numpy as np
import pytest
import torch

from tests.test_vocoder import _cached

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from drifting_tts.mlx.bigvgan import (  # noqa: E402
    Activation1d,
    BigVGAN,
    ConvTranspose1d,
    convert_bigvgan,
    load_bigvgan,
)

# tiny BigVGAN-v2-shaped generator (both upsampling shapes of the 24 kHz model, snakebeta with log-scale parameters)
TINY = {"num_mels": 100, "upsample_rates": [4, 2], "upsample_kernel_sizes": [8, 4], "upsample_initial_channel": 16,
        "resblock": "1", "resblock_kernel_sizes": [3, 5], "resblock_dilation_sizes": [[1, 3], [1, 5]],
        "activation": "snakebeta", "snake_logscale": True, "use_tanh_at_final": False, "use_bias_at_final": False}


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


@pytest.fixture(scope="module")
def tiny_torch():
    """Random tiny generator: its weight-norm state dict and the inference model (weight norm removed)."""
    if not _cached("bigvgan.py"):
        pytest.skip("BigVGAN-v2 code (HF repo) not cached")
    pytest.importorskip("librosa")
    from drifting_tts.vocoder import build_bigvgan

    torch.manual_seed(0)
    model = build_bigvgan(hparams=TINY, pretrained=False)
    with torch.no_grad():  # non-trivial snake parameters, gains and biases
        for name, p in model.named_parameters():
            p.copy_(torch.rand_like(p) + 0.5 if name.endswith("weight_g") else
                    torch.randn_like(p) * (0.5 if name.endswith(("alpha", "beta")) else 0.1 if "bias" in name else 1))
    sd_wn = {k: v.clone() for k, v in model.state_dict().items()}
    model.remove_weight_norm()
    return sd_wn, model.eval()


@pytest.mark.parametrize("frames", [7, 12])
def test_mlx_bigvgan_matches_torch(tiny_torch, tmp_path, frames):
    sd_wn, model = tiny_torch
    weights = convert_bigvgan(model.state_dict())
    folded = convert_bigvgan(sd_wn)
    assert folded.keys() == weights.keys()
    assert all(np.allclose(folded[k], weights[k], rtol=1e-5, atol=1e-6) for k in weights)
    path = str(tmp_path / "bigvgan.safetensors")
    mx.save_safetensors(path, {k: mx.array(v) for k, v in weights.items()})
    mlx_model = load_bigvgan(path, TINY)  # strict: every parameter is converted, nothing is left over

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
