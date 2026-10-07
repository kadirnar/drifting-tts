import torch

from drifting_tts.models.generator import DriftDiT
from drifting_tts.utils import count_params


def _model(**kw):
    args = dict(n_mels=16, cond_channels=12, hidden=64, depth=2, heads=4, patch=2, n_registers=4, num_speakers=3,
                noise_classes=8, noise_coords=4)
    args.update(kw)
    return DriftDiT(**args)


def test_shapes_odd_length_and_zero_init():
    g = _model()
    z, cond = torch.randn(3, 16, 37), torch.randn(3, 12, 37)
    out = g(z, cond, torch.tensor([0, 1, 2]), torch.tensor([1.0, 2.0, 3.0]))
    assert out.shape == (3, 16, 37)
    assert out.abs().max() == 0  # adaLN-zero / zero final layer: identity at init


def test_noise_and_condition_dependence_and_mask():
    torch.manual_seed(0)
    g = _model()
    for p in g.final.parameters():
        torch.nn.init.normal_(p, std=0.1)
    for b in g.blocks:
        torch.nn.init.normal_(b.ada.weight, std=0.1)
    cond, spk, cfg = torch.randn(1, 12, 20).expand(2, -1, -1), torch.tensor([1, 1]), torch.ones(2)
    out = g(torch.randn(2, 16, 20), cond, spk, cfg, noise_labels=torch.zeros(2, 4, dtype=torch.long))
    assert not torch.allclose(out[0], out[1])  # different noise -> different sample

    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 15:] = False
    out_m = g(torch.randn(2, 16, 20), torch.randn(2, 12, 20), spk, cfg, mask=mask)
    out_m.sum().backward()
    assert torch.isfinite(out_m).all()


def test_default_size_is_small():
    g = DriftDiT(num_speakers=722)
    assert 15 < count_params(g) < 40


def test_window_mask_equals_full_attention_on_short_inputs_and_is_local():
    torch.manual_seed(0)
    g = _model(n_registers=2)
    for b in g.blocks:
        torch.nn.init.normal_(b.ada.weight, std=0.1)
    torch.nn.init.normal_(g.final.linear.weight, std=0.1)
    torch.nn.init.normal_(g.final.ada.weight, std=0.1)
    z, cond = torch.randn(1, 16, 40), torch.randn(1, 12, 40)
    args = (torch.tensor([0]), torch.ones(1))
    nl = torch.zeros(1, 4, dtype=torch.long)
    full = g(z, cond, *args, noise_labels=nl)
    assert torch.allclose(full, g(z, cond, *args, noise_labels=nl, attn_window=100), atol=1e-6)
    m = g.window_mask(2 + 20, 3, "cpu")[0, 0]
    assert m[:2].all() and m[:, :2].all()  # registers global
    assert m[10, 13] and not m[10, 14]


def test_window_attention_is_local_without_registers():
    torch.manual_seed(0)
    g = _model(n_registers=0)  # depth 2, patch 2
    for b in g.blocks:
        torch.nn.init.normal_(b.ada.weight, std=0.1)
    torch.nn.init.normal_(g.final.linear.weight, std=0.1)
    torch.nn.init.normal_(g.final.ada.weight, std=0.1)
    z, cond = torch.randn(1, 16, 40), torch.randn(1, 12, 40)
    args = (torch.tensor([0]), torch.ones(1))
    nl = torch.zeros(1, 4, dtype=torch.long)
    z2 = z.clone()
    z2[..., -2:] += 1.0  # perturb the last token (frames 38-39)
    out1 = g(z, cond, *args, noise_labels=nl, attn_window=2)
    out2 = g(z2, cond, *args, noise_labels=nl, attn_window=2)
    # receptive field: 2 blocks x 2 tokens -> tokens >= 15 (frames >= 30) may change, earlier ones may not
    assert torch.allclose(out1[..., :30], out2[..., :30], atol=1e-5)
    assert not torch.allclose(out1[..., 30:], out2[..., 30:])
    full1 = g(z, cond, *args, noise_labels=nl)
    full2 = g(z2, cond, *args, noise_labels=nl)
    assert not torch.allclose(full1[..., :30], full2[..., :30])  # full attention is global


def test_masked_padding_does_not_change_valid_frames():
    torch.manual_seed(0)
    g = _model().eval()
    for b in g.blocks:
        torch.nn.init.normal_(b.ada.weight, std=0.1)
    torch.nn.init.normal_(g.final.linear.weight, std=0.1)
    torch.nn.init.normal_(g.final.ada.weight, std=0.1)
    z, cond = torch.randn(1, 16, 22), torch.randn(1, 12, 22)
    args = (torch.tensor([1]), torch.ones(1))
    nl = torch.zeros(1, 4, dtype=torch.long)
    alone = g(z, cond, *args, noise_labels=nl)
    zp, cp = torch.nn.functional.pad(z, (0, 10), value=3.0), torch.nn.functional.pad(cond, (0, 10), value=3.0)
    mask = torch.arange(32)[None] < 22
    padded = g(zp, cp, *args, noise_labels=nl, mask=mask)
    assert torch.allclose(alone, padded[..., :22], atol=1e-5)
