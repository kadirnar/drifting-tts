import torch

from drifting_tts.models.mae import MelMAE


def test_mae_forward_and_activations():
    torch.manual_seed(0)
    model = MelMAE(n_mels=20, base_channels=16, layers=(1, 2, 1, 1), num_classes=5)
    mel = torch.randn(3, 20, 64)
    loss, m = model(mel, mask_ratio=0.5, labels=torch.tensor([0, 1, 2]), cls_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss) and "cls_acc" in m

    acts = model.get_activations(mel, every_k_block=1)
    assert acts["mel_global"].shape == (3, 1, 20 * 64)
    assert acts["conv1"].shape == (3, 64, 16)
    assert acts["layer4"].shape == (3, 8, 128)
    assert acts["layer4_mean"].shape == (3, 1, 128)
    assert acts["layer2_mean4"].shape == (3, 8, 32)
    # official get_activations: every k-th block output (pre stage-GroupNorm), the stage's last block included
    assert {"layer1_blk1", "layer2_blk1", "layer2_blk2"} <= acts.keys()
    assert not torch.allclose(acts["layer2_blk2"], acts["layer2"])
    acts2 = model.get_activations(mel, every_k_block=2)
    assert "layer2_blk2" in acts2 and "layer2_blk1" not in acts2 and "layer1_blk1" not in acts2
    for v in acts.values():
        assert v.ndim == 3 and torch.isfinite(v).all()


def test_activations_are_differentiable():
    model = MelMAE(n_mels=8, base_channels=8, layers=(1, 1, 1, 1)).requires_grad_(False)
    x = torch.randn(2, 8, 32, requires_grad=True)
    acts = model.get_activations(x)
    sum(v.float().pow(2).mean() for v in acts.values()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_mae2d_patchify_roundtrip_forward_and_activations():
    from drifting_tts.models.mae2d import MelMAE2d

    torch.manual_seed(0)
    model = MelMAE2d(n_mels=20, base_channels=8, layers=(1, 1, 1, 1), input_patch=(2, 2), num_classes=3,
                     mask_patch=(4, 4))
    mel = torch.randn(2, 20, 33)
    assert torch.allclose(model.unpatchify(model.patchify(mel), 20, 33), mel)
    loss, m = model(mel, labels=torch.tensor([0, 2]), cls_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss) and "cls_acc" in m

    x = torch.randn(2, 20, 32, requires_grad=True)
    acts = model.get_activations(x, spectral_detail=True)
    assert acts["conv1"].shape == (2, 10 * 16, 8)  # (F/2 x T/2) locations, 8 channels
    assert acts["conv1_fprofile"].shape == (2, 10, 8)
    assert "detail_patch4" in acts and acts["detail_global"].shape == (2, 1, 20 * 32)
    assert "layer2_mean2" in acts
    sum(v.float().pow(2).mean() for v in acts.values()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_spectral_detail_removes_the_envelope():
    from drifting_tts.models.mae import spectral_detail

    flat = torch.ones(1, 10, 5) * 3.0
    assert spectral_detail(flat).abs().max() < 1e-6
    comb = torch.zeros(1, 10, 5)
    comb[:, ::2] = 1.0  # harmonic-like alternation along frequency
    assert spectral_detail(comb).abs().mean() > 0.3
