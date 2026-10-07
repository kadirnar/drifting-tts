import torch

from drifting_tts.models import bigvgan_disc as bd


def test_multi_resolution_discriminator_shapes_and_state_dict():
    h = {"resolutions": [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]], "discriminator_channel_mult": 1}
    mrd = bd.MultiResolutionDiscriminator(h)
    y, y_hat = torch.randn(2, 1, 8192), torch.randn(2, 1, 8192)
    real, fake, fmap_r, fmap_g = mrd(y, y_hat)
    assert len(real) == len(fake) == 3 and all(len(f) == 6 for f in fmap_r + fmap_g)
    assert all(r.shape[0] == 2 for r in real)
    keys = set(mrd.state_dict())  # the released BigVGAN v1 / base layout: 3 x (5 convs + post) x (bias, g, v)
    assert len(keys) == 54 and "discriminators.0.convs.0.weight_g" in keys and "discriminators.2.conv_post.bias" in keys
    loss = bd.discriminator_loss(real, fake) + bd.generator_loss(fake) + bd.feature_loss(fmap_r, fmap_g)
    assert torch.isfinite(loss)
