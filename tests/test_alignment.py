import itertools

import numba
import numpy as np
import pytest
import torch

from drifting_tts.alignment import gaussian_log_prior, generate_path, maximum_path, sequence_mask
from drifting_tts.models.text_encoder import (
    TextEncoder,
    align,
    duration_loss,
    durations_to_alignment,
    expand,
    prior_loss,
)


def test_generate_path():
    d = torch.tensor([[2.0, 3.0, 0.0, 1.0]])
    mask = torch.ones(1, 4, 6)
    path = generate_path(d, mask)
    assert path.sum(1).eq(1).all()  # each frame belongs to one token
    assert path.sum(2).tolist() == [[2, 3, 0, 1]]


def test_mas_recovers_durations():
    torch.manual_seed(0)
    durs = [[3, 1, 4, 2], [2, 2, 2, 0]]
    mu = torch.randn(2, 8, 4) * 3
    T = 10
    y = torch.zeros(2, 8, T)
    for b, ds in enumerate(durs):
        t = 0
        for n, d in enumerate(ds):
            y[b, :, t: t + d] = mu[b, :, n: n + 1] + 0.01 * torch.randn(8, d)
            t += d
    x_len, y_len = torch.tensor([4, 3]), torch.tensor([10, 6])
    x_mask = sequence_mask(x_len, 4)[:, None].float()
    y_mask = sequence_mask(y_len, T)[:, None].float()
    attn = maximum_path(gaussian_log_prior(mu, y), x_mask.transpose(1, 2) * y_mask)
    assert attn.sum(2)[0].tolist() == [3, 1, 4, 2]
    assert attn.sum(2)[1].tolist() == [2, 2, 2, 0]


@numba.njit(boundscheck=False)
def _reference_path_single(value, t_x, t_y, out):  # the previous implementation: strided DP on [N_text, T_mel]
    neg_inf = -1e9
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = neg_inf if x == y else value[x, y - 1]
            if x == 0:
                v_prev = 0.0 if y == 0 else neg_inf
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        out[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1


def reference_maximum_path(log_prior, mask):
    values = (log_prior * mask).float().cpu().numpy().astype(np.float32)
    out = np.zeros(values.shape, dtype=np.int32)
    t_xs = mask.sum(1)[:, 0].int().cpu().numpy()
    t_ys = mask.sum(2)[:, 0].int().cpu().numpy()
    for b in range(values.shape[0]):
        _reference_path_single(values[b], t_xs[b], t_ys[b], out[b])
    return torch.from_numpy(out).to(device=log_prior.device, dtype=log_prior.dtype)


def _random_problem(B, N, T, seed):
    """Random scores with varied text / mel lengths (``t_x <= t_y``) and a ``[B, N, T]`` mask."""
    g = torch.Generator().manual_seed(seed)
    x_len = torch.randint(1, N + 1, (B,), generator=g)
    x_len[0] = N
    y_len = x_len + (torch.rand(B, generator=g) * (T - x_len + 1)).long()
    y_len[-1] = T
    mask = sequence_mask(x_len, N)[:, :, None].float() * sequence_mask(y_len, T)[:, None, :].float()
    return 3 * torch.randn(B, N, T, generator=g), mask, x_len, y_len


def test_mas_matches_reference_implementation():
    for seed, (B, N, T) in enumerate([(6, 17, 40), (5, 60, 200), (3, 1, 9), (4, 33, 33), (8, 120, 400)]):
        log_prior, mask, _, _ = _random_problem(B, N, T, seed)
        attn = maximum_path(log_prior, mask)
        assert attn.dtype == log_prior.dtype and attn.is_contiguous()
        assert torch.equal(attn, reference_maximum_path(log_prior, mask))
        assert attn.sum(1).eq(mask[:, 0]).all()  # every valid frame on exactly one token, padding empty


def _brute_force_path(value: np.ndarray, t_x: int, t_y: int) -> np.ndarray:
    """Best monotonic alignment by enumerating every duration sequence (each token >= 1 frame)."""
    best, best_tokens = -np.inf, None
    for cuts in itertools.combinations(range(1, t_y), t_x - 1):
        tokens = np.repeat(np.arange(t_x), np.diff((0, *cuts, t_y)))
        score = value[tokens, np.arange(t_y)].astype(np.float64).sum()
        if score > best:
            best, best_tokens = score, tokens
    path = np.zeros((t_x, t_y), dtype=np.float32)
    path[best_tokens, np.arange(t_y)] = 1
    return path


def test_mas_matches_brute_force_on_tiny_inputs():
    for seed in range(50):
        log_prior, mask, x_len, y_len = _random_problem(4, 4, 7, 100 + seed)
        attn = maximum_path(log_prior, mask)
        for b in range(4):
            tx, ty = int(x_len[b]), int(y_len[b])
            assert np.array_equal(attn[b, :tx, :ty].numpy(), _brute_force_path(log_prior[b, :tx, :ty].numpy(), tx, ty))
            assert attn[b].sum() == ty


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mas_cuda_matches_cpu():
    log_prior, mask, _, _ = _random_problem(8, 90, 300, 7)
    attn = maximum_path(log_prior.cuda(), mask.cuda())
    assert attn.is_cuda and torch.equal(attn.cpu(), maximum_path(log_prior, mask))


def test_text_encoder_training_and_inference_paths():
    torch.manual_seed(0)
    enc = TextEncoder(n_vocab=20, n_mels=16, d=32, heads=2, layers=2, ffn=64, num_speakers=3, spk_dim=8)
    text = torch.randint(2, 20, (2, 7))
    text_len = torch.tensor([7, 5])
    y = torch.randn(2, 16, 30)
    y_len = torch.tensor([30, 21])
    y_mask = sequence_mask(y_len, 30)[:, None].float()
    h, mu, logw, x_mask = enc(text, text_len, torch.tensor([0, 2]))
    attn, logw_t = align(mu, x_mask, y, y_mask)
    assert attn.sum(1)[0].sum() == 30 and attn.sum(1)[1].sum() == 21
    mu_y, h_y = expand(mu, attn), expand(h, attn)
    assert mu_y.shape == (2, 16, 30) and h_y.shape == (2, 32, 30)
    loss = prior_loss(mu_y, y, y_mask) + duration_loss(logw, logw_t, x_mask)
    loss.backward()
    assert torch.isfinite(loss)
    with torch.no_grad():
        attn_inf, y_len_inf = durations_to_alignment(logw, x_mask)
    assert attn_inf.shape[0] == 2 and (attn_inf.sum((1, 2)) == y_len_inf).all()


def test_token_pitch_averages_voiced_frames():
    import math

    from drifting_tts.models.text_encoder import token_pitch

    attn = generate_path(torch.tensor([[2.0, 3.0]]), torch.ones(1, 2, 5))
    f0 = torch.tensor([[100.0, 0.0, 200.0, 200.0, 0.0]])
    tp = token_pitch(f0, attn, lf0_mean=math.log(100.0), lf0_std=1.0)
    assert tp.shape == (1, 1, 2)
    assert abs(tp[0, 0, 0].item()) < 1e-5  # token 1: one voiced frame at the mean pitch
    assert abs(tp[0, 0, 1].item() - math.log(2.0)) < 1e-5  # token 2: 200 Hz voiced frames only
    assert token_pitch(torch.zeros(1, 5), attn, 0.0, 1.0).abs().max() == 0  # all unvoiced


def test_text_encoder_is_invariant_to_padding():
    """A sequence must encode identically alone and padded inside a batch (no attention to padding)."""
    torch.manual_seed(0)
    enc = TextEncoder(n_vocab=20, n_mels=16, d=32, heads=2, layers=2, ffn=64, num_speakers=3, spk_dim=8).eval()
    text = torch.randint(2, 20, (1, 9))
    alone = enc(text, torch.tensor([9]), torch.tensor([1]))
    padded = torch.cat([text, torch.randint(2, 20, (1, 6))], 1)
    batch = enc(torch.cat([padded, torch.randint(2, 20, (1, 15))]), torch.tensor([9, 15]), torch.tensor([1, 2]))
    for a, b in zip(alone[:3], batch[:3]):
        assert torch.allclose(a[0], b[0, :, :9], atol=1e-5)
