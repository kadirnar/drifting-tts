import pytest
import torch

from drifting_tts.drift import (
    drift_force,
    drift_loss,
    energy_score,
    feature_drift_loss,
    feature_energy_loss,
    kyutai_drift_loss,
    learned_tau_drift_loss,
)


def test_anti_symmetry():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 16, 5, generator=g)
    a = torch.randn(2, 12, 5, generator=g) + 1.0
    b = torch.randn(2, 12, 5, generator=g) - 1.0
    f_ab, s_ab, _ = drift_force(x, pos=a, neg=b, gen_as_negative=False)
    f_ba, s_ba, _ = drift_force(x, pos=b, neg=a, gen_as_negative=False)
    assert torch.allclose(s_ab, s_ba)
    assert torch.allclose(f_ab, -f_ba, atol=1e-5)


def test_equilibrium_when_distributions_match():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 8, 4, generator=g)
    y = torch.randn(3, 10, 4, generator=g)
    _, _, info = drift_force(x, pos=y, neg=y.clone(), gen_as_negative=False)
    for k, v in info.items():
        if k.startswith("force_"):
            assert v.item() < 1e-10, (k, v)


def test_attraction_points_towards_data():
    x = torch.zeros(1, 4, 2) + torch.randn(1, 4, 2) * 0.01
    y = torch.tensor([[[5.0, 0.0]]]).expand(1, 6, 2) + torch.randn(1, 6, 2) * 0.01
    force, _, _ = drift_force(x, pos=y)
    assert (force[..., 0] > 0).all()


def test_gradient_only_through_generator():
    gen = torch.randn(2, 6, 3, requires_grad=True)
    pos = torch.randn(2, 5, 3, requires_grad=True)
    loss, info = drift_loss(gen, pos)
    loss.mean().backward()
    assert gen.grad is not None and gen.grad.abs().sum() > 0
    assert pos.grad is None
    assert "scale" in info


def test_cfg_weighted_negatives_change_force():
    g = torch.Generator().manual_seed(2)
    x, pos, neg = (torch.randn(1, 8, 3, generator=g) for _ in range(3))
    f0, _, _ = drift_force(x, pos, neg, weight_neg=torch.zeros(1, 8))
    f1, _, _ = drift_force(x, pos, neg, weight_neg=torch.ones(1, 8) * 2)
    assert not torch.allclose(f0, f1)


def test_feature_drift_loss_shapes():
    B, G, P, N, L, D = 2, 4, 3, 2, 5, 7
    gen = {"a": torch.randn(B, G, L, D, requires_grad=True), "b": torch.randn(B, G, 1, 3, requires_grad=True)}
    pos = {"a": torch.randn(B, P, L, D), "b": torch.randn(B, P, 1, 3)}
    neg = {"a": torch.randn(B, N, L, D), "b": torch.randn(B, N, 1, 3)}
    loss, info = feature_drift_loss(gen, pos, neg, weight_neg=torch.ones(B, N))
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "force_0.02/a" in info and "scale/b" in info


def test_toy_generator_matches_bimodal_target():
    """A tiny MLP generator driven only by the drift loss recovers both modes (no mode collapse)."""
    torch.manual_seed(0)
    nn = torch.nn
    net = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    def data(n):
        return torch.where(torch.rand(n, 1) < 0.5, -2.0, 2.0) + 0.3 * torch.randn(n, 1)

    for _ in range(300):
        x = net(torch.randn(256, 4))
        loss, _ = drift_loss(x[None], data(256)[None], affinity_floor=0.0)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
    with torch.no_grad():
        x = net(torch.randn(4000, 4)).squeeze(-1)
    frac_pos = (x > 0).float().mean().item()
    assert 0.35 < frac_pos < 0.65, frac_pos
    assert abs(x[x > 0].mean().item() - 2.0) < 0.3
    assert abs(x[x < 0].mean().item() + 2.0) < 0.3


def test_learned_tau_gradients_are_separated():
    gen = torch.randn(4, 6, 3, requires_grad=True)
    log_tau = torch.zeros((), requires_grad=True)
    loss, tau_loss, info = learned_tau_drift_loss(gen, torch.randn(4, 1, 3), log_tau, torch.randn(4, 2, 3),
                                                  weight_neg=torch.ones(4, 2))
    g_gen = torch.autograd.grad(loss.mean(), [gen, log_tau], allow_unused=True)
    assert g_gen[0].abs().sum() > 0 and g_gen[1] is None
    g_tau = torch.autograd.grad(tau_loss, [gen, log_tau], allow_unused=True)
    assert g_tau[0] is None and g_tau[1] is not None
    assert 0 < info["p_data"] < 1


def _kyutai_reference(x, y, tau):
    """Kyutai's ``Drifting.loss`` (pocket-tts ``samplers.py``) for one positive: columns [data, siblings]."""
    B, G, D = x.shape
    d = torch.cat([torch.cdist(x, y), torch.cdist(x, x)], -1)
    s = d.sum((1, 2), keepdim=True) / G**2  # per-row mean * n / (n - 1), n = G + 1
    d = d / s + torch.cat([torch.zeros(B, G, 1), 100 * torch.eye(G).expand(B, G, G)], -1)
    xs, ys = x / (s / D**0.5), y / (s / D**0.5)
    logit = -d / tau
    a = (logit.softmax(-1) * logit.softmax(-2)).clamp(min=1e-6).sqrt().detach()
    w_pos, w_neg = a[..., :1] * a[..., 1:].sum(-1, keepdim=True), a[..., 1:] * a[..., :1]
    v = w_pos @ ys - w_neg @ xs  # the coefficient on x, sum(w_pos) - sum(w_neg), is zero
    v = v / v.square().mean((1, 2), keepdim=True).sqrt().mean()
    return v.square().mean((1, 2)), -logit.log_softmax(-1)[..., 0].amax(-1).mean()


def test_kyutai_matches_pocket_tts_for_one_positive():
    g = torch.Generator().manual_seed(3)
    x, y = torch.randn(5, 6, 3, generator=g), torch.randn(5, 1, 3, generator=g)
    tau, tau_ref = torch.tensor(0.7, requires_grad=True), torch.tensor(0.7, requires_grad=True)
    loss, tau_loss, _ = kyutai_drift_loss(x, y, tau)
    loss_ref, tau_loss_ref = _kyutai_reference(x, y, tau_ref)
    assert torch.allclose(loss, loss_ref, rtol=1e-4) and torch.allclose(tau_loss, tau_loss_ref, rtol=1e-5)
    tau_loss.backward()
    tau_loss_ref.backward()
    assert torch.allclose(tau.grad, tau_ref.grad, rtol=1e-4)


def test_kyutai_anti_symmetry_and_equilibrium():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(3, 16, 5, generator=g)
    a, b = torch.randn(3, 12, 5, generator=g) + 1.0, torch.randn(3, 12, 5, generator=g) - 1.0
    tau = torch.tensor(0.3)

    def field(pos, neg):  # the loss gradient is -2 V / (G D s): the field up to a positive per-row factor
        gen = x.clone().requires_grad_(True)
        kyutai_drift_loss(gen, pos, tau, neg, gen_as_negative=False)[0].sum().backward()
        return gen.grad

    assert torch.allclose(field(a, b), -field(b, a), rtol=1e-4, atol=1e-8)
    assert kyutai_drift_loss(x, a, tau, a.clone(), gen_as_negative=False)[2]["force"] < 1e-10  # V_{p,p} = 0
    # training setting (siblings as negatives): generated ~ positives => field ~ 0 up to sampling noise
    x, y = torch.randn(32, 64, 2, generator=g), torch.randn(32, 64, 2, generator=g)
    force = lambda pos: kyutai_drift_loss(x, pos, torch.tensor(1.0))[2]["force"]
    assert force(y) < 0.1 * force(y + 1.0)


def test_kyutai_gradients_are_separated():
    gen = torch.randn(4, 6, 3, requires_grad=True)
    pos, neg = torch.randn(4, 2, 3, requires_grad=True), torch.randn(4, 3, 3, requires_grad=True)
    tau = torch.tensor(1.0, requires_grad=True)
    loss, tau_loss, info = kyutai_drift_loss(gen, pos, tau, neg, weight_neg=torch.rand(4, 3) * 2)
    g = torch.autograd.grad(loss.mean(), [gen, tau, pos, neg], allow_unused=True)
    assert g[0].abs().sum() > 0 and g[1] is None and g[2] is None and g[3] is None
    g = torch.autograd.grad(tau_loss, [gen, tau, pos, neg], allow_unused=True)
    assert g[1] is not None and g[0] is None and g[2] is None and g[3] is None
    assert 0 < info["p_data"] <= 1 and info["scale"] > 0


def test_feature_drift_loss_kyutai_shapes():
    B, G, P, N, L, D = 2, 4, 3, 2, 5, 7
    gen = {"a": torch.randn(B, G, L, D, requires_grad=True), "b": torch.randn(B, G, 1, 3, requires_grad=True)}
    pos = {k: torch.randn(B, P, *v.shape[2:]) for k, v in gen.items()}
    neg = {k: torch.randn(B, N, *v.shape[2:]) for k, v in gen.items()}
    shared = torch.tensor(1.0, requires_grad=True)
    per_map = {"a": torch.tensor(1.0, requires_grad=True), "b": torch.tensor(2.0, requires_grad=True)}
    for taus in (shared, per_map):
        loss, info = feature_drift_loss(gen, pos, neg, weight_neg=torch.ones(B, N), reduce="mean", taus=taus)
        loss.backward()
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert {"tau/a", "tau/b", "p_data/a", "force/b", "scale/a", "tau_loss"} <= info.keys()
    assert info["tau/b"] == 2.0 and shared.grad is not None and all(t.grad is not None for t in per_map.values())
    assert gen["a"].grad.shape == gen["a"].shape
    with pytest.raises(ValueError):
        feature_drift_loss(gen, pos, log_taus={"a": shared, "b": shared}, taus=shared)


def _conditional_toy(loss_fn, views: int, sigma: float, steps: int = 800, spread: float = 0.0, tau0: float = 0.0,
                     tau_lr: float = 1e-3):
    """K conditions, G samples each. Positives: the condition's target and ``views - 1`` perturbed views (the
    TTS regime) or, with ``spread``, one fresh draw from N(target, spread^2) per step (Kyutai's regime).
    ``tau`` starts at ``tau0`` (log tau for mode learned, raw tau for mode kyutai).
    Returns the centroid MSE and the spread of 256 samples per condition, and the tau trajectory."""
    torch.manual_seed(0)
    K, G, D = 16, 8, 4
    targets = torch.randn(K, D) * 2
    nn = torch.nn
    cond = nn.Embedding(K, 32)
    net = nn.Sequential(nn.Linear(32 + 8, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, D))
    tau = torch.tensor(tau0, requires_grad=True)
    opt = torch.optim.Adam([{"params": [*net.parameters(), *cond.parameters()]}, {"params": [tau], "lr": tau_lr}],
                           lr=1e-3)
    c = torch.arange(K)
    sample = lambda n=G: net(torch.cat([cond(c)[:, None].expand(K, n, 32), torch.randn(K, n, 8)], -1))
    taus = []
    for _ in range(steps):
        pos = targets[:, None].expand(K, views, D).clone()
        pos[:, 1:] += sigma * torch.randn_like(pos[:, 1:])
        if spread:
            pos += spread * torch.randn_like(pos)
        loss = loss_fn(sample(), pos, tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
        taus.append(tau.item())
    with torch.no_grad():
        x = sample(256)
    return ((x.mean(1) - targets) ** 2).mean().item(), x.std(1).mean().item(), taus


def _official(x, pos, _):
    return drift_loss(x, pos)[0].mean()


def _kyutai(x, pos, tau):
    loss, tau_loss, _ = kyutai_drift_loss(x, pos, tau)
    return loss.mean() + tau_loss


def _learned(x, pos, log_tau):
    loss, tau_loss, _ = learned_tau_drift_loss(x, pos, log_tau)
    return loss.mean() + tau_loss


def test_official_field_fits_a_single_positive_per_condition():
    mse, spread, _ = _conditional_toy(_official, views=1, sigma=0.0)
    assert mse < 0.05 and spread < 0.5


def test_kyutai_field_fits_a_single_positive_per_condition():
    mse, spread, taus = _conditional_toy(_kyutai, views=1, sigma=0.0, tau0=1.0, tau_lr=3e-3)
    assert mse < 0.05 and spread < 0.5
    # tau widens while the samples are far from the data, then anneals below its initial value
    assert max(taus) > 1.0 and taus[-1] < 0.8


@pytest.mark.parametrize("loss_fn", [_official, _kyutai], ids=["official", "kyutai"])
def test_one_positive_per_step_from_a_conditional_with_spread(loss_fn):
    mse, spread, _ = _conditional_toy(loss_fn, views=1, sigma=0.0, spread=0.5, tau0=1.0)
    assert mse < 0.05 and abs(spread - 0.5) < 0.1  # recovers the conditional mean and spread


def test_learned_tau_field_needs_several_positives_per_row():
    """Blog pseudo-code: converges with target + views; with one positive per row it diverges, whether the
    conditional is a delta or has spread (the union softmax's outward (1 - 2 w_data) x term)."""
    mse, spread, _ = _conditional_toy(_learned, views=8, sigma=0.1)
    assert mse < 0.05 and spread < 0.5
    for kw in ({}, {"spread": 0.5}):
        mse_one, _, _ = _conditional_toy(_learned, views=1, sigma=0.0, steps=300, **kw)
        assert mse_one > 1.0


def test_key_weights_exact_glob_and_weighted_mean():
    from drifting_tts.drift import key_weight

    kw = {"mel_global": 0.0, "detail_*": 4.0, "layer*": 2.0}
    assert key_weight("mel_global", kw) == 0.0 and key_weight("detail_patch4", kw) == 4.0
    assert key_weight("layer2_mean4", kw) == 2.0 and key_weight("conv1", kw) == 1.0 and key_weight("x", None) == 1.0

    torch.manual_seed(0)
    feats = lambda S: {k: torch.randn(2, S, 3, 4) for k in ("a", "b", "c")}  # noqa: E731
    gen, pos = feats(4), feats(3)
    per_key = {k: feature_drift_loss({k: gen[k]}, {k: pos[k]})[0] for k in gen}
    total, _ = feature_drift_loss(gen, pos, key_weights={"a": 3.0, "c": 0.0}, reduce="mean")
    assert torch.allclose(total, (3 * per_key["a"] + per_key["b"]) / 4)


def test_max_locations_subsamples_large_maps_only():
    torch.manual_seed(0)
    gen = {"big": torch.randn(2, 4, 50, 3, requires_grad=True), "small": torch.randn(2, 4, 5, 3)}
    pos = {"big": torch.randn(2, 3, 50, 3), "small": torch.randn(2, 3, 5, 3)}
    full, _ = feature_drift_loss(gen, pos, reduce="mean")
    same, _ = feature_drift_loss(gen, pos, reduce="mean", max_locations=50)
    assert torch.allclose(full, same)
    sub, info = feature_drift_loss(gen, pos, reduce="mean", max_locations=10)
    sub.backward()
    assert torch.isfinite(sub) and gen["big"].grad is not None
    assert (gen["big"].grad.abs().sum((0, 1, 3)) > 0).sum() == 10  # only the sampled locations get a gradient


def test_kyutai_row_weight_zero_rows_have_no_effect():
    """Weight-0 rows (padding) change neither the other rows' loss nor the temperature loss."""
    g = torch.Generator().manual_seed(3)
    x, y, tau = torch.randn(4, 6, 3, generator=g), torch.randn(4, 1, 3, generator=g), torch.tensor(0.5)
    loss, tau_loss, info = kyutai_drift_loss(x, y, tau)
    xp, yp = torch.cat([x, torch.zeros(2, 6, 3)]), torch.cat([y, torch.zeros(2, 1, 3)])
    w = torch.tensor([1.0, 1, 1, 1, 0, 0])
    loss_p, tau_loss_p, info_p = kyutai_drift_loss(xp, yp, tau, row_weight=w)
    assert torch.allclose(loss, loss_p[:4], atol=1e-5) and torch.allclose(tau_loss, tau_loss_p, atol=1e-6)
    assert torch.allclose(info["p_data"], info_p["p_data"], atol=1e-6)


# ---------------------------------------------------------------- one positive: the field against a proper score


def test_energy_score_is_smallest_for_the_data_distribution():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(20000, 1, 8, generator=g)
    scores, spreads = {}, {}
    for scale in (0.5, 0.9, 1.0, 1.1, 1.5):
        score, spread = energy_score(scale * torch.randn(20000, 16, 8, generator=g), y)
        assert score.shape == (20000,)
        scores[scale], spreads[scale] = float(score.mean()), float(spread)
    assert min(scores, key=scores.get) == 1.0  # neither narrower nor wider scores better
    assert spreads[0.5] < spreads[0.9] < 1.0 < spreads[1.1] < spreads[1.5]
    assert spreads[1.0] == pytest.approx(1.0, abs=0.01)
    x = torch.randn(4, 6, 3, generator=g, requires_grad=True)
    energy_score(x, torch.randn(4, 1, 3, generator=g))[0].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0  # the zero self distances give no NaN


def test_feature_energy_loss_weights_maps_and_reports_their_spread():
    g = torch.Generator().manual_seed(0)
    gen = {"a": torch.randn(64, 8, 3, 5, generator=g), "b": 100 * torch.randn(64, 8, 1, 7, generator=g)}
    pos = {"a": torch.randn(64, 1, 3, 5, generator=g), "b": 100 * torch.randn(64, 1, 1, 7, generator=g)}
    loss, info = feature_energy_loss(gen, pos)
    assert set(info) == {"spread/a", "spread/b"} and 0.8 < float(info["spread/a"]) < 1.2
    only_a, _ = feature_energy_loss(gen, pos, key_weights={"b": 0.0})
    only_b, _ = feature_energy_loss(gen, pos, key_weights={"a": 0.0})
    assert float(loss) == pytest.approx((float(only_a) + float(only_b)) / 2, rel=1e-5)  # the mean of the maps
    assert float(only_b) == pytest.approx(float(only_a), rel=0.3)  # each map in units of its own distances
    torch.manual_seed(0)
    sub, _ = feature_energy_loss(gen, pos, key_weights={"b": 0.0}, max_locations=2)
    assert float(sub) == pytest.approx(float(only_a), rel=0.3)


def _band_equilibrium(loss: str, bands: bool, tau: float = 0.3, steps: int = 300, seed: int = 0) -> torch.Tensor:
    """Variances (truth: 1) per band of a generator ``x = a_band * z`` trained **from the truth** on Gaussian data
    whose four bands of 25 dimensions carry 59 / 25 / 11 / 5% of the variance (the residual of a mel patch along
    temporal frequency, docs/HEAD_REALISM.md), with one positive per row. ``bands``: one kernel map per band instead
    of one over all 100 dimensions."""
    torch.manual_seed(seed)
    n, rows, samples = 25, 256, 16
    std = (torch.tensor([0.59, 0.25, 0.11, 0.05]) * 4).sqrt().repeat_interleave(n)
    log_a = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([log_a], lr=2e-2)

    def maps(t):
        return {f"b{j}": t[:, :, None, j * n:(j + 1) * n] for j in range(4)} if bands else {"all": t[:, :, None]}

    tail = []
    for step in range(steps):
        x = log_a.exp().repeat_interleave(n) * std * torch.randn(rows, samples, 4 * n)
        fx, fy = maps(x), maps(std * torch.randn(rows, 1, 4 * n))
        if loss == "kyutai":
            total, _ = feature_drift_loss(fx, fy, None, taus=torch.tensor(tau), reduce="mean")
        else:
            total, _ = feature_energy_loss(fx, fy)
        opt.zero_grad()
        total.backward()
        opt.step()
        if step >= steps - 50:
            tail.append(log_a.detach().exp() ** 2)
    return torch.stack(tail).mean(0)


def test_one_positive_field_shrinks_the_small_directions_and_a_proper_score_does_not():
    joint = _band_equilibrium("kyutai", bands=False)  # seeds 0-3: the smallest band ends at 0.72-0.81, the largest at
    assert joint[0] > 0.9 and joint[3] < 0.86 and joint[0] - joint[3] > 0.1  # 0.93-0.98: the truth is not a fixed point
    assert joint.argmin() == 3  # the band that carries the least of the distance shrinks the most
    per_band = _band_equilibrium("kyutai", bands=True)  # every band its own kernel: no band is small in its map
    assert per_band.min() > 0.93 and per_band.max() - per_band.min() < 0.06
    energy = _band_equilibrium("energy", bands=False)  # proper: no band shrinks (the small ones are noisy in this toy)
    assert energy[3] > 0.9 and abs(float(energy.mean()) - 1) < 0.1

