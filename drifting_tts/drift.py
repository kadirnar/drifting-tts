"""Drifting field and loss (Deng et al., *Generative Modeling via Drifting*, arXiv 2602.04770).

A PyTorch port of ``drift_loss.py`` from the official JAX release (github.com/lambertae/drifting).

For a generated sample ``x`` the drifting field is the anti-symmetric attraction / repulsion

    V(x) = E_{y+ ~ p, y- ~ q} [ k(x, y+) k(x, y-) (y+ - y-) ] / (Z_p Z_q),   k(x, y) = exp(-||x - y|| / tau)

so that ``p = q  =>  V = 0`` (Prop. 3.1). Training regresses the generator output onto the
frozen, drifted target ``stopgrad(x + V(x))``. Implementation details follow the official code:

* distances are normalised by their mean so the kernel temperature is scale free;
* the kernel is normalised with a softmax over targets *and* over generated samples
  (geometric mean of both), which keeps anti-symmetry;
* each generated sample is masked out of its own negative set;
* several temperatures are used and each force is normalised to unit RMS before summing;
* extra negatives can carry weights (used for training-time classifier-free guidance).

Two learned-temperature variants after Kyutai (*Pocket TTS with a drifting objective*, 2026) are included:
:func:`kyutai_drift_loss` (their released code, ``drift.mode: kyutai``) and :func:`learned_tau_drift_loss`
(the simplified pseudo-code of their blog post, ``drift.mode: learned``; unstable here, see docs/DESIGN.md §5).
"""

from __future__ import annotations

from fnmatch import fnmatchcase

import torch
from torch import Tensor

DEFAULT_TEMPERATURES = (0.02, 0.05, 0.2)


def cdist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Batched Euclidean distance ``[B, N, D] x [B, M, D] -> [B, N, M]``."""
    xy = torch.einsum("bnd,bmd->bnm", x, y)
    xx = (x * x).sum(-1)
    yy = (y * y).sum(-1)
    sq = xx[:, :, None] + yy[:, None, :] - 2.0 * xy
    return sq.clamp_min(eps).sqrt()


def _product_field(
    logits: Tensor, targets_w: Tensor, split: int, targets_s: Tensor, gen_s: Tensor, affinity_floor: float
) -> Tensor:
    """Two-sided product-form field (official ``drift_loss``; Kyutai ``samplers.py:233-238``).

    ``a = sqrt(softmax_row * softmax_col)`` (floored) times the target weights, ``a-`` / ``a+`` the columns
    before / after ``split``. ``V_i = sum_{j,k} a+_ij a-_ik (y+_j - y-_k)``: attraction and repulsion carry the
    same mass ``sum(a+) sum(a-)``, so the coefficient on ``gen_s`` (last term) is zero for any weights.
    """
    aff = (logits.softmax(dim=-1) * logits.softmax(dim=-2)).clamp_min(affinity_floor).sqrt()
    aff = aff * targets_w[:, None, :]
    aff_neg, aff_pos = aff[..., :split], aff[..., split:]
    sum_pos = aff_pos.sum(-1, keepdim=True)
    sum_neg = aff_neg.sum(-1, keepdim=True)
    # sum_{j,k} a+_j a-_k (y+_j - y-_k) written as linear coefficients on the targets
    coeff = torch.cat([-aff_neg * sum_pos, aff_pos * sum_neg], dim=-1)
    return torch.einsum("bij,bjd->bid", coeff, targets_s) - coeff.sum(-1, keepdim=True) * gen_s


@torch.no_grad()
def drift_force(
    gen: Tensor,
    pos: Tensor,
    neg: Tensor | None = None,
    weight_gen: Tensor | None = None,
    weight_pos: Tensor | None = None,
    weight_neg: Tensor | None = None,
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    gen_as_negative: bool = True,
    affinity_floor: float = 1e-6,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Compute the (normalised) drift of every generated sample.

    Args:
        gen: ``[B, G, D]`` generated samples (queries; also used as negatives).
        pos: ``[B, P, D]`` positive samples (data).
        neg: ``[B, N, D]`` extra negative samples (e.g. unconditional data for CFG).
        weight_*: per-sample weights ``[B, *]`` (default 1).
        temperatures: kernel temperatures ``R`` (in units of the mean distance).
        gen_as_negative: use the other generated samples as negatives (standard). If False only
            ``neg`` acts as the negative set, which is how the equilibrium can be unit tested.
        affinity_floor: lower clamp of the product of the two softmaxes (official code: 1e-6). It
            adds a weak uniform (moment-matching) term; very low-dimensional, multi-modal problems
            work better with a smaller floor.

    Returns:
        ``force`` ``[B, G, D]`` in normalised coordinates, ``scale`` (the coordinate scale such that
        ``x / scale`` has O(1) entries) and an info dict with the raw per-temperature force norms.
    """
    gen = gen.float()
    B, G, D = gen.shape
    if neg is None:
        neg = gen.new_zeros(B, 0, D)
    pos, neg = pos.float(), neg.float()
    P, N = pos.shape[1], neg.shape[1]
    ones = lambda t: t.new_ones(t.shape[:2])
    weight_gen = ones(gen) if weight_gen is None else weight_gen.float()
    weight_pos = ones(pos) if weight_pos is None else weight_pos.float()
    weight_neg = ones(neg) if weight_neg is None else weight_neg.float()
    if not gen_as_negative:
        weight_gen = torch.zeros_like(weight_gen)

    targets = torch.cat([gen, neg, pos], dim=1)  # [B, G+N+P, D]
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)  # [B, G+N+P]

    # --- scale normalisation (global over the whole batch, as in the official code) ---
    dist = cdist(gen, targets)  # [B, G, G+N+P]
    scale = (dist * targets_w[:, None, :]).mean() / targets_w.mean()
    scale_inputs = (scale / D**0.5).clamp_min(1e-3)
    gen_s = gen / scale_inputs
    targets_s = targets / scale_inputs
    dist_n = dist / scale.clamp_min(1e-3)

    # --- mask each generated sample out of its own negative set ---
    eye = torch.eye(G, device=gen.device, dtype=dist.dtype)
    dist_n = dist_n + 100.0 * torch.nn.functional.pad(eye, (0, N + P))[None]

    info: dict[str, Tensor] = {"scale": scale}
    force = torch.zeros_like(gen_s)
    for R in temperatures:
        f = _product_field(-dist_n / R, targets_w, G + N, targets_s, gen_s, affinity_floor)
        f_norm = (f**2).mean()
        info[f"force_{R:g}"] = f_norm
        force = force + f / f_norm.clamp_min(1e-8).sqrt()
    return force, scale_inputs, info


def drift_loss(
    gen: Tensor,
    pos: Tensor,
    neg: Tensor | None = None,
    weight_gen: Tensor | None = None,
    weight_pos: Tensor | None = None,
    weight_neg: Tensor | None = None,
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    affinity_floor: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Drifting loss ``|| x - stopgrad(x + V(x)) ||^2`` (Eq. 6), per batch element.

    Shapes as in :func:`drift_force`. Gradients flow only through ``gen``.
    Returns ``loss`` of shape ``[B]`` and an info dict of scalars.
    """
    force, scale_inputs, info = drift_force(
        gen.detach(), pos, neg, weight_gen, weight_pos, weight_neg, temperatures,
        affinity_floor=affinity_floor,
    )
    gen_s = gen.float() / scale_inputs
    goal = (gen_s.detach() + force).detach()
    loss = ((gen_s - goal) ** 2).mean(dim=(-1, -2))
    return loss, info


def kyutai_drift_loss(
    gen: Tensor,
    pos: Tensor,
    tau: Tensor,
    neg: Tensor | None = None,
    weight_neg: Tensor | None = None,
    gen_as_negative: bool = True,
    affinity_floor: float = 1e-6,
    min_tau: float = 1e-3,
    min_scale: float = 1e-3,
    row_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Kyutai's released drifting objective (pocket-tts ``training/modules/samplers.py``, class ``Drifting``).

    The paper's two-sided product-form field (as in :func:`drift_force`) at one learned temperature, with
    per-row normalisation; generalised to ``P`` positives and weighted extra negatives. Per drift problem (row):

    * ``s`` = mean distance from the generated samples to their siblings and the positives, self pairs
      excluded (Kyutai's ``n / (n - 1)``). Distances are divided by ``s``, the regression coordinates by
      ``s / sqrt(D)``. Weighted (CFG) negatives are left out of ``s``, so it does not depend on the guidance
      scale; with ``gen_as_negative=False`` (unit tests) they replace the siblings;
    * ``V`` is the product-form field at ``tau`` (raw, clamped at ``min_tau``), divided by the mean over rows
      of its per-row RMS; the generated samples regress onto ``stopgrad(x + V)``;
    * ``tau`` is trained only by ``-mean_rows max_i log sum_{j in pos} softmax_j(-d_ij / tau + log w_j)``: per
      row, the generated sample that puts the most kernel mass on the positives sets the temperature. The
      candidates are the positives, the siblings (self masked) and the extra negatives with multiplicity ``w``.

    ``row_weight`` ``[B]`` (optional) weights the rows in the force normalisation, the temperature loss and the
    info; weight-0 rows (e.g. padding, so that batches of variable length need no host sync to select rows) then
    have no effect on the others.

    Shapes as in :func:`drift_force`; ``tau`` is a scalar tensor. Returns the per-row drift loss ``[B]``
    (gradient only w.r.t. ``gen``), the temperature loss (gradient only w.r.t. ``tau``) and an info dict
    (mean row ``scale``, raw field ``force`` before normalisation, ``p_data``: the best sample's mass on the data).
    """
    B, G, D = gen.shape
    with torch.no_grad():
        x = gen.detach().float()
        pos = pos.float()
        neg = x.new_zeros(B, 0, D) if neg is None else neg.float()
        P, N = pos.shape[1], neg.shape[1]
        w_neg = x.new_ones(B, N) if weight_neg is None else weight_neg.float()
        targets = torch.cat([x, neg, pos], dim=1)  # siblings | extra negatives | positives
        targets_w = torch.cat([x.new_full((B, G), float(gen_as_negative)), w_neg, x.new_ones(B, P)], dim=1)
        dist = cdist(x, targets)  # [B, G, G+N+P]

        # per-row scale (samplers.py:254-256); the ~0 self distances are summed but not counted
        ref = torch.cat([dist[..., :G], dist[..., G + N:]], -1) if gen_as_negative else dist[..., G:]
        count = G * ref.shape[-1] - (G if gen_as_negative else 0)
        scale = (ref.sum((1, 2)) / count).clamp_min(min_scale)[:, None, None]
        eye = torch.eye(G, device=x.device, dtype=dist.dtype)
        dist_n = dist / scale + 100.0 * torch.nn.functional.pad(eye, (0, N + P))[None]  # self mask (:257-261)
        scale_inputs = scale / D**0.5  # per-row coordinates (:262-263)
        x_s, t_s = x / scale_inputs, targets / scale_inputs

    if row_weight is None:
        row_mean = lambda v: v.mean()  # noqa: E731
    else:
        rw = row_weight.float()
        row_mean = lambda v: (v * rw).sum() / rw.sum().clamp_min(1e-8)  # noqa: E731
    logits = -dist_n / tau.float().clamp_min(min_tau)  # differentiable w.r.t. tau only
    with torch.no_grad():
        f = _product_field(logits.detach(), targets_w, G + N, t_s, x_s, affinity_floor)
        rms = (f**2).mean((1, 2)).clamp_min(1e-8).sqrt()
        goal = x_s + f / row_mean(rms)  # normalize_force="batch": mean over rows of the per-row RMS (:239-241)
    loss = ((gen.float() / scale_inputs - goal) ** 2).mean(dim=(-1, -2))  # (:266)

    # tau as the calibration of a classifier that picks the data out of the candidates (:232, :268)
    log_p = torch.log_softmax(logits + targets_w.clamp_min(1e-12).log()[:, None, :], dim=-1)
    best = torch.logsumexp(log_p[..., G + N:], dim=-1).amax(dim=-1)  # [B]: max over the generated samples
    tau_loss = -row_mean(best)
    info = {"scale": row_mean(scale[:, 0, 0]), "force": row_mean((f**2).mean((1, 2))),
            "p_data": row_mean(best.detach().exp())}
    return loss, tau_loss, info


def learned_tau_drift_loss(
    gen: Tensor,
    pos: Tensor,
    log_tau: Tensor,
    neg: Tensor | None = None,
    weight_neg: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Learned-temperature drift loss after the *pseudo-code* of Kyutai's blog post (``drift.mode: learned``).

    Not what Kyutai's released code runs (that is :func:`kyutai_drift_loss`). Kept for reproducibility: with
    one positive, its union softmax has a self term ``(1 - 2 w_data) x`` that spreads the samples apart while
    the data holds less than half of the kernel mass, and it diverged here (docs/DESIGN.md §5).
    Built for conditional problems with a **single positive** per condition:

    * distances are normalised *per row* (each drift problem by its own mean distance), so ``tau``
      is a dimensionless fraction of that condition's typical distance;
    * one softmax over the union of candidates {data, siblings, extra negatives} gives the field
      ``V = sum_j w_j (y+_j - x) - sum_k w_k (y-_k - x)`` (anti-symmetric), normalised to unit RMS;
    * ``tau = exp(log_tau)`` is trained by ``-log sum_{j in data} softmax(-d / tau)_j``, i.e. as the
      calibration of a classifier that picks the data sample out of the candidates. It starts wide and
      anneals as the generated samples close in on the data.

    Shapes as in :func:`drift_force`; ``weight_neg`` enters the softmax as a sample multiplicity.
    Returns the per-problem drift loss ``[B]`` (gradient only w.r.t. ``gen``), the temperature loss
    (gradient only w.r.t. ``log_tau``) and an info dict.
    """
    B, G, D = gen.shape
    with torch.no_grad():
        x = gen.detach().float()
        pos = pos.float()
        neg = x.new_zeros(B, 0, D) if neg is None else neg.float()
        P, N = pos.shape[1], neg.shape[1]
        targets = torch.cat([x, neg, pos], dim=1)  # siblings | extra negatives | data
        log_c = x.new_zeros(B, G + N + P)
        if weight_neg is not None and N > 0:
            log_c[:, G: G + N] = weight_neg.float().clamp_min(1e-12).log()
        log_c = log_c[:, None, :].expand(B, G, -1).clone()
        log_c[:, :, :G] -= 1e4 * torch.eye(G, device=x.device)  # a sample neither attracts nor repels itself

        dist = cdist(x, targets)
        valid = (log_c > -1e3).float()
        scale = (dist * valid).sum() / valid.sum()  # global coordinate scale for the regression target
        scale_inputs = (scale / D**0.5).clamp_min(1e-3)
        row_mean = (dist * valid).sum((1, 2)) / valid.sum((1, 2))
        dist_n = dist / row_mean.clamp_min(1e-8)[:, None, None]

        w = torch.softmax(-dist_n / log_tau.detach().exp() + log_c, dim=-1)
        sign = torch.cat([-x.new_ones(G + N), x.new_ones(P)])
        coeff = w * sign
        x_s, t_s = x / scale_inputs, targets / scale_inputs
        f = torch.einsum("bij,bjd->bid", coeff, t_s) - coeff.sum(-1, keepdim=True) * x_s
        f_norm = (f**2).mean()
        force = f / f_norm.clamp_min(1e-8).sqrt()
        goal = x_s + force

    log_p = torch.log_softmax(-dist_n / log_tau.exp() + log_c, dim=-1)
    tau_loss = -torch.logsumexp(log_p[..., G + N:], dim=-1).mean()
    loss = ((gen.float() / scale_inputs - goal) ** 2).mean(dim=(-1, -2))
    info = {"scale": scale, "force": f_norm, "p_data": log_p.detach()[..., G + N:].exp().sum(-1).mean()}
    return loss, tau_loss, info


def key_weight(key: str, key_weights: dict[str, float] | None) -> float:
    """Weight of a feature map: an exact key, else the first matching glob pattern (``detail_*``), else 1."""
    if not key_weights:
        return 1.0
    if key in key_weights:
        return float(key_weights[key])
    return next((float(w) for pat, w in key_weights.items() if fnmatchcase(key, pat)), 1.0)


def _flatten_locations(t: Tensor) -> Tensor:
    """``[B, S, L, D] -> [B * L, S, D]``: one drift problem per (condition, location)."""
    B, S, L, D = t.shape
    return t.permute(0, 2, 1, 3).reshape(B * L, S, D)


def feature_drift_loss(
    gen_feats: dict[str, Tensor],
    pos_feats: dict[str, Tensor],
    neg_feats: dict[str, Tensor] | None = None,
    weight_neg: Tensor | None = None,
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    key_weights: dict[str, float] | None = None,
    affinity_floor: float = 1e-6,
    reduce: str = "sum",
    log_taus: dict[str, Tensor] | None = None,
    taus: Tensor | dict[str, Tensor] | None = None,
    max_locations: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum of drift losses over a dict of multi-scale features (Eq. 13-14).

    Every feature tensor has shape ``[B, S, L, D]``: ``B`` conditions, ``S`` samples for that
    condition, ``L`` feature locations, ``D`` channels. Each (condition, location) pair is an
    independent drift problem, exactly like the ``(b f) x d`` rearrangement of the official code.

    Args:
        gen_feats / pos_feats / neg_feats: feature dicts with matching keys.
        weight_neg: ``[B, N]`` weights of the extra negatives (CFG), broadcast over locations.
        key_weights: optional per-map multipliers, keyed by exact name or glob pattern (default 1, see
            :func:`key_weight`). Maps with weight 0 are skipped.
        reduce: ``"sum"`` over feature maps (official) or ``"mean"`` (the weighted mean, so the scale is
            independent of the number of maps and of the weights).
        log_taus: ``drift.mode: learned``: one learned log-temperature per feature map and the
            :func:`learned_tau_drift_loss` field (blog pseudo-code).
        taus: ``drift.mode: kyutai``: a raw learned temperature shared by all feature maps (a tensor, as in
            Kyutai) or one per map (a dict), and the :func:`kyutai_drift_loss` field.

        max_locations: if set, maps with more locations use a random subset of this many (the same for
            every condition and sample) per call: an unbiased estimate of the per-map mean over locations at a
            fraction of the cost (the 2-D MAE's ``conv1`` / ``layer1`` maps have 6400 locations per window).

    With ``log_taus`` or ``taus`` the returned total also contains the temperature losses (averaged over the
    maps with ``reduce: mean``); their gradient only reaches the temperatures.
    """
    if log_taus is not None and taus is not None:
        raise ValueError("pass log_taus (mode learned) or taus (mode kyutai), not both")
    total = tau_total = None
    info: dict[str, Tensor] = {}
    weight_sum, n_maps = 0.0, 0
    for key, g in gen_feats.items():
        w = key_weight(key, key_weights)
        if w == 0:
            continue
        weight_sum, n_maps = weight_sum + w, n_maps + 1
        p, n = pos_feats[key], neg_feats.get(key) if neg_feats is not None else None
        if max_locations and g.shape[2] > max_locations:
            idx = torch.randperm(g.shape[2], device=g.device)[:max_locations]
            g, p, n = g[:, :, idx], p[:, :, idx], None if n is None else n[:, :, idx]
        B, L = g.shape[0], g.shape[2]
        g, p = _flatten_locations(g), _flatten_locations(p)
        wn = tau_loss = None
        if n is not None:
            n = _flatten_locations(n)
            if weight_neg is not None:
                wn = weight_neg[:, None, :].expand(B, L, -1).reshape(B * L, -1)
        if log_taus is not None:
            loss, tau_loss, kinfo = learned_tau_drift_loss(g, p, log_taus[key], n, weight_neg=wn)
            kinfo["tau"] = log_taus[key].detach().exp()
        elif taus is not None:
            tau = taus if isinstance(taus, Tensor) else taus[key]
            loss, tau_loss, kinfo = kyutai_drift_loss(g, p, tau, n, weight_neg=wn, affinity_floor=affinity_floor)
            kinfo["tau"] = tau.detach().reshape(())
        else:
            loss, kinfo = drift_loss(g, p, n, weight_neg=wn, temperatures=temperatures, affinity_floor=affinity_floor)
        if tau_loss is not None:
            tau_total = tau_loss if tau_total is None else tau_total + tau_loss
        loss = loss.mean() * w
        total = loss if total is None else total + loss
        for k, v in kinfo.items():
            info[f"{k}/{key}"] = v
    if reduce == "mean":
        total = total / weight_sum
        tau_total = None if tau_total is None else tau_total / n_maps
    if tau_total is not None:
        info["tau_loss"] = tau_total.detach()
        total = total + tau_total
    return total, info
