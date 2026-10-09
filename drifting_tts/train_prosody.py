"""Train a stochastic prosody predictor on the prosody targets of a trained TTS model (``train-prosody``, #39).

Inputs come from ``prosody-cache`` (MAS durations and token pitch of every utterance under the frozen model) and the
frozen text encoder of the same model (its hidden states, run in eval mode). ``net.kind`` picks the objective of
:class:`~drifting_tts.models.prosody_net.ProsodyNet`: ``drift`` (drifting loss at a learned temperature,
:func:`~drifting_tts.drift.kyutai_drift_loss`, on multi-scale feature maps of the prosody sequence), ``mse``
(regression baseline) or ``flow`` (conditional flow matching baseline). Token-level metrics on ``dev`` are logged
during training (:func:`token_metrics`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config, save_config
from .drift import key_weight, kyutai_drift_loss
from .models.prosody_net import ProsodyPredictor, prosody_features, summary_features, word_index
from .text import PAD_ID
from .utils import EMA, count_params, lr_lambda, save_checkpoint, seed_everything


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/prosody_drift.yaml")
    p.add_argument("--workdir", default="runs/prosody")
    p.add_argument("--calibrate-only", action="store_true",
                   help="only (re-)measure the per-voice duration factors of <workdir>/prosody_ema.pt at "
                        "calibrate.temperature / calibrate.spread, and store that temperature as the preferred one")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.steps=1000")


# ----------------------------------------------------------------------------------------------------------- data
def interpolate_unvoiced(pitch: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Continuous token pitch contour: unvoiced tokens linearly interpolated from the voiced ones (edges held)."""
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return np.zeros_like(pitch)
    return np.interp(np.arange(len(pitch)), idx, pitch[idx]).astype(np.float32)


class ProsodyData:
    """The utterances of one or more splits of a prosody cache, as padded batches."""

    def __init__(self, cache: dict, splits: tuple[str, ...], speakers: list[int] | None = None):
        t = cache["tokens"]
        self.utts = [u for u in cache["utts"] if u["split"] in splits and (speakers is None or u["spk"] in speakers)]
        ids, dur, pitch, voiced = (t[k].numpy() for k in ("ids", "dur", "pitch", "voiced"))
        logw_det, pitch_det = t["logw_det"].numpy(), t["pitch_det"].numpy()
        wf = cache.get("word_feats")
        self.items = []
        for u in self.utts:
            s = slice(u["start"], u["start"] + u["n"])
            v = voiced[s] > 0
            self.items.append({
                "ids": ids[s].astype(np.int64), "spk": u["spk"], "dur": dur[s].astype(np.float32),
                "pitch": pitch[s], "voiced": v, "pcont": interpolate_unvoiced(pitch[s], v),
                "logw_det": logw_det[s], "pitch_det": pitch_det[s], "repeat": u["repeat"],
            })
            if wf is not None:
                self.items[-1]["wf"] = wf[u["word_start"]: u["word_start"] + u["n_words"]]

    def __len__(self) -> int:
        return len(self.items)

    def subset(self, idx) -> ProsodyData:
        out = object.__new__(ProsodyData)
        out.utts, out.items = [self.utts[i] for i in idx], [self.items[i] for i in idx]
        return out

    def batch(self, idx: list[int], device) -> dict:
        items = [self.items[i] for i in idx]
        B, N = len(items), max(len(it["ids"]) for it in items)
        out = {"ids": torch.full((B, N), PAD_ID, dtype=torch.long),
               "len": torch.tensor([len(it["ids"]) for it in items]), "spk": torch.tensor([it["spk"] for it in items])}
        for k, dt in (("dur", torch.float32), ("pitch", torch.float32), ("voiced", torch.bool),
                      ("pcont", torch.float32), ("logw_det", torch.float32), ("pitch_det", torch.float32)):
            x = torch.zeros(B, N, dtype=dt)
            for i, it in enumerate(items):
                x[i, : len(it["ids"])] = torch.from_numpy(np.asarray(it[k]))
            out[k] = x
        for i, it in enumerate(items):
            out["ids"][i, : len(it["ids"])] = torch.from_numpy(it["ids"])
        out["word"] = word_index(out["ids"])
        out["mask"] = torch.arange(N)[None] < out["len"][:, None]
        n_words = int(out["word"].max()) + 1
        pin = torch.cuda.is_available() and str(device).startswith("cuda")
        out = {k: (v.pin_memory() if pin else v).to(device, non_blocking=True) for k, v in out.items()}
        out["n_words"] = n_words
        out["dur"] = out["dur"].clamp_min(1) * out["mask"]
        if "wf" in items[0]:  # contextual word features broadcast to the tokens [B, D, N]
            word = word_index(out["ids"]).cpu()
            wt = torch.zeros(B, N, items[0]["wf"].shape[1])
            for i, it in enumerate(items):
                n = len(it["ids"])
                wt[i, :n] = it["wf"].float()[word[i, :n].clamp_max(it["wf"].shape[0] - 1)]
            out["word_tok"] = wt.transpose(1, 2).to(device, non_blocking=True)
        return out

    def batches(self, batch_size: int, seed: int, bucket: int = 2048):
        """Infinite stream of length-bucketed, shuffled batches; utterances listed ``repeat`` times (as in training)."""
        rng = np.random.default_rng(seed)
        order = np.concatenate([np.full(it["repeat"], i) for i, it in enumerate(self.items)])
        while True:
            rng.shuffle(order)
            batches = []
            for s in range(0, len(order), bucket):
                chunk = sorted(order[s: s + bucket], key=lambda i: len(self.items[i]["ids"]))
                batches += [chunk[j: j + batch_size] for j in range(0, len(chunk) - batch_size + 1, batch_size)]
            rng.shuffle(batches)
            yield from batches


# ------------------------------------------------------------------------------------------------- model glue
@torch.no_grad()
def encode(tts, pred: ProsodyPredictor, b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen encoder -> condition ``[B, C, N]``, standardised regressor predictions ``[B, 2, N]``, mask."""
    h, _, logw, x_mask = tts.encoder(b["ids"], b["len"], b["spk"])
    word_tok = b.get("word_tok") if pred.net.word_dim else None
    cond, base = ProsodyPredictor.condition(tts, h, x_mask, b["spk"], logw, pred.stats, word_tok)
    return cond, base, x_mask


def target(pred: ProsodyPredictor, b: dict) -> torch.Tensor:
    """Standardised ground truth ``[B, 2, N]``: log MAS duration and the continuous token pitch contour."""
    return pred.stats.norm(torch.log(b["dur"].clamp_min(1)), b["pcont"]) * b["mask"][:, None]


@torch.no_grad()
def set_stats(pred: ProsodyPredictor, data: ProsodyData, device, n: int = 4000, seed: int = 0) -> None:
    """Channel and summary-feature statistics from (a sample of) the training targets."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), min(n, len(data)), replace=False)
    ld = np.concatenate([np.log(np.maximum(data.items[i]["dur"], 1)) for i in idx])
    pc = np.concatenate([data.items[i]["pcont"] for i in idx])
    pred.stats.seq.copy_(torch.tensor([[ld.mean(), ld.std()], [pc.mean(), pc.std()]]))
    words, utts = [], []
    for s in range(0, len(idx), 256):
        b = data.batch(list(idx[s: s + 256]), device)
        y = target(pred, b)[:, None]
        w, u, wv = summary_features(y, b["mask"], b["word"], pred.stats, standardize=False, n_words=b["n_words"])
        words.append(w[:, 0][wv].cpu())
        utts.append(u[:, 0, 0].cpu())
    w, u = torch.cat(words), torch.cat(utts)
    pred.stats.word.copy_(torch.stack([w.mean(0), w.std(0).clamp_min(1e-3)]).to(device))
    pred.stats.utt.copy_(torch.stack([u.mean(0), u.std(0).clamp_min(1e-3)]).to(device))
    if pred.net.word_dim:
        wf = torch.cat([data.items[i]["wf"].float() for i in idx])
        pred.stats.wfeat.copy_(torch.stack([wf.mean(0), wf.std(0).clamp_min(1e-3)]).to(device))


def drift_maps_loss(gen: dict, pos: dict, valid: dict, tau: torch.Tensor,
                    key_weights: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Kyutai drifting loss on every feature map (``reduce: mean`` over maps), one drift problem per valid location.

    Invalid locations (padding) stay in the batch with weight 0, so no host sync is needed to select rows."""
    total, tau_total, wsum, n = 0.0, 0.0, 0.0, 0
    info = {}
    for k, g in gen.items():
        w = key_weight(k, key_weights)
        if w == 0:
            continue
        B, S, L, D = g.shape
        rw = valid[k].reshape(B * L).float()
        g_rows = g.permute(0, 2, 1, 3).reshape(B * L, S, D)
        p_rows = pos[k].permute(0, 2, 1, 3).reshape(B * L, -1, D)
        loss, tau_loss, kinfo = kyutai_drift_loss(g_rows, p_rows, tau, row_weight=rw)
        total = total + w * (loss * rw).sum() / rw.sum().clamp_min(1)
        tau_total = tau_total + tau_loss
        wsum, n = wsum + w, n + 1
        info[f"p_data/{k}"] = kinfo["p_data"]
    return total / wsum, tau_total / n, info


def training_loss(pred: ProsodyPredictor, tts, b: dict, cfg, tau: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
    net, kind = pred.net, pred.net.kind
    cond, base, x_mask = encode(tts, pred, b)
    y = target(pred, b)
    mask = b["mask"]
    B, _, N = cond.shape
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cond.is_cuda)
    metrics = {}
    vt = b["voiced"].float()
    if kind == "drift":
        G = cfg.drift.gen_per_cond
        rep = lambda x: x.repeat_interleave(G, 0)
        z_tok = torch.randn(B * G, net.noise_tok, N, device=cond.device)
        z_glob = torch.randn(B * G, net.noise_glob, device=cond.device)
        with amp:
            out = net(rep(cond), rep(x_mask), z_tok, z_glob)
        out = out.float()
        y_gen = (rep(base) + out[:, :2]) * rep(x_mask)
        kw = dict(pools=tuple(cfg.drift.pools), windows=tuple(cfg.drift.windows), n_words=b["n_words"])
        f_gen, valid = prosody_features(y_gen.view(B, G, 2, N), mask, b["word"], pred.stats, **kw)
        with torch.no_grad():
            f_pos, _ = prosody_features(y[:, None], mask, b["word"], pred.stats, **kw)
        l_main, l_tau, info = drift_maps_loss(f_gen, f_pos, valid, tau, cfg.drift.get("key_weights"))
        logit = out[:, 2]
        l_v = (F.binary_cross_entropy_with_logits(logit, rep(vt), reduction="none") * rep(mask)).sum() / (
            rep(mask).sum())
        loss = l_main + l_tau + cfg.loss.voiced * l_v
        with torch.no_grad():
            yg = y_gen.view(B, G, 2, N)
            mf = mask[:, None].float()
            metrics["spread"] = (yg.std(1) * mf).sum() / mf.sum() / 2
            metrics["centroid_mse"] = (((yg.mean(1) - y) ** 2) * mf).sum() / mf.sum() / 2
            metrics["tau"], metrics["tau_loss"] = tau.detach(), l_tau.detach()
            metrics["p_data"] = torch.stack(list(info.values())).mean()
            for k in ("tok", "utt"):
                metrics[f"p_data_{k}"] = info[f"p_data/{k}"]
    elif kind == "mse":
        with amp:
            out = net(cond, x_mask).float()
        y_hat = base + out[:, :2]
        mf = mask[:, None].float()
        l_main = (((y_hat - y) ** 2) * mf).sum() / mf.sum() / 2
        l_v = (F.binary_cross_entropy_with_logits(out[:, 2], vt, reduction="none") * mask).sum() / mask.sum()
        loss = l_main + cfg.loss.voiced * l_v
    else:  # flow matching on the residual over the regressors, K time samples per utterance
        K = cfg.flow.get("t_per_cond", 4)
        rep = lambda x: x.repeat_interleave(K, 0)
        x1 = rep((y - base) * x_mask)
        x0 = torch.randn_like(x1) * rep(x_mask)
        t = torch.rand(B * K, device=cond.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        with amp:
            out = net(rep(cond), rep(x_mask), x_t=xt, t=t).float()
        mf = rep(mask[:, None].float())
        l_main = (((out[:, :2] - (x1 - x0)) ** 2) * mf).sum() / mf.sum() / 2
        l_v = (F.binary_cross_entropy_with_logits(out[:, 2], rep(vt), reduction="none") * rep(mask)).sum() / (
            rep(mask).sum())
        loss = l_main + cfg.loss.voiced * l_v
    metrics.update(loss=loss.detach(), main=l_main.detach(), voiced_bce=l_v.detach())
    return loss, metrics


# ---------------------------------------------------------------------------------------------------- metrics
@torch.no_grad()
def sample_split(pred: ProsodyPredictor | None, tts, data: ProsodyData, seeds: list[int], temperature: float,
                 device, batch_size: int = 50, apply_scales: bool = False, spread: float = 1.0) -> list[dict]:
    """Per utterance: ``frames`` ``[K, n]`` (integer), ``pitch`` (token pitch as fed to the generator, 0 unvoiced),
    ``pcont`` (the continuous contour) and ``voiced``, for ``K`` seeds. ``pred=None``: the TTS model's own
    regressors (ceil of ``exp(logw)`` times its per-voice ``duration_scales``, as at inference)."""
    out = [None] * len(data)
    if spread != 1.0:  # every utterance is sampled 16 times in one batch
        batch_size = max(1, batch_size // 16)
    for s in range(0, len(data), batch_size):
        idx = list(range(s, min(s + batch_size, len(data))))
        b = data.batch(idx, device)
        mask = b["mask"]
        if pred is None:
            sc = torch.tensor([tts.duration_scales.get(int(k), tts.duration_scale) for k in b["spk"]], device=device)
            fr = torch.ceil(torch.exp(b["logw_det"]) * sc[:, None]).clamp_min(0) * mask
            res = [(fr, b["pitch_det"], b["pitch_det"], torch.ones_like(mask))] * len(seeds)
        else:
            cond, base, x_mask = encode(tts, pred, b)
            res = []
            for seed in seeds:
                g = torch.Generator(device=device).manual_seed(seed)
                y, vlogit = pred.sample(cond, base, x_mask, temperature, generator=g, spread=spread)
                sc = 1.0
                if apply_scales:
                    sc = torch.tensor([pred.duration_scales.get(int(k), 1.0) for k in b["spk"]], device=device)[:, None]
                fr, pitch = pred.frames_and_pitch(y, vlogit, x_mask, sc)
                _, pc = pred.stats.denorm(y)
                res.append((fr, pitch[:, 0], pc, vlogit > 0))
        for j, i in enumerate(idx):
            n = int(b["len"][j])
            out[i] = {k: torch.stack([r[c][j, :n] for r in res]).float().cpu().numpy()
                      for c, k in enumerate(("frames", "pitch", "pcont", "voiced"))}
    return out


def _pair_mean(x: np.ndarray) -> np.ndarray:
    """Unbiased ``E|X - X'|`` per column over the ``K`` rows (distinct pairs only; 0 for one sample)."""
    K = x.shape[0]
    return np.abs(x[:, None] - x[None]).sum((0, 1)) / (K * (K - 1)) if K > 1 else np.zeros(x.shape[1:])


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    q = np.linspace(0.005, 0.995, 199)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def reversal_rate(p: np.ndarray, voiced: np.ndarray, threshold: float) -> float:
    """Local pitch reversals per letter: the letter-level pitch contour (mean over the voiced tokens of a character and
    the blank after it) changes direction, both steps larger than ``threshold`` (normalised log-F0 units)."""
    grp = np.maximum(np.arange(len(p)) - 1, 0) // 2
    n = grp.max() + 1
    num, den = np.zeros(n), np.zeros(n)
    np.add.at(num, grp, np.where(voiced, p, 0))
    np.add.at(den, grp, voiced.astype(float))
    lp = num[den > 0] / den[den > 0]
    d = np.diff(lp)
    if len(d) < 2:
        return 0.0
    big = np.abs(d) > threshold
    return float(((d[1:] * d[:-1] < 0) & big[1:] & big[:-1]).sum() / len(lp))


def token_metrics(samples: list[dict], data: ProsodyData, reversal_threshold: float = 0.077) -> dict:
    """Token / letter / word / utterance prosody statistics of sampled sequences against the recordings' targets.

    ``*_std_ratio``: within-utterance spread, predicted / ground truth (1 = as varied as the recordings);
    ``*_corr``: Pearson with the ground truth; ``*_crps``: continuous ranked probability score over the seeds
    (unbiased pair term; the MAE for one deterministic sample; lower is better, comparable between deterministic and
    stochastic models);
    ``w1_*``: 1-Wasserstein distance between pooled distributions; ``jitter_ratio``: mean |Δ pitch| between
    neighbouring voiced tokens, predicted / ground truth; ``reversal_rate``: local pitch reversals per voiced letter
    (:func:`reversal_rate`; threshold 0.077 normalised log-F0 = 0.5 semitone with v3.1's statistics; the recordings'
    value is ``reversal_rate_gt``); ``rate``: median predicted / recorded length;
    ``div_*``: standard deviation across seeds."""
    from .text import SYMBOL_TO_ID

    space = SYMBOL_TO_ID[" "]
    acc = {k: [] for k in ("p_std", "p_std_gt", "ld_std", "ld_std_gt", "let_std", "let_std_gt", "wd_std", "wd_std_gt",
                           "wp_std", "wp_std_gt", "p_corr", "ld_corr", "let_corr", "wd_corr", "p_crps", "ld_crps",
                           "jit", "jit_gt", "rev", "rev_gt", "rate", "div_p", "div_ld", "div_total", "div_mean_p",
                           "vacc")}
    pdev, pdev_gt, ld_all, ld_gt_all, ustd, ustd_gt = [], [], [], [], [], []
    for s, it in zip(samples, data.items):
        v = it["voiced"]
        ids = it["ids"]
        ld_gt = np.log(np.maximum(it["dur"], 1))
        p_gt = it["pitch"]
        fr = np.maximum(s["frames"], 1)
        ld = np.log(fr)  # [K, n]
        pc = s["pcont"]
        K = ld.shape[0]
        let_grp = np.maximum(np.arange(len(ids)) - 1, 0) // 2
        wd_grp = np.cumsum(ids == space)

        def group(x, grp):
            out = np.zeros(x.shape[:-1] + (grp.max() + 1,))
            if x.ndim == 1:
                np.add.at(out, grp, x)
            else:
                for k in range(x.shape[0]):
                    np.add.at(out[k], grp, x[k])
            return out

        let_gt, let = np.log(group(it["dur"], let_grp)), np.log(group(fr, let_grp))
        wcnt = group(np.ones(len(ids)), wd_grp)
        wd_gt, wd = np.log(group(it["dur"], wd_grp)), np.log(group(fr, wd_grp))
        vw = group(v.astype(float), wd_grp)
        wp_gt = group(np.where(v, p_gt, 0), wd_grp) / np.maximum(vw, 1)
        wp = group(np.where(v, pc, 0), wd_grp) / np.maximum(vw, 1)
        wsel = (vw > 0) & (wcnt > 2)
        for k in range(K):
            if v.sum() > 2:
                acc["p_std"].append(pc[k][v].std())
                acc["p_corr"].append(np.corrcoef(pc[k][v], p_gt[v])[0, 1] if pc[k][v].std() > 0 else 0.0)
                d = np.abs(np.diff(pc[k]))[v[1:] & v[:-1]]
                acc["jit"].append(d.mean() if len(d) else 0.0)
                acc["rev"].append(reversal_rate(pc[k], v, reversal_threshold))
                pdev.append(pc[k][v] - pc[k][v].mean())
                ustd.append(pc[k][v].std())
            acc["ld_std"].append(ld[k].std())
            acc["ld_corr"].append(np.corrcoef(ld[k], ld_gt)[0, 1] if ld[k].std() > 0 else 0.0)
            acc["let_std"].append(let[k].std())
            acc["let_corr"].append(np.corrcoef(let[k], let_gt)[0, 1] if let[k].std() > 0 else 0.0)
            acc["wd_std"].append(wd[k].std())
            acc["wd_corr"].append(np.corrcoef(wd[k], wd_gt)[0, 1] if len(wd_gt) > 1 and wd[k].std() > 0 else 0.0)
            if wsel.sum() > 1:
                acc["wp_std"].append(wp[k][wsel].std())
            acc["rate"].append(fr[k].sum() / it["dur"].sum())
            acc["vacc"].append((s["voiced"][k] > 0.5).astype(bool).__eq__(v).mean())
            ld_all.append(ld[k])
        if v.sum() > 2:
            acc["p_std_gt"].append(p_gt[v].std())
            d = np.abs(np.diff(it["pcont"]))[v[1:] & v[:-1]]
            acc["jit_gt"].append(d.mean() if len(d) else 0.0)
            acc["rev_gt"].append(reversal_rate(it["pcont"], v, reversal_threshold))
            pdev_gt.append(p_gt[v] - p_gt[v].mean())
            ustd_gt.append(p_gt[v].std())
            # CRPS over the seeds: E|X - y| - 0.5 E|X - X'|
            x = pc[:, v]
            acc["p_crps"].append((np.abs(x - p_gt[v]).mean(0) - 0.5 * _pair_mean(x)).mean())
        acc["ld_std_gt"].append(ld_gt.std())
        acc["let_std_gt"].append(let_gt.std())
        acc["wd_std_gt"].append(wd_gt.std())
        if wsel.sum() > 1:
            acc["wp_std_gt"].append(wp_gt[wsel].std())
        acc["ld_crps"].append((np.abs(ld - ld_gt).mean(0) - 0.5 * _pair_mean(ld)).mean())
        ld_gt_all.append(ld_gt)
        if K > 1:
            acc["div_p"].append(pc[:, v].std(0).mean() if v.any() else 0.0)
            acc["div_ld"].append(ld.std(0).mean())
            acc["div_total"].append(np.log(fr.sum(1)).std())
            acc["div_mean_p"].append(pc[:, v].mean(1).std() if v.any() else 0.0)
    m = lambda k: float(np.mean(acc[k])) if acc[k] else float("nan")
    return {
        "p_std_ratio": m("p_std") / m("p_std_gt"), "ld_std_ratio": m("ld_std") / m("ld_std_gt"),
        "letter_std_ratio": m("let_std") / m("let_std_gt"), "word_dur_std_ratio": m("wd_std") / m("wd_std_gt"),
        "word_p_std_ratio": m("wp_std") / m("wp_std_gt"),
        "p_corr": m("p_corr"), "ld_corr": m("ld_corr"), "letter_corr": m("let_corr"), "word_dur_corr": m("wd_corr"),
        "p_crps": m("p_crps"), "ld_crps": m("ld_crps"),
        "w1_pdev": _w1(np.concatenate(pdev), np.concatenate(pdev_gt)),
        "w1_ld": _w1(np.concatenate(ld_all), np.concatenate(ld_gt_all)),
        "w1_utt_pstd": _w1(np.array(ustd), np.array(ustd_gt)),
        "jitter_ratio": m("jit") / m("jit_gt"), "reversal_rate": m("rev"), "reversal_rate_gt": m("rev_gt"),
        "rate": float(np.median(acc["rate"])), "voicing_acc": m("vacc"),
        "div_p": m("div_p"), "div_ld": m("div_ld"), "div_total": m("div_total"), "div_mean_p": m("div_mean_p"),
    }


@torch.no_grad()
def calibrate(pred: ProsodyPredictor, tts, cache: dict, device, temperature: float, speakers=(722, 389, 323),
              num: int = 300, seed: int = 0, spread: float = 1.0) -> dict[int, float]:
    """Per-voice duration factors of the sampler: the median recorded / sampled length over ``num`` *training*
    utterances of each voice (never the evaluation splits), as ``calibrate-durations`` does for the regressors."""
    rng = np.random.default_rng(seed)
    scales = {}
    for spk in speakers:
        data = ProsodyData(cache, ("train",), [spk])
        if len(data) < 20:
            continue
        sub = data.subset(rng.choice(len(data), min(num, len(data)), replace=False))
        samples = sample_split(pred, tts, sub, [seed], temperature, device, spread=spread)
        r = [it["dur"].sum() / s["frames"][0].sum() for s, it in zip(samples, sub.items)]
        scales[int(spk)] = float(np.median(r))
        print(f"speaker {spk}: recorded / sampled length median {scales[spk]:.3f} (p10 {np.percentile(r, 10):.3f}, "
              f"p90 {np.percentile(r, 90):.3f}) over {len(sub)} training utterances, T={temperature:g}", flush=True)
    return scales


# ------------------------------------------------------------------------------------------------------- train
def load_frozen_tts(path: str, device):
    from .train import load_tts

    tts, _, _ = load_tts(path, device)
    if not tts.pitch_enabled:
        raise SystemExit("the prosody predictor needs a pitch-conditioned TTS model")
    return tts.eval().requires_grad_(False)


def run(args) -> None:
    from torch.utils.tensorboard import SummaryWriter

    cfg = load_config(args.config, args.overrides)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    save_config(cfg, work / "config.yaml")
    tc = cfg.train
    seed_everything(tc.seed)
    device = "cuda" if torch.cuda.is_available() and not tc.get("cpu", False) else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    tts = load_frozen_tts(cfg.tts, device)
    cache = torch.load(cfg.cache, map_location="cpu", weights_only=False)
    cal = cfg.get("calibrate", {})
    if args.calibrate_only:
        path = work / "prosody_ema.pt"
        pred = ProsodyPredictor.load(path, device, tts=tts)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck["duration_scales"] = calibrate(pred, tts, cache, device, cal.get("temperature", 1.0),
                                          spread=cal.get("spread", 1.0))
        ck["calibrate_temperature"], ck["calibrate_spread"] = cal.get("temperature", 1.0), cal.get("spread", 1.0)
        ck["temperature"] = cal.get("temperature", 1.0)  # the default of Synthesizer(prosody_temperature=None)
        save_checkpoint(path, **ck)
        print(f"stored duration_scales {ck['duration_scales']} in {path}")
        return
    train = ProsodyData(cache, ("train",))
    dev = ProsodyData(cache, (cfg.get("eval_split", "dev"),))
    if not len(dev):
        print(f"no '{cfg.get('eval_split', 'dev')}' utterances in the cache: no evaluation during training", flush=True)
    cond_dim = tts.encoder.d + tts.encoder.spk.embedding_dim + 2
    if cfg.net.get("word_dim", 0):
        if "word_feats" not in cache:
            raise SystemExit(f"net.word_dim > 0 needs word features in {cfg.cache} (prosody-cache --word-model)")
        cfg.net.word_dim, cfg.net.word_model = int(cache["word_feats"].shape[1]), cache["word_model"]
    pred = ProsodyPredictor(cfg.net.to_dict(), cond_dim).to(device)
    pred.flow_steps = int(cfg.get("flow", {}).get("steps", 8))
    set_stats(pred, train, device)
    print(f"ProsodyNet ({pred.kind}) {count_params(pred.net):.2f}M params; {len(train)} training utterances; "
          f"stats seq {pred.stats.seq.tolist()}", flush=True)
    tau = torch.nn.Parameter(torch.tensor(float(cfg.drift.tau_init), device=device)) if pred.kind == "drift" else None
    groups = [{"params": list(pred.net.parameters())}]
    if tau is not None:
        groups.append({"params": [tau], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=tc.lr, betas=(0.9, 0.999), weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup, tc.steps, tc.schedule))
    ema = EMA(pred.net, tc.ema_decay)
    writer = SummaryWriter(work / "tb")
    step = 0
    if (work / "last.pt").exists():
        ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
        pred.net.load_state_dict(ck["net"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        pred.stats.load_state_dict(ck["stats"])
        if tau is not None:
            tau.data.fill_(ck["tau"])
        step = ck["step"]
        print(f"resumed from step {step}", flush=True)
    stream = train.batches(tc.batch_size, tc.seed + step)
    pred.net.train()
    t0, agg = time.time(), {}
    history = []

    def save(path: Path, full: bool) -> None:
        state = dict(ema=ema.model.state_dict(), stats=pred.stats.state_dict(), net_cfg=pred.net_cfg,
                     cond_dim=cond_dim, step=step, tts=str(cfg.tts), config=cfg.to_dict(),
                     tau=None if tau is None else float(tau.detach()), flow_steps=pred.flow_steps,
                     duration_scales=pred.duration_scales)
        if full:
            state.update(net=pred.net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
        save_checkpoint(path, **state)

    while step < tc.steps:
        b = train.batch(next(stream), device)
        loss, metrics = training_loss(pred, tts, b, cfg, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(pred.net.parameters(), tc.grad_clip)
        if tau is not None:
            torch.nn.utils.clip_grad_norm_([tau], tc.grad_clip)
        opt.step()
        sched.step()
        if tau is not None:
            with torch.no_grad():
                tau.clamp_(min=1e-3)
        ema.update(pred.net)
        step += 1
        metrics["grad_norm"] = gn
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + (v.detach() if torch.is_tensor(v) else v)
        if step % tc.log_every == 0:
            agg = {k: float(v) / tc.log_every for k, v in agg.items()}
            for k, v in agg.items():
                writer.add_scalar(f"train/{k}", v, step)
            print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in agg.items())
                  + f" ({tc.log_every / (time.time() - t0):.2f} it/s)", flush=True)
            agg, t0 = {}, time.time()
        if len(dev) and (step % tc.eval_every == 0 or step == tc.steps):
            live = pred.net
            pred.net = ema.model
            temps = cfg.get("eval_temperatures", [1.0]) if pred.kind != "mse" else [1.0]
            for T in temps:
                seeds = [0] if pred.kind == "mse" else [0, 1, 2, 3]
                res = token_metrics(sample_split(pred, tts, dev, seeds, T, device), dev)
                for k, v in res.items():
                    writer.add_scalar(f"dev_T{T:g}/{k}", v, step)
                keys = ("p_std_ratio", "ld_std_ratio", "word_p_std_ratio", "word_dur_std_ratio", "p_corr",
                        "p_crps", "ld_crps", "w1_pdev", "w1_ld", "jitter_ratio", "rate", "div_p", "div_total")
                print(f"  dev T={T:g} " + " ".join(f"{k}={res[k]:.3f}" for k in keys), flush=True)
                history.append({"step": step, "temperature": T, **res})
            pred.net = live
            (work / "dev_history.json").write_text(json.dumps(history, indent=1))
        if step % tc.save_every == 0 or step == tc.steps:
            save(work / "last.pt", True)
            save(work / "prosody_ema.pt", False)
    if cal.get("enabled", True):
        pred.net = ema.model
        pred.duration_scales = calibrate(pred, tts, cache, device, cal.get("temperature", 1.0),
                                         spread=cal.get("spread", 1.0))
        save(work / "prosody_ema.pt", False)
    print(f"done: {work / 'prosody_ema.pt'}", flush=True)
