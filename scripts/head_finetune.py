"""Which objective makes ARDrift's head realistic? A controlled experiment on the head alone (docs/HEAD_REALISM.md).

``scripts/c2st_mel.py`` shows that a classifier tells a single teacher-forced patch of the head from the recorded one
(English 200k model: 81%), and ``scripts/drift_calibration.py`` where: the samples are too smooth in time (spread-skill
0.80 in the patch's fastest temporal frequency), although the head is calibrated overall (0.98).

This script fine-tunes **only the head** of a trained model, on backbone states computed once (text encoder and
backbone frozen, teacher-forced on the recording with the training's history noise). A run is a choice of loss
(``--loss``: the training's drifting field with one learned temperature per map, or the energy score, a proper
scoring rule: :func:`drifting_tts.drift.energy_score`) and of the maps it is computed on (``--maps``):

* ``patch`` / ``frame`` / ``detail``: the training's (:func:`drifting_tts.models.ar.ar_features`);
* ``delta``: the change from each frame to the next, the last recorded frame before the patch included (one drift
  problem per pair of neighbouring frames): how the patch moves and how it joins the history;
* ``delta_detail``: the same on the harmonic detail;
* ``tile``: time-frequency tiles of the patch (all its frames x ``--tile-bins`` mel bins, half overlapping): local
  joint structure in few dimensions, where an L2 kernel still resolves it;
* ``tile_detail``: the same on the harmonic detail;
* ``delta2``: the second difference along time (the change of the change), the last recorded frame included: whether
  the patch moves in a straight line or as speech does;
* ``ctile<B>`` / ``ctile<B>_detail``: tiles of ``B`` mel bins over the last recorded frame and the patch's frames
  (``ctile2``: 10-d, how two neighbouring bins move through the join and the patch; ``ctile25``: 125-d);
* ``wdct``: the patch's 2-D cosine transform (time x mel), one map per temporal frequency (the mean of the frames,
  their trend, ..., their alternation), every coefficient divided by the spread of the recordings around the head's
  mean in that coefficient (measured once, before training). An L2 kernel shrinks the directions that carry little of
  its distance (``tests/test_drift.py``: on data with the patch's energy profile the true distribution is not a fixed
  point of the one-positive field, and with one kernel per band it nearly is); in whitened bands no direction is
  small;
* ``wpatch``: the same whitened coefficients as one 400-d vector: the patch in a space where every direction
  carries the same share of the distance, as the normalised codec latent of Kyutai's recipe does;
* ``mae``: the feature maps of a frozen 2-D Mel-MAE (``--mae``, ``drifting-tts train-mae`` with
  ``configs/mae2d.yaml``) on the window of the last ``--context`` recorded patches followed by the patch, at the
  locations of the patch: the paper's recipe (a kernel in the features of a masked autoencoder) on the head's own
  window (#97).

Every run starts from the same head and sees the same rows; the control is the training's own loss and three maps.
The result is the head alone (``head.pt``: the EMA weights), which ``c2st_mel.py --head`` and
``drift_calibration.py --head`` load over the model.

    python scripts/head_finetune.py --model ardrift_en.pt --data data/en_ft --cache rows.pt --out runs/head_control
    python scripts/head_finetune.py --model ardrift_en.pt --data data/en_ft --cache rows.pt --out runs/head_energy \
        --maps patch frame detail wdct --loss energy
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

import torch

from drifting_tts.alignment import sequence_mask
from drifting_tts.data import MelDataset, collate
from drifting_tts.drift import feature_drift_loss, feature_energy_loss
from drifting_tts.models.ar import ar_features, mae_features, patchify
from drifting_tts.models.mae import spectral_detail
from drifting_tts.models.text_encoder import align
from drifting_tts.train import load_tts
from drifting_tts.utils import dct_matrix

MAPS = ("patch", "frame", "detail", "delta", "delta_detail", "tile", "tile_detail", "delta2", "mae", "wdct",
        "wpatch")
CTILE = re.compile(r"ctile(\d+)(_detail)?")


def map_name(name: str) -> str:
    """argparse type of ``--maps``: a name of :data:`MAPS` or ``ctile<bins>[_detail]``."""
    if name not in MAPS and not CTILE.fullmatch(name):
        raise argparse.ArgumentTypeError(f"unknown kernel map {name!r}: one of {', '.join(MAPS)} or "
                                         "ctile<bins>[_detail]")
    return name


def dct2(frames: torch.Tensor) -> torch.Tensor:
    """2-D cosine transform of patches ``[..., k, c]`` (time, then mel): ``[..., k, c]`` coefficients."""
    k, c = frames.shape[-2:]
    return dct_matrix(k, frames.device) @ frames @ dct_matrix(c, frames.device).T


@torch.no_grad()
def residual_scale(head, rows: dict[str, torch.Tensor], k: int, c: int, samples: int, dev: str, n: int = 4096,
                   seed: int = 0) -> torch.Tensor:
    """RMS of the recording around the head's mean, per coefficient of the patch's 2-D cosine transform ``[k, c]``:
    the spread the head's samples should have in that coefficient (``n`` random rows, ``samples`` draws each)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(rows["state"].shape[0], generator=g)[:n]
    sq = torch.zeros(k, c, device=dev)
    for part in idx.split(512):
        s, target, base = (rows[name][part].to(dev).float() for name in ("state", "target", "base"))
        z = torch.randn(len(part) * samples, head.noise_dim, device=dev)
        x = base[:, None] + head(s.repeat_interleave(samples, 0), z).float().view(len(part), samples, -1)
        err = dct2(target.view(-1, k, c)) - dct2(x.view(len(part), samples, k, c)).mean(1)
        sq += err.pow(2).sum(0)
    return (sq / len(idx) / (1 + 1 / samples)).sqrt().clamp_min(1e-4)


def kernel_features(x: torch.Tensor, prev: torch.Tensor, k: int, c: int, maps: tuple[str, ...],
                    tile_bins: int = 10, mae=None, scale: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Kernel maps ``[R, S, L, D]`` of patches ``x`` ``[R, S, k * c]`` whose history ends with the recorded patches
    ``prev`` ``[R, n * k * c]`` (the last ``n`` of them, oldest first; see the module docstring). ``scale``
    ``[k, c]`` (:func:`residual_scale`): the divisor of the ``wdct`` maps' coefficients."""
    R, S, _ = x.shape
    base = ar_features(x, k, c, detail=True)
    out = {m: base[m] for m in ("patch", "frame", "detail") if m in maps}
    fr = x.view(R, S, k, c)
    hist = prev.view(R, -1, c)  # [R, n * k, c] frames
    if "wdct" in maps or "wpatch" in maps:  # whitened coefficients of the patch's 2-D cosine transform
        coef = dct2(fr.float()) / scale
        if "wdct" in maps:  # one map per temporal frequency
            out.update({f"wdct{j}": coef[:, :, j, None] for j in range(k)})
        if "wpatch" in maps:  # the whole whitened patch as one vector (as Kyutai's normalised latent)
            out["wpatch"] = coef.reshape(R, S, 1, k * c)
    if "mae" in maps:
        window = torch.cat([hist[:, None].expand(-1, S, -1, -1), fr], 2)  # [R, S, n * k + k, c]
        feats = mae_features(mae, window.reshape(R * S, -1, c).transpose(1, 2), k)
        out.update({name: f.view(R, S, *f.shape[1:]) for name, f in feats.items()})
    if any(m.startswith(("delta", "tile", "ctile")) for m in maps):
        last = hist[:, None, -1:].expand(-1, S, -1, -1)
        seq = torch.cat([last, fr], 2)  # [R, S, k + 1, c]: the last recorded frame, then the patch
        det = spectral_detail(seq.reshape(R * S, k + 1, c).transpose(1, 2)).transpose(1, 2).reshape(R, S, k + 1, c)
        if "delta" in maps:
            out["delta"] = seq[:, :, 1:] - seq[:, :, :-1]
        if "delta_detail" in maps:
            out["delta_detail"] = det[:, :, 1:] - det[:, :, :-1]
        for name, src in (("tile", fr), ("tile_detail", det[:, :, 1:])):
            if name in maps:  # [R, S, k, L, bins] -> [R, S, L, k * bins]
                t = src.unfold(3, tile_bins, tile_bins // 2)
                out[name] = t.permute(0, 1, 3, 2, 4).reshape(R, S, t.shape[3], k * tile_bins)
        if "delta2" in maps:
            d = seq[:, :, 1:] - seq[:, :, :-1]
            out["delta2"] = d[:, :, 1:] - d[:, :, :-1]
        for name in maps:
            m = CTILE.fullmatch(name)
            if m:  # the last recorded frame and the patch: [R, S, k + 1, L, bins] -> [R, S, L, (k + 1) * bins]
                bins = int(m.group(1))
                t = (det if m.group(2) else seq).unfold(3, bins, max(bins // 2, 1))
                out[name] = t.permute(0, 1, 3, 2, 4).reshape(R, S, t.shape[3], (k + 1) * bins)
    return out


@torch.no_grad()
def cache_states(model, ds: MelDataset, dev: str, noise_max: float, batch: int, seed: int,
                 context: int = 1) -> dict[str, torch.Tensor]:
    """The head's training rows of a dataset, the rest of the model frozen: backbone states (teacher-forced on the
    recording plus noise of a random level per utterance), the recorded patch, the aligned prior mean and the
    ``context`` recorded patches before it (oldest first). Whole patches after the first ``context`` steps only; half
    precision, on ``dev``."""
    k, C = model.patch, model.n_mels
    g = torch.Generator(device=dev).manual_seed(seed)
    order = sorted(range(len(ds)), key=ds.frames)
    rows: dict[str, list] = {"state": [], "target": [], "base": [], "prev": []}
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    for b0 in range(0, len(order), batch):
        b = collate([ds[i] for i in order[b0: b0 + batch]])
        text, text_len, y, y_len, spk = (b[n].to(dev) for n in ("text", "text_len", "mel", "mel_len", "spk"))
        y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
        with amp:
            h, mu, _, x_mask = model.encoder(text, text_len, spk)
        h, mu = h.float(), mu.float()
        attn, _ = align(mu, x_mask, y, y_mask)
        cond, yp = model.pad_frames(model.frame_condition(h, mu, attn)), model.pad_frames(y)
        noise = torch.rand(y.shape[0], device=dev, generator=g) * noise_max
        torch.manual_seed(seed + b0)  # teacher_forced draws its history noise from the global generator
        with amp:
            s = model.teacher_forced(cond, yp, spk, noise).float()
        S = s.shape[1]
        t = torch.arange(S, device=dev)[None]
        keep = ((t + 1) * k <= y_len[:, None]) & (t >= context)
        target = patchify(yp, k)
        before = [torch.cat([target[:, :1].expand(-1, n, -1), target[:, :-n]], 1) for n in range(context, 0, -1)]
        rows["state"].append(s[keep].half())
        rows["target"].append(target[keep].half())
        rows["base"].append(patchify(cond[:, :C], k)[keep].half())
        rows["prev"].append(torch.cat(before, -1)[keep].half())
        if (b0 // batch) % 20 == 0:
            print(f"states: {min(b0 + batch, len(order))}/{len(order)} utterances", flush=True)
    return {name: torch.cat(v) for name, v in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps", nargs="+", default=["patch", "frame", "detail"], type=map_name)
    ap.add_argument("--loss", choices=("kyutai", "energy", "both"), default="kyutai",
                    help="the drifting field with a learned temperature per map (the training's), the energy score "
                         "(a proper scoring rule: drifting_tts.drift.energy_score), or their sum")
    ap.add_argument("--tile-bins", type=int, default=10)
    ap.add_argument("--mae", default=None, help="a Mel-MAE (train-mae, model.type 2d) for the `mae` maps")
    ap.add_argument("--context", type=int, default=1, help="recorded patches kept before every row (the MAE's window)")
    ap.add_argument("--max-locations", type=int, default=0,
                    help="maps with more locations use a random subset of this many per step (0: all)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rows", type=int, default=2048, help="states per update")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per update (memory: rows / accum x samples)")
    ap.add_argument("--samples", type=int, default=32, help="head samples per state (G)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau-lr", type=float, default=1e-3)
    ap.add_argument("--tau-init", type=float, default=0.06, help="the training's maps (its converged value)")
    ap.add_argument("--tau-init-new", type=float, default=0.2, help="maps the head was not trained with")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--input-noise", type=float, default=0.3)
    ap.add_argument("--cache", default=None, help="keep the rows here (reused when the file exists)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg, stats = load_tts(args.model, dev)
    k, C, G = model.patch, model.n_mels, args.samples
    if args.cache and Path(args.cache).exists():
        rows = torch.load(args.cache)
    else:
        ds = MelDataset(args.data, args.split, min_frames=int(cfg.drift.get("crop_frames", 32)),
                        max_frames=cfg.data.get("max_frames", 1500))
        ds.stats.mean, ds.stats.std = float(stats["mean"]), float(stats["std"])
        rows = {n: v.cpu() for n, v in cache_states(model, ds, dev, args.input_noise, 16, args.seed,
                                                    args.context).items()}
        if args.cache:
            Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(rows, args.cache)
    model.to("cpu")  # only the head trains: the rows stay in host memory, the rest of the model leaves the GPU
    model.head.to(dev)
    torch.cuda.empty_cache()
    N = rows["state"].shape[0]
    if rows["prev"].shape[1] != args.context * k * C:
        raise SystemExit(f"{args.cache} holds {rows['prev'].shape[1] // (k * C)} patches of history, not "
                         f"{args.context}")
    mae = None
    if "mae" in args.maps:
        from drifting_tts.train_mae import load_mae

        if not args.mae:
            raise SystemExit("the mae maps need --mae <checkpoint>")
        mae = load_mae(args.mae, dev)
    print(f"{N} rows ({N * k / 93.75 / 3600:.1f} h of patches); maps: {' '.join(args.maps)}", flush=True)

    head = model.head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(True)
    head.train()
    ema = copy.deepcopy(head).eval()
    maps = tuple(args.maps)
    scale = residual_scale(head, rows, k, C, G, dev) if "wdct" in maps or "wpatch" in maps else None
    if scale is not None:
        share = scale.pow(2).sum(1) / scale.pow(2).sum()
        print("residual energy by temporal frequency: " + " ".join(f"{v:.3f}" for v in share.tolist()), flush=True)
    with torch.no_grad():  # the names of the maps as produced (the MAE gives one per stage)
        names = list(kernel_features(rows["target"][:2, None].to(dev).float(), rows["prev"][:2].to(dev).float(), k, C,
                                     maps, args.tile_bins, mae, scale))
    taus = {m: torch.nn.Parameter(torch.tensor(args.tau_init if m in ("patch", "frame", "detail")
                                                else args.tau_init_new, device=dev)) for m in names}
    opt = torch.optim.AdamW([{"params": list(head.parameters()), "lr": args.lr},
                             {"params": list(taus.values()), "lr": args.tau_lr}],
                            betas=(0.9, 0.999), weight_decay=0.0, fused=dev == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    gc = torch.Generator().manual_seed(args.seed)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    log, agg, t0 = [], {}, time.time()
    R = args.rows // args.accum  # states per micro-batch; an update sees ``rows`` states
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            idx = torch.randint(N, (R,), generator=gc)
            s, target, base, prev = (rows[n][idx].to(dev).float() for n in ("state", "target", "base", "prev"))
            z = torch.randn(R * G, head.noise_dim, device=dev, generator=g)
            with amp:
                res = head(s.repeat_interleave(G, 0), z)
            x = base[:, None] + res.float().view(R, G, -1)
            with amp:
                f_gen = kernel_features(x, prev, k, C, maps, args.tile_bins, mae, scale)
                with torch.no_grad():
                    f_pos = kernel_features(target[:, None], prev, k, C, maps, args.tile_bins, mae, scale)
            f_gen, f_pos = ({n: v.float() for n, v in f.items()} for f in (f_gen, f_pos))
            loss, info = 0.0, {}
            if args.loss in ("kyutai", "both"):
                loss, info = feature_drift_loss(f_gen, f_pos, None, taus=taus, reduce="mean",
                                                max_locations=args.max_locations or None)
            if args.loss in ("energy", "both"):
                e_loss, e_info = feature_energy_loss(f_gen, f_pos, max_locations=args.max_locations or None)
                loss, info = loss + e_loss, {**info, **e_info, "energy": e_loss.detach()}
            (loss / args.accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(head.parameters(), 3.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for t in taus.values():
                t.clamp_(min=1e-3)
            for pe, pm in zip(ema.parameters(), head.parameters()):
                pe.lerp_(pm, 1 - args.ema)
            m = {"loss": loss.detach(), "grad_norm": gnorm,
                 "centroid_mse": ((x.mean(1) - target) ** 2).mean(), "sample_std": x.std(1).mean()}
            if "tau_loss" in info:
                m["drift"] = loss.detach() - info["tau_loss"] - info.get("energy", 0.0)
                m.update({f"tau/{name}": t.detach() for name, t in taus.items()})
            m.update({name: v for name, v in info.items()
                      if name.startswith(("p_data/", "spread/")) or name == "energy"})
            for name, v in m.items():
                agg[name] = agg.get(name, 0.0) + float(v)
        if step % 100 == 0:
            rec = {"step": step, **{name: v / 100 for name, v in agg.items()}}
            log.append(rec)
            print(f"step {step} " + " ".join(f"{name}={v:.4g}" for name, v in rec.items() if name != "step")
                  + f" ({100 / (time.time() - t0):.1f} it/s)", flush=True)
            agg, t0 = {}, time.time()
    torch.save({"head": ema.state_dict(), "maps": names, "steps": args.steps,
                "taus": {name: float(t.detach()) for name, t in taus.items()}, "args": vars(args)},
               out / "head.pt")
    (out / "log.json").write_text(json.dumps(log))
    print(f"done: {out / 'head.pt'}", flush=True)


if __name__ == "__main__":
    main()
