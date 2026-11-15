"""Is ARDrift's head calibrated? Its samples against the recordings, in the drifting kernel's own spaces.

The drifting field vanishes when the head's distribution ``q(patch | state)`` equals the data's ``p(patch | state)``,
but with one recorded patch per state nothing in training reports whether that is where the head ended up. This script
measures it on held-out utterances, teacher-forced on the recording (the states the head was trained on):

* **spread against error** (``d_neg / d_pos``): the mean distance between two samples of a state, over the mean
  distance from a sample to the recorded patch. If ``q = p`` the recorded patch is one more sample and the two agree
  (ratio 1); below 1 the samples sit closer to each other than to the data: an under-dispersed head. Reported per
  kernel map (``patch``, ``frame``, ``detail``, as :func:`drifting_tts.models.ar.ar_features`).
* **spread-skill ratio** per mel band: ``sqrt((1 + 1/G) var_samples) / rmse(sample mean, recording)``, the ensemble
  forecasters' calibration check (1 when calibrated).
* **rank of the recording** among the ``G`` samples for three scalars of a patch: its level (mean), its harmonic depth
  (RMS of the spectral detail) and its motion (RMS of the frame-to-frame change). Uniform ranks when ``q = p``;
  ``top`` is the share of states whose recording exceeds every sample (expected ``1 / (G + 1)``): for the depth and
  the motion that is over-smoothing, stated as a probability.

and on whole mels (the recording, teacher-forced samples, free-running generation under the recording's alignment):

* the variance of the mel's Laplacian (VarL, Ren et al., arXiv 2202.13066: lower = over-smoothed), the RMS of the
  spectral detail and of the frame-to-frame change, and that change across patch boundaries over the change inside
  a patch (#97: do the patches join as the recording's frames do?).

    python scripts/drift_calibration.py --model checkpoints/en/ardrift_en.pt --data data/en_val --split train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from drifting_tts.alignment import sequence_mask
from drifting_tts.data import MelDataset, collate
from drifting_tts.drift import cdist
from drifting_tts.models.ar import ar_features, patchify
from drifting_tts.models.mae import spectral_detail
from drifting_tts.models.text_encoder import align
from drifting_tts.train import load_tts

BANDS = {"low": (0, 33), "mid": (33, 66), "high": (66, 100)}


def texture(mel: torch.Tensor, length: torch.Tensor, patch: int) -> dict[str, torch.Tensor]:
    """Per-utterance texture statistics of normalised mels ``[B, C, T]`` over their valid frames ``length`` ``[B]``."""
    B, C, T = mel.shape
    m = sequence_mask(length, T)[:, None].float()  # [B, 1, T]
    n = m.sum((1, 2)).clamp_min(1)
    lap = F.conv2d(mel[:, None], mel.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None])[:, 0]  # [B, C-2, T-2]
    ml = (m[:, :, 1:-1] * m[:, :, 2:] * m[:, :, :-2]).expand(-1, C - 2, -1)
    lmean = (lap * ml).sum((1, 2)) / ml.sum((1, 2)).clamp_min(1)
    varl = (((lap - lmean[:, None, None]) ** 2) * ml).sum((1, 2)) / ml.sum((1, 2)).clamp_min(1)
    det = spectral_detail(mel)
    detail = ((det**2 * m).sum((1, 2)) / (n * C)).sqrt()
    d = (mel[:, :, 1:] - mel[:, :, :-1]) ** 2  # change from frame t to t + 1
    md = (m[:, :, 1:] * m[:, :, :-1])
    delta = ((d * md).sum((1, 2)) / (md.sum((1, 2)).clamp_min(1) * C)).sqrt()
    t = torch.arange(T - 1, device=mel.device)
    edge = ((t + 1) % patch == 0)[None, None].float()  # the pair (t, t + 1) straddles two patches
    at = lambda w: ((d * md * w).sum((1, 2)) / ((md * w).sum((1, 2)).clamp_min(1) * C)).sqrt()  # noqa: E731
    return {"varl": varl, "detail_rms": detail, "delta_rms": delta, "boundary_ratio": at(edge) / at(1 - edge)}


def patch_scalars(x: torch.Tensor, k: int, c: int) -> dict[str, torch.Tensor]:
    """Patches ``[R, S, k * c]`` -> scalars ``[R, S]``: level, harmonic depth, motion."""
    R, S, _ = x.shape
    fr = x.view(R * S, k, c)
    det = spectral_detail(fr.transpose(1, 2))
    return {"level": fr.mean((1, 2)).view(R, S), "depth": det.pow(2).mean((1, 2)).sqrt().view(R, S),
            "motion": (fr[:, 1:] - fr[:, :-1]).pow(2).mean((1, 2)).sqrt().view(R, S)}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--samples", type=int, default=32, help="head samples per state (G)")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--input-noise", type=float, default=0.0, help="noise on the teacher-forced history")
    ap.add_argument("--max-rows", type=int, default=1024, help="states per batch for the per-state statistics")
    ap.add_argument("--spk-shift", type=int, default=0, help="add to every speaker id (to check a speaker table)")
    ap.add_argument("--no-free", action="store_true", help="skip free-running generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, cfg, stats = load_tts(args.model, dev)
    ds = MelDataset(args.data, args.split, min_frames=1, max_frames=cfg.data.get("max_frames", 1500))
    ds.stats.mean, ds.stats.std = float(stats["mean"]), float(stats["std"])  # the model's own normalisation
    k, C, G = model.patch, model.n_mels, args.samples
    if dev == "cuda":
        model.enable_fast()  # CUDA graphs for the free-running passes
    idx = list(range(min(args.num, len(ds))))

    acc: dict[str, list] = {}
    add = lambda key, v: acc.setdefault(key, []).append(v.double().cpu())  # noqa: E731
    for b0 in range(0, len(idx), args.batch):
        batch = collate([ds[i] for i in idx[b0: b0 + args.batch]])
        text, text_len, y, y_len, spk = (batch[n].to(dev) for n in ("text", "text_len", "mel", "mel_len", "spk"))
        spk = (spk + args.spk_shift) % model.spk.num_embeddings
        y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
        h, mu, _, x_mask = model.encoder(text, text_len, spk)
        attn, _ = align(mu, x_mask, y, y_mask)
        cond, yp = model.pad_frames(model.frame_condition(h, mu, attn)), model.pad_frames(y)
        add("prior_mse", ((cond[:, :C, : y.shape[-1]] - y) ** 2 * y_mask).sum((1, 2)) / (y_mask.sum((1, 2)) * C))
        noise = None if not args.input_noise else torch.full((y.shape[0],), args.input_noise, device=dev)
        s = model.teacher_forced(cond, yp, spk, noise)
        S = s.shape[1]
        full = (torch.arange(S, device=dev)[None] + 1) * k <= y_len[:, None]  # whole patches inside the utterance
        rows_all = full.flatten().nonzero()[:, 0]
        rows = rows_all[torch.randperm(rows_all.numel(), device=dev)[: args.max_rows]]
        st, target = s.flatten(0, 1)[rows], patchify(yp, k).flatten(0, 1)[rows]
        base = patchify(cond[:, :C], k).flatten(0, 1)[rows]
        R = st.shape[0]
        tex_rec = texture(y, y_len, k)
        for name, v in tex_rec.items():
            add(f"texture/recording/{name}", v)
        for T in args.temperatures:
            tag = f"T{T:g}"
            z = torch.randn(R * G, model.head.noise_dim, device=dev) * T
            x = base[:, None] + model.head(st.repeat_interleave(G, 0), z).view(R, G, -1)
            f_gen, f_pos = ar_features(x, k, C), ar_features(target[:, None], k, C)
            for key in f_gen:
                g, p = (t.permute(0, 2, 1, 3).flatten(0, 1) for t in (f_gen[key], f_pos[key]))  # [R * L, S, D]
                d_pos = cdist(g, p)[..., 0].mean(1)
                d_neg = cdist(g, g).sum((1, 2)) / (G * (G - 1))
                add(f"{tag}/{key}/d_pos", d_pos)
                add(f"{tag}/{key}/d_neg", d_neg)
            fr = x.view(R, G, k, C)
            var = fr.var(1).mean(1)  # [R, C]: sample variance, mean over the patch's frames
            err = (fr.mean(1) - target.view(R, k, C)).pow(2).mean(1)  # [R, C]
            for band, (a, b) in BANDS.items():
                add(f"{tag}/band/{band}/var", var[:, a:b].mean(1))
                add(f"{tag}/band/{band}/err", err[:, a:b].mean(1))
            sg, sp = patch_scalars(x, k, C), patch_scalars(target[:, None], k, C)
            for name in sg:
                rank = (sg[name] < sp[name]).sum(1)  # 0 .. G: samples below the recording
                add(f"{tag}/rank/{name}", rank)
                add(f"{tag}/scalar/{name}/gen", sg[name].mean(1))
                add(f"{tag}/scalar/{name}/rec", sp[name][:, 0])
            # whole mels: one teacher-forced sample per state, and free-running under the recording's alignment
            g = torch.Generator(device=dev).manual_seed(args.seed)
            for name, v in texture(model.gta(cond, yp, spk, T, g)[..., : y.shape[-1]], y_len, k).items():
                add(f"texture/{tag}/teacher_forced/{name}", v)
            if not args.no_free:
                gen = model.fast_generator or model.generate
                free = gen(cond[..., : y.shape[-1]], spk, T, g)
                for name, v in texture(free, y_len, k).items():
                    add(f"texture/{tag}/free_running/{name}", v)
                add(f"{tag}/free_mse", ((free - y) ** 2 * y_mask).sum((1, 2)) / (y_mask.sum((1, 2)) * C))
        print(f"{min(b0 + args.batch, len(idx))}/{len(idx)} utterances", flush=True)

    cat = {key: torch.cat(v) for key, v in acc.items()}
    mean = lambda key: float(cat[key].mean())  # noqa: E731
    out: dict = {"model": args.model, "utterances": len(idx), "samples": G, "input_noise": args.input_noise,
                 "prior_mse": mean("prior_mse"), "temperatures": {}, "texture": {}}
    rec = {n: mean(f"texture/recording/{n}") for n in ("varl", "detail_rms", "delta_rms", "boundary_ratio")}
    out["texture"]["recording"] = rec
    for T in args.temperatures:
        tag = f"T{T:g}"
        r: dict = {"spread_over_error": {}, "spread_skill": {}, "rank": {}, "scalar_gen_over_rec": {}}
        for key in ("patch", "frame", "detail"):
            r["spread_over_error"][key] = mean(f"{tag}/{key}/d_neg") / mean(f"{tag}/{key}/d_pos")
        for band in BANDS:
            r["spread_skill"][band] = ((1 + 1 / G) * mean(f"{tag}/band/{band}/var") / mean(f"{tag}/band/{band}/err")
                                       ) ** 0.5
        for name in ("level", "depth", "motion"):
            rank = cat[f"{tag}/rank/{name}"]
            r["rank"][name] = {"top": float((rank == G).double().mean()), "bottom": float((rank == 0).double().mean()),
                               "mean": float(rank.double().mean()) / G}
            r["scalar_gen_over_rec"][name] = mean(f"{tag}/scalar/{name}/gen") / mean(f"{tag}/scalar/{name}/rec")
        if f"{tag}/free_mse" in cat:
            r["free_mse"] = mean(f"{tag}/free_mse")
        out["temperatures"][tag] = r
        for mode in ("teacher_forced", "free_running"):
            if f"texture/{tag}/{mode}/varl" in cat:
                out["texture"][f"{tag}/{mode}"] = {n: mean(f"texture/{tag}/{mode}/{n}") for n in rec}

    print(f"\n{len(idx)} utterances, G = {G}, history noise {args.input_noise:g}, prior MSE {out['prior_mse']:.4f}; "
          f"expected rank share at an end: {1 / (G + 1):.3f}")
    for tag, r in out["temperatures"].items():
        print(f"\n[{tag}] spread / error (1 = calibrated): "
              + ", ".join(f"{key} {v:.3f}" for key, v in r["spread_over_error"].items()))
        print("      spread-skill by band: " + ", ".join(f"{b} {v:.3f}" for b, v in r["spread_skill"].items()))
        for name, v in r["rank"].items():
            print(f"      {name:6s} recording above every sample {v['top']:.3f}, below every sample {v['bottom']:.3f}, "
                  f"mean rank {v['mean']:.3f}; samples / recording {r['scalar_gen_over_rec'][name]:.3f}")
    print("\ntexture (ratio to the recording in brackets)")
    for name, t in out["texture"].items():
        print(f"  {name:24s} " + "  ".join(f"{n} {v:.4f} ({v / rec[n]:.2f})" for n, v in t.items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
