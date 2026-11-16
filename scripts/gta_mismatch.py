"""How far are the mels a vocoder is fine-tuned on from the mels it decodes (#90)?

``finetune-vocoder`` draws ARDrift's mels teacher-forced: every patch from the state after the *recorded* previous
patches (``ARDriftTTS.gta``). At inference every patch follows the model's *own* patches. This script generates both
for held-out utterances **under the recording's alignment** (so all three mels have the same frames) and measures the
difference the vocoder never trains on:

* ``distance``: mean absolute log-mel difference in dB, to the recording and between the two kinds, overall and by
  quarter of the utterance (does the free-running error grow along the utterance?). Two teacher-forced draws give the
  spread of the head's own noise, the floor of every other distance;
* ``pause``: the energy of the recording's quietest 10% of frames, in dB above the recording (docs/AR_DRIFT.md);
* ``frame change``: the RMS change from one frame to the next, inside a patch and across a patch boundary. Two
  teacher-forced patches are drawn independently after the recorded history, so they need not join;
* ``c2st``: can a classifier tell teacher-forced windows from free-running ones (``scripts/c2st_mel.py``; two
  teacher-forced draws are its floor, 0.5)? What it can tell is what the vocoder meets for the first time at
  inference. One-patch windows on the patch grid (``whole``) and half a patch off it (``straddling``) say whether
  the tell is inside the patches or where they join.

    python scripts/gta_mismatch.py --model checkpoints/en/ardrift_en.pt --data data/train --split val --num 200 \
        --temperature 0.5 --out outputs/diag/gta_mismatch_T0.5.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2st_mel import c2st  # noqa: E402

DB = 20 / math.log(10)  # log-mels are natural logs of magnitudes: 1 unit = 8.69 dB
KINDS = ("rec", "tf", "tf2", "free")


@torch.no_grad()
def renditions(model, ds, idx: list[int], dev: str, temperature: float, seed: int = 0, batch: int = 10,
               cfg_scale: float = 1.0) -> dict[str, list[torch.Tensor]]:
    """Normalised mels ``[n_mels, T]`` of the utterances ``idx``: the recording (``rec``), two teacher-forced draws
    (``tf``, ``tf2``) and a free-running one (``free``), all under the recording's alignment, in the order of ``idx``.
    """
    from drifting_tts.alignment import sequence_mask
    from drifting_tts.data import collate
    from drifting_tts.models.text_encoder import align

    out: dict[str, list[torch.Tensor]] = {k: [] for k in KINDS}
    order = sorted(idx, key=ds.frames)  # similar lengths per batch
    for b0 in range(0, len(order), batch):
        b = collate([ds[i] for i in order[b0: b0 + batch]])
        text, text_len, y, y_len, spk = (b[n].to(dev) for n in ("text", "text_len", "mel", "mel_len", "spk"))
        h, mu, _, x_mask = model.encoder(text, text_len, spk)
        attn, _ = align(mu, x_mask, y, sequence_mask(y_len, y.shape[-1])[:, None].float())
        cond = model.frame_condition(h, mu, attn)
        g = [torch.Generator(device=dev).manual_seed(seed + 3 * b0 + j) for j in range(3)]
        free = (model.fast_generator or model.generate)(cond, spk, temperature, g[2], cfg_scale=cfg_scale)
        mels = {"rec": y, "tf": model.gta(cond, y, spk, temperature, g[0]),
                "tf2": model.gta(cond, y, spk, temperature, g[1]), "free": free}
        for k, m in mels.items():
            out[k] += [m[i, :, :n].float().cpu() for i, n in enumerate(y_len.tolist())]
    back = sorted(range(len(order)), key=lambda j: idx.index(order[j]))  # batches were by length
    return {k: [v[j] for j in back] for k, v in out.items()}


def distance_db(a: list[torch.Tensor], b: list[torch.Tensor], std: float, parts: int = 4) -> dict:
    """Mean absolute difference of two lists of normalised log-mels in dB: over all frames (every utterance weighs
    the same) and in each of ``parts`` equal parts of the utterance."""
    per = torch.stack([(x - y).abs().mean() for x, y in zip(a, b)]) * std * DB
    by_part = [torch.stack([(x - y).abs().tensor_split(parts, 1)[p].mean() for x, y in zip(a, b)]).mean() * std * DB
               for p in range(parts)]
    return {"mean": float(per.mean()), "sem": float(per.std() / len(per) ** 0.5), "parts": [float(v) for v in by_part]}


def pause_excess_db(rec: list[torch.Tensor], gen: list[torch.Tensor], mean: float, std: float,
                    share: float = 0.1) -> dict:
    """Energy of ``gen`` on the recording's quietest ``share`` of frames, in dB above the recording's own there
    (mean over utterances; power summed over the mel bins)."""
    out = []
    for r, g in zip(rec, gen):
        pr, pg = ((2 * (m * std + mean)).exp().sum(0) for m in (r, g))  # frame power
        quiet = pr.argsort()[: max(1, int(share * len(pr)))]
        out.append(10 * torch.log10(pg[quiet].mean() / pr[quiet].mean()))
    out = torch.stack(out)
    return {"mean": float(out.mean()), "sem": float(out.std() / len(out) ** 0.5), "max": float(out.max())}


def frame_change_db(mels: list[torch.Tensor], std: float, patch: int) -> dict:
    """RMS change from frame ``t`` to ``t + 1`` in dB (mean over utterances): for the pairs inside a patch and for the
    pairs that straddle two patches, and their ratio (1 for a recording: it has no patch grid)."""
    inside, across = [], []
    for m in mels:
        d = (m[:, 1:] - m[:, :-1]).pow(2).mean(0)  # [T - 1]
        edge = (torch.arange(len(d)) + 1) % patch == 0
        inside.append(d[~edge].mean().sqrt())
        across.append(d[edge].mean().sqrt())
    a, b = float(torch.stack(inside).mean()) * std * DB, float(torch.stack(across).mean()) * std * DB
    return {"inside": a, "across": b, "ratio": b / a}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--cfg", type=float, default=1.0, help="guidance of the free-running rendition")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--views", nargs="+", default=["full", "envelope", "detail", "delta"])
    ap.add_argument("--steps", type=int, default=1500, help="classifier steps; 0 skips the classifier test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="keep the mels here (reused when the file exists)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from drifting_tts.data import MelDataset
    from drifting_tts.train import load_tts

    model, cfg, stats = load_tts(args.model, dev)
    mean, std = float(stats["mean"]), float(stats["std"])
    if args.cache and Path(args.cache).exists():
        mels = torch.load(args.cache)
    else:
        ds = MelDataset(args.data, args.split, min_frames=1, max_frames=cfg.data.get("max_frames", 1500))
        ds.stats.mean, ds.stats.std = mean, std
        if dev == "cuda":
            model.enable_fast()
        mels = renditions(model, ds, list(range(min(args.num, len(ds)))), dev, args.temperature, args.seed,
                          args.batch, args.cfg)
        if args.cache:
            Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(mels, args.cache)
    n = len(mels["rec"])
    res: dict = {"model": args.model, "temperature": args.temperature, "cfg": args.cfg, "utterances": n,
                 "distance_db": {}, "pause_db": {}, "c2st": {}}
    print(f"{n} utterances, T {args.temperature:g}, under the recordings' alignment\n")
    print("mean |log-mel difference| in dB (and by quarter of the utterance)")
    for a, b in (("tf", "tf2"), ("tf", "free"), ("tf", "rec"), ("free", "rec")):
        d = res["distance_db"][f"{a}-{b}"] = distance_db(mels[a], mels[b], std)
        print(f"  {a:>4s} - {b:<4s} {d['mean']:6.2f} +- {d['sem']:.2f}   "
              + " ".join(f"{v:6.2f}" for v in d["parts"]))
    print("\nenergy on the recording's quietest 10% of frames, dB above the recording")
    for k in ("tf", "free"):
        p = res["pause_db"][k] = pause_excess_db(mels["rec"], mels[k], mean, std)
        print(f"  {k:>4s} {p['mean']:+6.2f} +- {p['sem']:.2f} (worst utterance {p['max']:+.1f})")
    k = model.patch
    print(f"\nframe-to-frame change in dB: inside a patch, across a patch boundary ({k} frames), ratio")
    res["frame_change_db"] = {}
    for kind in ("rec", "tf", "free"):
        c = res["frame_change_db"][kind] = frame_change_db(mels[kind], std, k)
        print(f"  {kind:>4s} {c['inside']:6.2f} {c['across']:6.2f} {c['ratio']:6.2f}")
    if args.steps > 0:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
        floor = min(m.min() for m in mels["rec"])  # the log-mel floor of the front end
        sets = {k: [mels[k][i].clamp_min(floor) for i in perm] for k in KINDS}
        print("\nclassifier accuracy on held-out utterances (0.5 = indistinguishable)")
        print(f"{'pair':>12s} {'window':>6s}" + "".join(f"{v:>10s}" for v in args.views))
        for a, b in (("tf", "tf2"), ("tf", "free"), ("rec", "tf"), ("rec", "free")):
            for w in args.windows:
                row = {v: c2st(sets[a], sets[b], w, v, dev, args.steps, 128, args.seed, None, 0.02, 1)["accuracy"]
                       for v in args.views}
                res["c2st"].setdefault(f"{a}-{b}", {})[str(w)] = row
                print(f"{a + '-' + b:>12s} {w:6d}" + "".join(f"{row[v]:10.3f}" for v in args.views), flush=True)
        print(f"one-patch windows ({k} frames), teacher-forced against free-running")
        for name, off in (("whole", 0), ("straddling", k // 2)):
            a, b = ([m[:, off:] for m in sets[kind]] for kind in ("tf", "free"))
            row = {v: c2st(a, b, k, v, dev, args.steps, 128, args.seed, None, 0.02, k)["accuracy"] for v in args.views}
            res["c2st"][f"tf-free {name}"] = row
            print(f"{name:>19s}" + "".join(f"{row[v]:10.3f}" for v in args.views), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
