"""Token-level comparison of prosody predictors against the recordings' targets (no audio): `train-prosody` #39.

Rows: the TTS model's deterministic regressors (as used at inference: ceil of exp(log-duration) times the per-voice
factor) and every given prosody checkpoint at every temperature, K seeds each. Metrics: see
`drifting_tts.train_prosody.token_metrics` (ratios of 1 = as varied as the recordings).

    python scripts/eval_prosody_tokens.py --cache runs/pm_cache/targets_v31.pt --tts runs/release/drifting_tts_v3.1.pt \
        --prosody drift=runs/pm_drift/prosody_ema.pt mse=runs/pm_mse/prosody_ema.pt --temperatures 0.5 0.7 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drifting_tts.models.prosody_net import ProsodyPredictor
from drifting_tts.train_prosody import ProsodyData, load_frozen_tts, sample_split, token_metrics

COLS = [("p_std_ratio", "pitch std ratio"), ("word_p_std_ratio", "word pitch std ratio"),
        ("ld_std_ratio", "log-dur std ratio"), ("letter_std_ratio", "letter dur std ratio"),
        ("word_dur_std_ratio", "word dur std ratio"), ("jitter_ratio", "pitch jitter ratio"),
        ("reversal_rate", "reversals / letter"),
        ("p_corr", "pitch r"), ("letter_corr", "letter dur r"), ("word_dur_corr", "word dur r"),
        ("p_crps", "pitch CRPS"), ("ld_crps", "log-dur CRPS"), ("w1_pdev", "W1 pitch dev"), ("w1_ld", "W1 log-dur"),
        ("w1_utt_pstd", "W1 utt pitch std"), ("rate", "length ratio"), ("voicing_acc", "voicing acc"),
        ("div_p", "seed div pitch"), ("div_ld", "seed div log-dur"), ("div_total", "seed div length")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--tts", required=True)
    p.add_argument("--prosody", nargs="*", default=[], help="name=checkpoint")
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    p.add_argument("--spreads", type=float, nargs="+", default=[1.0], help="output-space temperatures")
    p.add_argument("--pitch-temperatures", type=float, nargs="+", default=None,
                   help="temperatures of the pitch channel (default: the same as --temperatures); the durations use "
                        "--temperatures")
    p.add_argument("--min-letter-frames", type=float, default=0.0, help="floor_letters on the sampled durations")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--scales", action="store_true", help="apply the checkpoints' per-voice duration factors")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tts = load_frozen_tts(args.tts, args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    sets = {"val+dev (all speakers)": ProsodyData(cache, ("val", "dev")),
            "val+dev studio (722)": ProsodyData(cache, ("val", "dev"), [722])}
    out_json = Path(args.out).with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out_json.read_text()) if out_json.exists() else []  # resume
    done = {(r["set"], r["system"], r["T"]) for r in rows}

    def add(row: dict) -> None:
        rows.append(row)
        out_json.write_text(json.dumps(rows, indent=1))

    for set_name, data in sets.items():
        if (set_name, "v3.1 regressors", None) not in done:
            res = token_metrics(sample_split(None, tts, data, [0], 1.0, args.device), data)
            add({"set": set_name, "system": "v3.1 regressors", "T": None, **res, "voicing_acc": None,
                 "div_p": 0.0, "div_ld": 0.0, "div_total": 0.0})
        for spec in args.prosody:
            name, path = spec.split("=", 1)
            pred = ProsodyPredictor.load(path, args.device, tts=tts)
            temps = [1.0] if pred.kind == "mse" else args.temperatures
            seeds = [0] if pred.kind == "mse" else list(range(args.seeds))
            spreads = [1.0] if pred.kind == "mse" else args.spreads
            pitch_temps = args.pitch_temperatures or [None]
            for T, lam, tp in ((t, s, q) for t in temps for s in spreads for q in pitch_temps):
                label = name if lam == 1.0 else f"{name} (spread {lam:g})"
                if tp is not None and tp != T:
                    label += f" (pitch T {tp:g})"
                if args.min_letter_frames:
                    label += f" (floor {args.min_letter_frames:g})"
                if (set_name, label, None if pred.kind == "mse" else T) in done:
                    continue
                s = sample_split(pred, tts, data, seeds, T, args.device, apply_scales=args.scales, spread=lam,
                                 pitch_temperature=tp, min_letter_frames=args.min_letter_frames)
                res = token_metrics(s, data)
                if pred.kind == "mse":
                    res.update(div_p=0.0, div_ld=0.0, div_total=0.0)
                add({"set": set_name, "system": label, "T": None if pred.kind == "mse" else T, **res})
                print(set_name, label, T, {k: round(v, 3) for k, v in res.items() if v == v}, flush=True)
    lines = []
    for set_name in sets:
        lines += [f"### {set_name}", "", "| system | T | " + " | ".join(c[1] for c in COLS) + " |",
                  "|---|---|" + "---|" * len(COLS)]
        lines.append("| recordings | – | " + " | ".join("1" if "ratio" in k and k != "rate" else
                                                       "1" if k in ("rate", "p_corr", "letter_corr", "word_dur_corr")
                                                       else "0" if k.startswith(("w1", "p_crps", "ld_crps")) else "–"
                                                       for k, _ in COLS) + " |")
        for r in rows:
            if r["set"] != set_name:
                continue
            cells = ["–" if r.get(k) is None or r.get(k) != r.get(k) else f"{r[k]:.3f}" for k, _ in COLS]
            temp = "–" if r["T"] is None else f"{r['T']:g}"
            lines.append(f"| {r['system']} | {temp} | " + " | ".join(cells) + " |")
        lines.append("")
    Path(args.out).with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
