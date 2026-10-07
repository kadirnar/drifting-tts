"""Calibrate predicted durations: store the median ground-truth / predicted length ratio in the checkpoint.

The duration predictor regresses log-durations, so ``exp(E[log d])`` under-estimates ``E[d]`` and
synthesised speech tends to be too fast. The ratio is measured on *training* utterances (never on the
evaluation split) and multiplied into ``length_scale`` at synthesis time. ``--temperature`` also stores the
preferred noise temperature, which the ``synthesize`` / ``evaluate`` CLIs then use by default.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import MelDataset
from .models.text_encoder import durations_to_alignment
from .utils import save_checkpoint


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="runs/tts/model_ema.pt")
    p.add_argument("--num", type=int, default=500, help="number of training utterances")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=None,
                   help="also store the preferred noise temperature (chosen with `evaluate --split dev`); "
                        "synthesize / evaluate use it by default")
    p.add_argument("--dry-run", action="store_true", help="only print the ratio")
    p.add_argument("--speakers", nargs="*", default=None,
                   help="voices / speaker IDs that get their own factor (default: the offered voices)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def measure_ratio(model, ds: MelDataset, num: int, seed: int, device, items: list[int] | None = None) -> np.ndarray:
    """Recorded / predicted length of ``num`` random training utterances (of ``items`` if given)."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(ds)) if items is None else np.asarray(items)
    ratios = []
    for i in rng.choice(pool, min(num, len(pool)), replace=False):
        item = ds[int(i)]
        text = item["text"][None].to(device)
        _, _, logw, x_mask = model.encoder(text, torch.tensor([text.shape[1]], device=device),
                                           torch.tensor([item["spk"]], device=device))
        _, y_len = durations_to_alignment(logw, x_mask)
        ratios.append(item["mel"].shape[1] / int(y_len))
    return np.array(ratios)


def run(args) -> None:
    from .train import load_tts
    from .voices import VOICES, voice_id

    model, cfg, _ = load_tts(args.model, args.device)
    d = cfg.data
    ds = MelDataset(d.root, "train", min_quality=d.min_quality, max_frames=d.max_frames, filters=d.get("filters"))
    r = measure_ratio(model, ds, args.num, args.seed, args.device)
    scale = float(np.median(r))
    print(f"ground-truth / predicted length over {len(r)} training utterances: median {scale:.3f}, "
          f"mean {r.mean():.3f}, p10 {np.percentile(r, 10):.3f}, p90 {np.percentile(r, 90):.3f}")
    # speakers differ in tempo, so every offered voice gets its own factor (the global one stays the fallback)
    n_spk = model.encoder.spk.num_embeddings
    speakers = [voice_id(v) for v in (args.speakers or VOICES)]
    scales = {}
    for spk in speakers:
        items = [i for i, e in enumerate(ds.items) if e["spk_id"] == spk]
        if spk >= n_spk or len(items) < 20:
            print(f"speaker {spk}: not in the model or < 20 training utterances, uses the global factor")
            continue
        scales[spk] = float(np.median(measure_ratio(model, ds, args.num, args.seed, args.device, items)))
        print(f"speaker {spk}: median {scales[spk]:.3f} over {min(args.num, len(items))} utterances")
    if args.dry_run:
        return
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    ck["duration_scale"], ck["duration_scales"] = scale, scales
    if args.temperature is not None:
        ck["temperature"] = args.temperature
    save_checkpoint(args.model, **ck)
    extra = "" if args.temperature is None else f" and temperature={args.temperature:g}"
    print(f"stored duration_scale={scale:.3f}, {len(scales)} per-voice factors{extra} in {args.model}")
