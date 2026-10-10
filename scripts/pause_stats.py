"""Pause lengths at internal punctuation in the training data, and the edge silence of generated sentences.

For every sampled training utterance, MAS durations under the model's own prior (as in training,
:func:`drifting_tts.prosody.token_targets`) locate each internal mark (``,`` ``.`` ``?`` ``!``). The pause there is
measured twice: (a) the MAS duration of the mark plus the space after it, (b) the energy-based silence
(:func:`drifting_tts.prosody.silent_frames`, 10 ms frames) overlapping that span ± 50 ms (0 if none). Groups: the
three voices (speaker IDs) and ``base``, the multi-speaker corpus without the studio voice.

``--generated N`` also synthesises N held-out sentences per voice (T 0.3, α 2, ``vocos-ft``) and measures their
leading and trailing silence. A sentence join of :class:`drifting_tts.prosody.PausePolicy` inserts the measured
pause minus that edge silence, so the whole gap matches the data.

    python scripts/pause_stats.py --model runs/release/drifting_tts_v3.1.pt --out runs/pe_pauses.json --generated 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drifting_tts.audio import HOP_LENGTH, SAMPLE_RATE
from drifting_tts.data import MelDataset
from drifting_tts.prosody import FRAME_MS, PUNCTUATION, edge_silence, frame_level_db, silent_frames, token_targets
from drifting_tts.text import SYMBOLS, split_sentences

VOICES = (722, 389, 323)


def mark_spans(ids: list[int], frames: np.ndarray) -> list[tuple[str, int, int]]:
    """Internal marks: ``(mark, first frame, end frame)`` of the mark's token, its blank and the space after it."""
    start = np.concatenate([[0], np.cumsum(frames)])
    out = []
    for k in range(1, len(ids) - 2, 2):  # character tokens sit at odd positions
        c = SYMBOLS[ids[k]]
        if c in PUNCTUATION:
            end = k + 4 if k + 2 < len(ids) and SYMBOLS[ids[k + 2]] == " " else k + 2
            if end < len(ids):  # a following character: the mark is internal
                out.append((c, int(start[k]), int(start[end])))
    return out


def silence_runs(silent: np.ndarray, min_frames: int = 3) -> list[tuple[int, int]]:
    runs, s = [], None
    for i, x in enumerate(np.append(silent, False)):
        if x and s is None:
            s = i
        elif not x and s is not None:
            if i - s >= min_frames:
                runs.append((s, i))
            s = None
    return runs


def pause_at(runs: list[tuple[int, int]], t0: float, t1: float, margin: float = 0.05) -> float:
    """Longest silent run (seconds) overlapping ``[t0 - margin, t1 + margin)``."""
    a, b = (t0 - margin) * 1000 / FRAME_MS, (t1 + margin) * 1000 / FRAME_MS
    lengths = [e - s for s, e in runs if s < b and e > a]
    return max(lengths, default=0) * FRAME_MS / 1000


def summary(values: dict[str, list]) -> dict:
    out = {}
    for mark, rows in sorted(values.items()):
        a = np.asarray(rows)
        out[mark] = {"n": len(a), "mas_mean": float(a[:, 0].mean()), "mas_std": float(a[:, 0].std()),
                     "silence_mean": float(a[:, 1].mean()), "silence_std": float(a[:, 1].std()),
                     "silence_median": float(np.median(a[:, 1])), "silence_p90": float(np.percentile(a[:, 1], 90)),
                     "pause_ge_100ms": float((a[:, 1] >= 0.1).mean())}
    return out


@torch.no_grad()
def data_pauses(model, ds: MelDataset, items: list[int], device: str) -> dict:
    values: dict[str, list] = {}
    frame_s = HOP_LENGTH / SAMPLE_RATE
    for i in items:
        it = ds[i]
        t = token_targets(model, it["text"][None].to(device), torch.tensor([it["spk"]], device=device),
                          it["mel"][None].to(device))
        spans = mark_spans(t["ids"].tolist(), t["durations_gt"].cpu().numpy())
        if not spans:
            continue
        runs = silence_runs(silent_frames(frame_level_db(it["audio"].numpy())))
        for mark, a, b in spans:
            values.setdefault(mark, []).append([(b - a) * frame_s, pause_at(runs, a * frame_s, b * frame_s)])
    return values


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="runs/release/drifting_tts_v3.1.pt")
    p.add_argument("--data", default=None, help="prepared data (default: the model's data.root)")
    p.add_argument("--num", type=int, default=1000, help="utterances with an internal mark per group")
    p.add_argument("--generated", type=int, default=0, help="held-out sentences synthesised per voice (0: none)")
    p.add_argument("--vocoder", default="vocos-ft")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/pe_pauses.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from drifting_tts.train import load_tts

    model, cfg, _ = load_tts(args.model, args.device)
    root = Path(args.data or cfg.data.root)
    ds = MelDataset(root, "train", min_frames=1, max_frames=10**9, with_audio=True)
    rng = np.random.default_rng(args.seed)
    has_mark = [any(c in PUNCTUATION for c in e["norm_text"][:-1]) for e in ds.items]
    groups = {str(v): [i for i, e in enumerate(ds.items) if e["spk_id"] == v and has_mark[i]] for v in VOICES}
    groups["base"] = [i for i, e in enumerate(ds.items) if e["spk_id"] != 722 and has_mark[i]]
    out: dict = {"model": args.model, "num": args.num, "seed": args.seed, "data": {}, "generated": {}}
    for name, pool in groups.items():
        items = sorted(rng.choice(pool, min(args.num, len(pool)), replace=False).tolist())
        out["data"][name] = summary(data_pauses(model, ds, items, args.device))
        out["data"][name]["utterances"] = len(items)
        print(name, json.dumps(out["data"][name]), flush=True)
    if args.generated:
        from drifting_tts.synthesize import Synthesizer

        synth = Synthesizer(args.model, args.device, vocoder=args.vocoder)
        held = [e for split in ("val", "dev") for e in MelDataset(root, split, min_frames=1, max_frames=10**9).items]
        for v in VOICES:
            pool = [e for e in held if e["spk_id"] == v]
            if len(pool) < 20:  # 389 / 323 have (almost) no held-out utterances: their training texts
                pool += [ds.items[i] for i in groups[str(v)]]
            sentences = [s for e in pool for s in split_sentences(e["norm_text"])][: args.generated]
            out["generated"][str(v)] = edge_silence(synth, sentences, v)
            print(v, json.dumps(out["generated"][str(v)]), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
