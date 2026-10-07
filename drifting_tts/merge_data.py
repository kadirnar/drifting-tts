"""Merge prepared datasets into one, e.g. the base corpus plus a new voice for fine-tuning.

The first dataset is the base. Its speaker IDs, its normalisation statistics (``stats.json``, which the base model
was trained with) and its ``scores.jsonl`` are kept, and its utterances come first, so the line numbers that
``scores.jsonl`` refers to stay valid. Speakers new in later datasets get the next IDs. ``--repeat`` lists the
training utterances of a dataset several times, so its share of the training batches grows.

    drifting-tts merge-data --out data/train_new_voice data/train data/new_voice --repeat 1 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# binary -> (index key of its offset, bytes per unit); mels and F0 share the frame offset, audio has its own
BINARIES = {"mels.bin": ("offset", 200), "f0.bin": ("offset", 2), "audio.bin": ("audio_offset", 2)}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="prepared datasets; the first one is the base")
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, nargs="+", default=None, help="times each dataset's utterances are listed")


def run(args) -> None:
    roots = [Path(r) for r in args.inputs]
    repeat = args.repeat or [1] * len(roots)
    if len(repeat) != len(roots):
        raise SystemExit("--repeat needs one value per input")
    stats = [json.loads((r / "stats.json").read_text()) for r in roots]
    if len({s.get("backend", "vocos") for s in stats}) != 1:
        raise SystemExit("all inputs must use the same mel backend")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    speakers = json.loads((roots[0] / "speakers.json").read_text())
    for r in roots[1:]:
        for name in json.loads((r / "speakers.json").read_text()):
            speakers.setdefault(name, len(speakers))

    files = [f for f in BINARIES if all((r / f).exists() for r in roots)]
    base = {"offset": 0, "audio_offset": 0}  # frames / samples already written
    with open(out / "index.jsonl", "w") as index:
        for r, n in zip(roots, repeat):
            entries = [json.loads(line) for line in open(r / "index.jsonl")]
            for e in entries:
                e["spk_id"] = speakers[e["speaker"]]
                for key, start in base.items():
                    if key in e:
                        e[key] += start
            for k in range(n):  # repeat training utterances only: held-out splits stay unique
                for e in entries:
                    if k == 0 or e["split"] == "train":
                        index.write(json.dumps(e, ensure_ascii=False) + "\n")
            for f in files:
                with open(r / f, "rb") as src, open(out / f, "ab" if r != roots[0] else "wb") as dst:
                    shutil.copyfileobj(src, dst, 64 << 20)
                base[BINARIES[f][0]] = (out / f).stat().st_size // BINARIES[f][1]
    (out / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=0))
    merged = {**stats[0], "merged_from": [str(r) for r in roots], "repeat": repeat}
    (out / "stats.json").write_text(json.dumps(merged, indent=1))
    if (roots[0] / "scores.jsonl").exists():  # line numbers of the base dataset are unchanged
        shutil.copy(roots[0] / "scores.jsonl", out / "scores.jsonl")
    print(f"merged {len(roots)} datasets -> {out} ({len(speakers)} speakers, binaries: {', '.join(files)})")
