"""Turkish polar-question diagnostic for prosody predictors (#39 / #40), token level, held-out utterances.

In Turkish yes/no questions the question particle mI (mı / mi / mu / mü, written as a separate word, with optional
person / copula suffixes) is unaccented: the word before it carries the peak, and most such questions end low.
For every held-out (val + dev) utterance with a "?" and a mI word, this measures in semitones, on voiced tokens:

* ``pre_mi``: mean pitch of the word before the (last) mI word minus the utterance mean;
* ``fall``: mean pitch of the last word minus that of the word before mI;

for the recordings (token pitch targets) and every predictor (mean over seeds). A predictor that learned the pattern
has the recordings' positive ``pre_mi`` and negative ``fall``.

    python scripts/eval_question_pitch.py --cache runs/pm_cache/targets_v31.pt --tts runs/release/drifting_tts_v3.1.pt \
        --prosody drift=runs/pm_drift/prosody_ema.pt --out runs/pm_eval/questions.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from drifting_tts.models.prosody_net import ProsodyPredictor
from drifting_tts.text import SYMBOL_TO_ID
from drifting_tts.train_prosody import ProsodyData, load_frozen_tts, sample_split

MI = re.compile(r"m[ıiuü](?:s[ıiuü]n(?:[ıiuü]z)?|y[ıiuü][mz]|d[ıiuü]r(?:l[ae]r)?|yd[ıiuü][mnk]?|ym[ıiuü]ş)?")


def question_words(text: str) -> tuple[int, int] | None:
    """(index of the word before the last mI word, index of the last word) for a polar question, else None."""
    if "?" not in text:
        return None
    words = text.split(" ")
    mi = [i for i, w in enumerate(words) if MI.fullmatch(w.strip("?,.!")) and i > 0]
    return (mi[-1] - 1, len(words) - 1) if mi else None


def word_pitch(pitch: np.ndarray, voiced: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean pitch per word over voiced tokens (nan without any) and the utterance mean."""
    word = np.cumsum(ids == SYMBOL_TO_ID[" "])
    out = np.full(word.max() + 1, np.nan)
    for w in range(word.max() + 1):
        sel = (word == w) & voiced
        if sel.any():
            out[w] = pitch[sel].mean()
    return out, float(pitch[voiced].mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--tts", required=True)
    p.add_argument("--prosody", nargs="*", default=[], help="name=checkpoint[@temperature]")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tts = load_frozen_tts(args.tts, args.device)
    st = float(tts.lf0_stats[1]) * 12 / math.log(2)  # normalised log-F0 -> semitones
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    data = ProsodyData(cache, ("val", "dev"))
    keep = [i for i, u in enumerate(data.utts) if question_words(u["norm_text"])]
    data = data.subset(keep)
    print(f"{len(data)} held-out polar questions")
    systems = {"recordings": None, "v3.1 regressors": "v31"}
    systems.update({s.split("=")[0]: s.split("=", 1)[1] for s in args.prosody})
    rows = []
    for name, spec in systems.items():
        if spec is None:
            samples = [{"pcont": it["pcont"][None]} for it in data.items]
        elif spec == "v31":
            samples = sample_split(None, tts, data, [0], 1.0, args.device)
        else:
            path, _, t = spec.partition("@")
            pred = ProsodyPredictor.load(path, args.device, tts=tts)
            seeds = [0] if pred.kind == "mse" else list(range(args.seeds))
            samples = sample_split(pred, tts, data, seeds, float(t or 1.0), args.device)
        pre, fall = [], []
        for s, it, u in zip(samples, data.items, data.utts):
            b, last = question_words(u["norm_text"])
            for pc in s["pcont"]:
                w, mean = word_pitch(pc, it["voiced"], it["ids"])
                if not (np.isnan(w[b]) or np.isnan(w[last])):
                    pre.append((w[b] - mean) * st)
                    fall.append((w[last] - w[b]) * st)
        rows.append({"system": name, "pre_mi_st": float(np.mean(pre)), "pre_mi_sd": float(np.std(pre)),
                     "fall_st": float(np.mean(fall)), "fall_sd": float(np.std(fall)),
                     "end_low_frac": float(np.mean(np.array(fall) < 0)), "n": len(pre)})
        print(json.dumps(rows[-1]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"questions": len(data), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
