"""Audio-level prosody (:func:`drifting_tts.prosody.prosody_features`) of sets of wavs, overall and per sentence type
(docs/POCKET_TTS_GATE.md). Harvest F0 in ``--workers`` processes.

``--spec`` is a JSON ``{system: {dir, pattern, texts, text_key, type_key}}``: ``texts`` a .jsonl with one item per
file, ``pattern`` the file name formatted with the item's fields (e.g. ``"{index:05d}.wav"``), ``text_key`` the
field with the text (normalised for the speaking rate), ``type_key`` the field to split on (absent: one group).
Extra columns: ``final_st`` (median F0 of the last 25 voiced frames against the utterance median, semitones) and
``final_slope`` (their least-squares slope, st/s); ``g_*``: the F0 measures again with F0 zeroed on silent frames."""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile as sf


def _init() -> None:
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")


def final_contour(f0: np.ndarray) -> dict:
    v = np.nonzero(f0 > 0)[0]
    if len(v) < 30:
        return {}
    tail = v[-25:]
    st = 12 * np.log2(f0[tail] / np.median(f0[v]))
    return {"final_st": float(np.median(st)), "final_slope": float(np.polyfit(tail * 0.01, st, 1)[0])}


def features(path: str, text: str) -> dict:
    """``prosody_features`` on harvest F0 as is, plus ``g_*``: the same with F0 zeroed on the silent frames of
    ``silent_frames`` (harvest tracks low-level tonal noise in near-silent tails as voiced)."""
    from drifting_tts.prosody import f0_contour, frame_level_db, prosody_features, silent_frames

    wav, sr = sf.read(path, dtype="float64")
    f0 = f0_contour(wav, sr)
    row = prosody_features(wav, sr, f0=f0, text=text)
    row.update(final_contour(f0))
    sil = silent_frames(frame_level_db(wav, sr))
    g = f0.copy()
    n = min(len(g), len(sil))
    g[:n][sil[:n]] = 0
    g[len(sil):] = 0
    gated = prosody_features(wav, sr, f0=g, text=text)
    gated.update(final_contour(g))
    row["g_voiced_dropped_pct"] = float(100 * ((f0 > 0) & (g == 0)).sum() / max((f0 > 0).sum(), 1))
    row.update({f"g_{k}": v for k, v in gated.items() if k.startswith(("f0_", "final_", "voiced"))})
    return row


def aggregate(rows: list[dict]) -> dict:
    skip = {"pause_lengths"}
    keys = sorted({k for r in rows for k, v in r.items() if k not in skip and isinstance(v, (int, float))})
    out = {"utterances": len(rows)}
    for k in keys:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not math.isnan(r[k])]
        if v:
            out[k] = float(np.mean(v))
    lengths = [x for r in rows for x in r.get("pause_lengths", [])]
    out["pause_mean_s"] = float(np.mean(lengths)) if lengths else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="json: {system: {dir, pattern, texts(jsonl), text_key, type_key}}")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    from drifting_tts.text import normalize

    spec = json.loads(Path(args.spec).read_text())
    results = {}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init) as pool:
        for name, s in spec.items():
            items = [json.loads(line) for line in open(s["texts"])]
            futs = []
            for it in items:
                path = Path(s["dir"]) / s["pattern"].format(**it)
                text = it.get("norm") or normalize(it[s["text_key"]])
                futs.append((it, pool.submit(features, str(path), text)))
            rows = []
            for it, f in futs:
                r = f.result()
                r["type"] = it.get(s.get("type_key", "type"), "all")
                rows.append(r)
            res = {"all": aggregate(rows)}
            for t in sorted({r["type"] for r in rows}):
                res[t] = aggregate([r for r in rows if r["type"] == t])
            results[name] = res
            with open(Path(args.out).with_suffix(f".{name}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            a = res["all"]
            print(name, {k: round(a[k], 3) for k in ("f0_std", "f0_range", "f0_cv", "f0_move", "f0_micro",
                                                      "f0_reversals", "pauses", "rate_sps", "g_f0_std", "g_f0_range",
                                                      "g_f0_cv", "g_voiced_dropped_pct") if k in a}, flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
