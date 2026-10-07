"""Rank candidate voices on held-out dev sentences (never the reported test sets) by UTMOSv2 and Whisper CER.

    python scripts/select_voices.py --model runs/tts_v3/model_ema.pt --vocoder runs/vocoder_v3/bigvgan_ft.pt \\
        --data data/train --candidates voices.json --num 30 --out voice_ranking.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from drifting_tts.benchmark import band_match
from drifting_tts.data import MelDataset
from drifting_tts.evaluate import _plain
from drifting_tts.judges import load_judges
from drifting_tts.metrics import error_counts
from drifting_tts.synthesize import Synthesizer


def median_f0(data: Path, spk: int) -> float:
    """Median voiced F0 (Hz) of a speaker's training utterances (``prepare --f0``)."""
    f0 = np.memmap(data / "f0.bin", dtype=np.float16, mode="r")
    vals = []
    for line in open(data / "index.jsonl"):
        e = json.loads(line)
        if e["spk_id"] == spk and e["split"] == "train":
            x = np.asarray(f0[e["offset"]: e["offset"] + e["frames"]], dtype=np.float32)
            vals.append(x[x > 0])
    return float(np.median(np.concatenate(vals))) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", default=None)
    p.add_argument("--data", default="data/train")
    p.add_argument("--candidates", required=True, help="json list of {'id': speaker id, ...}")
    p.add_argument("--num", type=int, default=30, help="dev sentences per voice")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--out", default="voice_ranking.json")
    args = p.parse_args()
    data = Path(args.data)
    texts = [e["norm_text"] for e in MelDataset(data, "dev", min_frames=1, max_frames=10**9).items[: args.num]]
    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder)
    judges = load_judges("large-v3", None, "utmosv2", "cuda")
    rows = []
    for cand in json.loads(Path(args.candidates).read_text()):
        spk, ce, cn, mos = cand["id"], 0, 0, []
        for i, text in enumerate(texts):
            wav, _ = synth(text, speaker=spk, cfg_scale=args.cfg, temperature=args.temperature, seed=i)
            ref = _plain(text)
            c = error_counts(ref, _plain(judges.asr(band_match(wav, 0))))
            ce, cn = ce + c["char_errors"], cn + c["chars"]
            mos.append(judges.mos(band_match(wav, 0)))
        rows.append({"id": spk, "utmosv2": float(np.mean(mos)), "cer": ce / cn, "f0_hz": median_f0(data, spk),
                     "train_minutes": cand.get("minutes")})
        print(rows[-1], flush=True)
    rows.sort(key=lambda r: -r["utmosv2"])
    Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
