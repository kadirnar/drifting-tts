"""Listening samples for the stochastic prosody predictor (#39): the same texts under the deterministic regressors and
the sampler at several prosody temperatures and seeds, plus the recording when the text is a held-out utterance.

Held-out texts come from the `val` / `dev` splits of the data (studio voice 722; male 389 and female 323 have only
a few), external texts from Freya-TR-Eval. Generated with `Synthesizer` as a user would (sentence by sentence,
T = 0.3, alpha = 2, vocos-ft). Writes <out>/<voice>_<n>_<system>.wav and <out>/index.md. Recordings stay local.

    python scripts/prosody_samples.py --model runs/release/drifting_tts_v3.1.pt --prosody runs/pm_drift/prosody_ema.pt \
        --temperatures 0.5 0.8 1.0 --seeds 1 2 --out runs/pm_samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer

FREYA_IDS = [0, 7, 21]  # Freya-TR-Eval sentences for the voices without held-out recordings


def heldout(root: Path, spk: int, num: int) -> list[dict]:
    rows = [json.loads(line) for line in open(root / "index.jsonl")]
    return [r for r in rows if r["split"] in ("val", "dev") and r["spk_id"] == spk][:num]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prosody", required=True)
    p.add_argument("--data", default="/workspace/data/tr12_eleven")
    p.add_argument("--vocoder", default="vocos-ft")
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--num-studio", type=int, default=6)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from drifting_tts.benchmark import FREYA, load_texts

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data)
    audio = np.memmap(root / "audio.bin", dtype=np.float16, mode="r")
    base = Synthesizer(args.model, args.device, vocoder=args.vocoder)
    drift = Synthesizer(args.model, args.device, vocoder=args.vocoder, prosody=args.prosody)
    freya = load_texts(FREYA)
    items = [("studio", 722, r) for r in heldout(root, 722, args.num_studio)]
    for voice, spk in (("male", 389), ("female", 323)):
        items += [(voice, spk, r) for r in heldout(root, spk, 3)]
        items += [(voice, spk, {"norm_text": freya[i]["text"]}) for i in FREYA_IDS]
    kw = dict(cfg_scale=2.0, temperature=0.3)
    lines = ["# Prosody listening samples (#39)", "",
             f"v3.1 + vocos-ft, T = 0.3, α = 2, sentence by sentence (0.15 s pauses). `v31`: the deterministic "
             f"duration / pitch regressors (seed 0). `drift_T<t>_s<k>`: the stochastic prosody predictor "
             f"`{args.prosody}` at prosody temperature t, seed k (the seed also drives the generator noise). "
             "`recording`: the held-out recording of the same text (local only, never publish).", ""]
    counts: dict[str, int] = {}
    for voice, spk, r in items:
        n = counts[voice] = counts.get(voice, 0) + 1
        stem = f"{voice}_{n}"
        files = []
        if "audio_offset" in r:
            a = np.array(audio[r["audio_offset"]: r["audio_offset"] + r["audio_samples"]], dtype=np.float32)
            sf.write(out / f"{stem}_recording.wav", a, SAMPLE_RATE)
            files.append(f"{stem}_recording.wav")
        wav, _ = base(r["norm_text"], speaker=spk, seed=0, **kw)
        sf.write(out / f"{stem}_v31.wav", wav.numpy(), SAMPLE_RATE)
        files.append(f"{stem}_v31.wav")
        for T in args.temperatures:
            drift.prosody_temperature = T
            for seed in args.seeds:
                wav, _ = drift(r["norm_text"], speaker=spk, seed=seed, **kw)
                name = f"{stem}_drift_T{T:g}_s{seed}.wav"
                sf.write(out / name, wav.numpy(), SAMPLE_RATE)
                files.append(name)
        lines += [f"## {stem} (speaker {spk}{', held-out' if 'audio_offset' in r else ', Freya-TR-Eval'})", "",
                  f"> {r['norm_text']}", ""] + [f"- [{f}]({f})" for f in files] + [""]
        print(stem, len(files), flush=True)
    (out / "index.md").write_text("\n".join(lines))
    print(f"-> {out / 'index.md'}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
