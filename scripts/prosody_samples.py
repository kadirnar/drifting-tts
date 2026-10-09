"""Listening samples for the stochastic prosody predictor (#39): the same texts under the deterministic regressors and
the sampler at several prosody temperatures and seeds, plus the recording when the text is a held-out utterance.

Held-out texts come from the `val` / `dev` splits of the data (studio voice 722; male 389 and female 323 have only
a few), external texts from Freya-TR-Eval. Generated with `Synthesizer` as a user would (sentence by sentence,
T = 0.3, alpha = 2, vocos-ft). Writes <out>/<voice>_<n>_<system>.wav and <out>/index.md. Recordings stay local.

    python scripts/prosody_samples.py --model runs/release/drifting_tts_v3.1.pt \
        --prosody drift=runs/pm_drift/prosody_ema.pt flow=runs/pm_flow/prosody_ema.pt \
        --settings drift:1 drift:0.5 drift:1:0.8 flow:1 --seeds 1 2 --out runs/pm_samples
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
    p.add_argument("--prosody", nargs="+", required=True, help="name=checkpoint")
    p.add_argument("--settings", nargs="+", required=True,
                   help="name:temperature[:spread][:pitch] (each rendered with every --seeds; pitch: only the token "
                        "pitch is sampled, the durations stay the regressors')")
    p.add_argument("--data", default="/workspace/data/tr12_eleven")
    p.add_argument("--vocoder", default="vocos-ft")
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
    preds = {}
    for spec in args.prosody:
        name, path = spec.split("=", 1)
        preds[name] = Synthesizer(args.model, args.device, vocoder=args.vocoder, prosody=path)
    settings = []
    for spec in args.settings:
        name, *v = spec.split(":")
        pitch_only = "pitch" in v
        v = [x for x in v if x != "pitch"]
        settings.append((name, float(v[0]), float(v[1]) if len(v) > 1 else 1.0, pitch_only))
    freya = load_texts(FREYA)
    items = [("studio", 722, r) for r in heldout(root, 722, args.num_studio)]
    for voice, spk in (("male", 389), ("female", 323)):
        items += [(voice, spk, r) for r in heldout(root, spk, 3)]
        items += [(voice, spk, {"norm_text": freya[i]["text"]}) for i in FREYA_IDS]
    kw = dict(cfg_scale=2.0, temperature=0.3)
    lines = ["# Prosody listening samples (#39)", "",
             "v3.1 + vocos-ft, T = 0.3, α = 2, sentence by sentence (0.15 s pauses). `v31`: the deterministic "
             "duration / pitch regressors (seed 0). `<name>_T<t>[_S<spread>][_pitch]_s<k>`: a stochastic prosody "
             "predictor at prosody temperature t (output-space spread; `_pitch`: only the token pitch is sampled, "
             "the durations stay v3.1's), seed k; the seed also drives the generator noise. "
             f"Predictors: {', '.join(args.prosody)}. "
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
        if not (out / f"{stem}_v31.wav").exists():
            wav, _ = base(r["norm_text"], speaker=spk, seed=0, **kw)
            sf.write(out / f"{stem}_v31.wav", wav.numpy(), SAMPLE_RATE)
        files.append(f"{stem}_v31.wav")
        for name, T, spread, pitch_only in settings:
            synth = preds[name]
            synth.prosody_temperature, synth.prosody_spread = T, spread
            synth.prosody_durations = "regressor" if pitch_only else "sampled"
            for seed in args.seeds:
                tag = (f"{name}_T{T:g}" + (f"_S{spread:g}" if spread != 1.0 else "") + ("_pitch" if pitch_only else "")
                       + f"_s{seed}")
                if not (out / f"{stem}_{tag}.wav").exists():  # re-runs only add what is missing
                    wav, _ = synth(r["norm_text"], speaker=spk, seed=seed, **kw)
                    sf.write(out / f"{stem}_{tag}.wav", wav.numpy(), SAMPLE_RATE)
                files.append(f"{stem}_{tag}.wav")
        lines += [f"## {stem} (speaker {spk}{', held-out' if 'audio_offset' in r else ', Freya-TR-Eval'})", "",
                  f"> {r['norm_text']}", ""] + [f"- [{f}]({f})" for f in files] + [""]
        print(stem, len(files), flush=True)
    (out / "index.md").write_text("\n".join(lines))
    print(f"-> {out / 'index.md'}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
