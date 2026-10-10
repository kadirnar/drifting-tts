"""Mimi codec gate inputs (docs/POCKET_TTS_GATE.md): the studio-voice (speaker 722) and the first base-corpus ``val``
recordings, written as float32 wavs (local only: recording-derived audio is never published), plus ``vocos-ft`` copy
synthesis from the dataset's BigVGAN-style mels. CPU.

    python scripts/pocket_tts_gate/dump_recordings.py --data data/train --out runs/pg_mimi
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.data import MelDataset
from drifting_tts.evaluate import _plain
from drifting_tts.vocoder import load_vocoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="prepared dataset with audio.bin")
    p.add_argument("--out", default="runs/pg_mimi")
    p.add_argument("--vocoder", default="vocos-ft")
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--speaker", type=int, default=722, help="the studio voice; the base set is every other speaker")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    ds = MelDataset(args.data, "val", min_frames=1, max_frames=10**9, with_audio=True)
    sets = {"studio": [i for i, e in enumerate(ds.items) if e["spk_id"] == args.speaker][: args.num],
            "base": [i for i, e in enumerate(ds.items) if e["spk_id"] != args.speaker][: args.num]}
    voc = load_vocoder(args.vocoder, "cpu")
    for name, idx in sets.items():
        (out / "recording" / name).mkdir(parents=True, exist_ok=True)
        (out / "vocos_ft" / name).mkdir(parents=True, exist_ok=True)
        with open(out / f"{name}.jsonl", "w") as f:
            for i in idx:
                it, e = ds[i], ds.items[i]
                wav = it["audio"].float()
                sf.write(out / "recording" / name / f"{i:05d}.wav", wav.numpy(), SAMPLE_RATE, subtype="FLOAT")
                y = voc(ds.stats.denormalize(it["mel"])[None].float())[0][: len(wav)]  # the dataset item is normalised
                sf.write(out / "vocos_ft" / name / f"{i:05d}.wav", y.numpy(), SAMPLE_RATE, subtype="FLOAT")
                f.write(json.dumps({"ds_index": i, "spk_id": e["spk_id"], "ref": _plain(e["norm_text"]),
                                    "seconds": len(wav) / SAMPLE_RATE, "vocoded_samples": int(y.numel()),
                                    "samples": int(wav.numel())}, ensure_ascii=False) + "\n")
        print(name, len(idx), flush=True)


if __name__ == "__main__":
    main()
