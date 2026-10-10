"""Synthesise Freya-TR-Eval sentences with a Pocket TTS config on the CPU (docs/POCKET_TTS_GATE.md): one wav per
sentence, ``torch.manual_seed(sentence index)``, the original (cased, punctuated) text. Records the wall time, the
audio length and the time to the first streamed chunk in ``gen.jsonl``.

Runs in a separate venv with the ``pocket-tts`` package (not a dependency of this repo):

    python scripts/pocket_tts_gate/gen_pocket.py --config pocket_tts_tr.yaml --voice prompt.wav --out runs/pg_freya
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pocket_tts import TTSModel


def load_texts(spec: str) -> list[dict]:
    """``[{"index", "id", "text"}]`` from a .jsonl or the ``freya_tr_eval.jsonl`` of a HF dataset."""
    path = Path(spec)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(spec, "freya_tr_eval.jsonl", repo_type="dataset"))
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{**it, "index": it.get("index", i), "id": it.get("id", f"{i:04d}")} for i, it in enumerate(items)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--texts", default="freyavoice/freya-tr-eval", help="HF dataset id or a .jsonl (field 'text')")
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--temp", type=float, default=None)
    args = p.parse_args()

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = TTSModel.load_model(config=args.config, temp=args.temp)
    torch.set_num_threads(args.threads)  # pocket_tts sets 1 at import; the Mimi decoder runs in its own thread
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    state = model.get_state_for_audio_prompt(Path(args.voice))
    prompt_s = time.perf_counter() - t0
    sr = model.sample_rate
    items = load_texts(args.texts)[args.offset: args.offset + args.num]

    torch.manual_seed(12345)
    for _ in model.generate_audio_stream(state, "Merhaba, nasılsınız?"):  # warm-up
        pass

    rows_path = out / "gen.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(line)["index"] for line in open(rows_path)}
    with open(rows_path, "a") as f:
        for it in items:
            i = it["index"]
            if i in done:
                continue
            torch.manual_seed(i)
            chunks, first = [], None
            t = time.perf_counter()
            for c in model.generate_audio_stream(state, it["text"]):
                if first is None:
                    first = time.perf_counter() - t
                chunks.append(c)
            wall = time.perf_counter() - t
            wav = torch.cat(chunks).float().numpy()
            sf.write(out / "wav" / f"{i:05d}.wav", np.clip(wav, -1, 1), sr, subtype="PCM_16")
            sec = len(wav) / sr
            row = {"index": i, "id": it["id"], "seconds": sec, "wall_s": wall, "rtf": wall / sec,
                   "first_chunk_s": first, "peak": float(np.abs(wav).max())}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)
    meta = {"config": args.config, "voice": args.voice, "threads": args.threads, "temp": model.temp,
            "sampler_decode_steps": model.sampler_decode_steps, "eos_threshold": model.eos_threshold,
            "load_s": load_s, "prompt_s": prompt_s, "sample_rate": sr, "torch": torch.__version__,
            "omp": os.environ.get("OMP_NUM_THREADS")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
