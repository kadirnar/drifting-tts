"""Benchmark on an external text set: Freya-TR-Eval by default (495 everyday Turkish sentences, CC-BY-4.0).

Protocol of the FreyaTTS technical report (arXiv 2607.09530, Table 2), applied as published:

* every sentence is synthesised once (seed = sentence index) with a fixed voice (``--speaker``, default ``male``);
* the audio is band-matched: downsampled to 8 kHz before transcription, so wideband and telephony-band systems
  are scored alike (``--band 0`` keeps the full band);
* Whisper large-v3 (Turkish, beam 5) transcribes it; transcript and reference go through the same Turkish
  normaliser without punctuation; WER / CER are corpus-level with 95% bootstrap intervals.

UTMOSv2 is reported on the full-band audio as a relative naturalness proxy. The report's MOS column comes from a
listening study, so it is not comparable. Texts: a Hugging Face dataset id with a ``freya_tr_eval.jsonl`` file,
a local ``.jsonl`` (field ``text``) or a ``.txt`` file (one sentence per line).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .audio import SAMPLE_RATE
from .voices import DEFAULT_VOICE, VOICES

FREYA = "freyavoice/freya-tr-eval"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="TTS checkpoint")
    p.add_argument("--vocoder", default=None, help="fine-tuned vocoder (bigvgan_ft.pt / vocos_ft.pt)")
    p.add_argument("--texts", default=FREYA, help="HF dataset id, .jsonl (field 'text') or .txt")
    p.add_argument("--num", type=int, default=0, help="first N sentences only (0: all)")
    p.add_argument("--speaker", nargs="+", default=[DEFAULT_VOICE],
                   help=f"voice(s): {', '.join(VOICES)} or speaker IDs; with several, sentence i uses the (i %% n)-th")
    p.add_argument("--temperature", type=float, default=None, help="default: the checkpoint's preferred one")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--band", type=int, default=8000, help="band-match: resample to this rate before ASR (0: off)")
    p.add_argument("--asr", default="large-v3")
    p.add_argument("--mos", default="utmosv2", choices=["utmosv2", "utmos22", "none"])
    p.add_argument("--save-wavs", type=int, default=20)
    p.add_argument("--cuda-kernel", action="store_true")
    p.add_argument("--stream", action="store_true", help="synthesise with Synthesizer.stream (streaming vocoder)")
    p.add_argument("--fast", action="store_true", help="CUDA graphs, Synthesizer(fast=True); implies --stream")
    p.add_argument("--out", default="outputs/benchmark")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def load_texts(spec: str) -> list[dict]:
    """``[{"id", "text", ...}]`` from a HF dataset id, a .jsonl or a .txt file."""
    path = Path(spec)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(spec, "freya_tr_eval.jsonl", repo_type="dataset"))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{"id": f"{i:04d}", "text": t.strip()} for i, t in enumerate(path.read_text().splitlines()) if t.strip()]


def band_match(wav: torch.Tensor, band: int) -> np.ndarray:
    """24 kHz waveform -> 16 kHz ASR input, through ``band`` Hz first when ``band`` > 0."""
    import torchaudio

    x = wav.float().cpu()
    if band:
        x = torchaudio.functional.resample(x, SAMPLE_RATE, band)
        return torchaudio.functional.resample(x, band, 16_000).numpy()
    return torchaudio.functional.resample(x, SAMPLE_RATE, 16_000).numpy()


def run(args) -> None:
    from .evaluate import _plain, format_table, summarize
    from .judges import load_judges
    from .metrics import error_counts
    from .synthesize import Synthesizer

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    items = load_texts(args.texts)
    items = items[: args.num] if args.num else items
    synth = Synthesizer(args.model, args.device, vocoder=args.vocoder, cuda_kernel=args.cuda_kernel, fast=args.fast)
    temperature = synth.default_temperature if args.temperature is None else args.temperature
    judges = load_judges(args.asr, None, args.mos, args.device)

    def generate(text: str, speaker: str, seed: int) -> tuple[torch.Tensor, dict]:
        if not (args.stream or args.fast):
            return synth(text, speaker=speaker, cfg_scale=args.cfg, temperature=temperature, seed=seed)
        t0 = time.perf_counter()
        wav = torch.cat(list(synth.stream(text, speaker=speaker, cfg_scale=args.cfg, temperature=temperature,
                                          seed=seed)))
        return wav, {"rtf_total": (time.perf_counter() - t0) / max(wav.numel() / SAMPLE_RATE, 1e-6)}

    generate("Merhaba.", args.speaker[0], 0)  # warm-up for the RTF

    rows = []
    for i, item in enumerate(items):
        spk = args.speaker[i % len(args.speaker)]
        wav, info = generate(item["text"], spk, i)
        ref = _plain(item["text"])
        row = {"id": item.get("id", i), "speaker": spk, "ref": ref, "rtf": info["rtf_total"]}
        if judges.asr is not None:
            row["hyp"] = _plain(judges.asr(band_match(wav, args.band)))
            row.update(error_counts(ref, row["hyp"]))
        if judges.mos is not None:
            row["mos"] = judges.mos(band_match(wav, 0))
        rows.append(row)
        if i < args.save_wavs:
            sf.write(out / "wav" / f"{row['id']}.wav", wav.numpy(), SAMPLE_RATE)
    with open(out / "utterances.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = {"system": "drifting_tts", "temperature": temperature, "cfg": args.cfg, **summarize(rows),
           "rtf": float(np.mean([r["rtf"] for r in rows]))}
    results = {"texts": args.texts, "sentences": len(rows), "band_hz": args.band, "voices": args.speaker,
               "stream": args.stream or args.fast, "fast": args.fast,
               "model": args.model, "vocoder": args.vocoder or "stock", "asr": args.asr, "rows": [res]}
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / "results.md").write_text(format_table([res]) + "\n")
    print(format_table([res]))
    print(f"-> {out / 'results.json'}")
