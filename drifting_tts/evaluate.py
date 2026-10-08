"""Objective evaluation on held-out sentences: CER / WER, speaker similarity, UTMOSv2 and real-time factor.

Judges (see :mod:`drifting_tts.judges`; each optional):

* intelligibility: Whisper large-v3 (Turkish, beam 5, deterministic) transcripts. Hypotheses and references go
  through the same Turkish normaliser without punctuation. CER / WER are corpus-level edit-distance ratios with
  95% bootstrap intervals over utterances;
* speaker similarity: cosine of WavLM-Large + ECAPA-TDNN embeddings (the Seed-TTS-eval SIM model) with the
  *recording* of the same sentence (``prepare --save-audio``), or with its vocoded ground truth otherwise;
* naturalness: UTMOSv2. It is trained on English, so compare against the vocoded ground truth.

Rows: the recording (when stored), the vocoded ground truth (copy synthesis, the system's upper bound) and one
row per (noise temperature, guidance scale). Utterance ``i`` of the split always uses seed ``i``.

``--harmonic`` skips the vocoder and the judges. It generates mels under the ground-truth alignment (and
ground-truth pitch for pitch-conditioned models) and compares them with the recordings. The comparison reports,
per band, the harmonic contrast ratio (|second difference along mel frequency| on voiced frames), the
global-variance ratio and the level difference (:mod:`drifting_tts.metrics`).

Unbiased selection: tune temperature, guidance and steps on ``--split dev`` (``prepare --dev-size N``), or on a
disjoint ``--offset`` slice of ``val``, and report on ``val`` once.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .audio import SAMPLE_RATE
from .data import MelDataset, collate
from .judges import SV_MODELS, Judges
from .metrics import SpectralComparison, bootstrap_ci, error_counts, mel_center_freqs
from .synthesize import add_vocoder_args
from .text import normalize


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="runs/tts/model_ema.pt")
    p.add_argument("--split", default="val", choices=["val", "dev"],
                   help="held-out split: tune on dev (prepare --dev-size), report on val")
    p.add_argument("--offset", type=int, default=0, help="skip the first N utterances of the split")
    p.add_argument("--num", type=int, default=100, help="number of utterances")
    p.add_argument("--cfg", type=float, nargs="+", default=[1.0])
    p.add_argument("--temperature", type=float, nargs="+", default=None,
                   help="noise temperatures (default: the checkpoint's preferred one, else 0.5)")
    p.add_argument("--length-scale", type=float, default=1.0, help="on top of the calibrated duration_scale")
    p.add_argument("--attn-window", type=int, default=None, help="sliding-window attention radius (tokens)")
    p.add_argument("--steps", type=int, default=None, help="generator evaluations (default: as trained)")
    p.add_argument("--harmonic", action="store_true",
                   help="only spectral statistics under the ground-truth alignment (no vocoder, no judges)")
    p.add_argument("--asr", default="large-v3", help="faster-whisper model (large-v3-turbo: faster); 'none' disables")
    p.add_argument("--sv", default="wavlm-large-ecapa",
                   help=f"speaker-verification model: {', '.join(SV_MODELS)}; 'none' disables")
    p.add_argument("--mos", default="utmosv2", choices=["utmosv2", "utmos22", "none"], help="naturalness predictor")
    p.add_argument("--mos-repetitions", type=int, default=1,
                   help="UTMOSv2 random crops averaged per utterance (more: less noise, slower)")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--save-wavs", type=int, default=10, help="wavs written per row")
    p.add_argument("--no-gt", action="store_true", help="skip the recording / copy-synthesis rows")
    add_vocoder_args(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def _plain(text: str) -> str:
    """Normalised text without punctuation: what CER / WER compare."""
    return re.sub(r"\s+", " ", re.sub(r"[.,!?]", " ", normalize(text))).strip()


def to16k(wav: torch.Tensor) -> np.ndarray:
    return torchaudio.functional.resample(wav.float().cpu(), SAMPLE_RATE, 16_000).numpy()


def score_utterance(judges: Judges, wav: torch.Tensor, ref: str,
                    spk_ref: torch.Tensor | None = None) -> tuple[dict, torch.Tensor | None]:
    """Judge one 24 kHz utterance against its plain reference text (and reference speaker embedding).

    Returns the utterance row and its speaker embedding."""
    w16 = to16k(wav)
    row: dict = {"ref": ref}
    if judges.asr is not None:
        row["hyp"] = _plain(judges.asr(w16))
        row.update(error_counts(ref, row["hyp"]))
    emb = judges.sv(w16) if judges.sv is not None else None
    if emb is not None and spk_ref is not None:
        row["speaker_sim"] = float(emb @ spk_ref)
    if judges.mos is not None:
        row["mos"] = judges.mos(w16)
    return row, emb


def summarize(rows: list[dict]) -> dict:
    """Corpus-level CER / WER and mean speaker similarity / MOS, each with a 95% bootstrap interval."""
    out: dict = {}
    if rows and "hyp" in rows[0]:
        for name, unit in (("cer", "char"), ("wer", "word")):
            err, n = [r[f"{unit}_errors"] for r in rows], [r[f"{unit}s"] for r in rows]
            out[name], out[f"{name}_ci"] = sum(err) / max(sum(n), 1), bootstrap_ci(err, n)
    for k in ("speaker_sim", "mos"):
        v = [r[k] for r in rows if k in r]
        if v:
            out[k], out[f"{k}_ci"] = float(np.mean(v)), bootstrap_ci(v)
    return out


def select(ds: MelDataset, split: str, offset: int, num: int) -> list[int]:
    if offset >= len(ds):
        hint = " (create one with `drifting-tts prepare --dev-size N`)" if split == "dev" else ""
        raise SystemExit(f"--offset {offset} but the '{split}' split has {len(ds)} utterances{hint}")
    return list(range(offset, min(offset + num, len(ds))))


def format_table(rows: list[dict]) -> str:
    cols = [c for c in ("system", "temperature", "cfg", "cer", "wer", "speaker_sim", "mos", "rtf", "hc_low", "hc_mid",
                        "hc_high", "gv_low", "gv_mid", "gv_high", "level_db_low", "level_db_mid", "level_db_high")
            if any(c in r for r in rows)]

    def fmt(c, v):
        if v is None or v == "":
            return "–"
        if c in ("cer", "wer"):
            return f"{100 * v:.2f}%"
        if c in ("temperature", "cfg"):
            return f"{v:g}"
        return f"{v:.4f}" if c == "rtf" else f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(c, r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def _write(out: Path, results: dict) -> None:
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / "results.md").write_text(format_table(results["rows"]) + "\n")
    print(format_table(results["rows"]))
    print(f"-> {out / 'results.json'}")


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.harmonic:
        return run_harmonic(args, out)
    from .judges import load_judges
    from .synthesize import Synthesizer

    synth = Synthesizer(args.model, args.device, vocoder=args.vocoder, cuda_kernel=args.cuda_kernel)
    temps = args.temperature or [synth.default_temperature]
    root = Path(synth.cfg.data.root)
    with_audio = (root / "audio.bin").exists() and not args.no_gt
    ds = MelDataset(root, args.split, min_frames=1, max_frames=10**9, with_audio=with_audio)
    idx = select(ds, args.split, args.offset, args.num)
    judges = load_judges(args.asr, args.sv, args.mos, args.device, args.mos_repetitions)
    refs = [_plain(ds.items[i]["norm_text"]) for i in idx]
    spk_refs: list[torch.Tensor | None] = [None] * len(idx)
    (out / "wav").mkdir(exist_ok=True)

    def evaluate_system(tag: str, wavs, info: dict) -> dict:
        utts, rtf = [], []
        for k, (wav, extra) in enumerate(wavs):
            row, emb = score_utterance(judges, wav, refs[k], spk_refs[k])
            if info["system"] != "drifting_tts" and spk_refs[k] is None:
                spk_refs[k] = emb  # the first ground-truth row is the speaker reference
            utts.append({"index": idx[k], **row, **extra})
            rtf += [extra["rtf"]] if "rtf" in extra else []
            if k < args.save_wavs:
                sf.write(out / "wav" / f"{tag}_{idx[k]:04d}.wav", wav.numpy(), SAMPLE_RATE)
        with open(out / f"utterances_{tag}.jsonl", "w") as f:
            f.writelines(json.dumps(u, ensure_ascii=False) + "\n" for u in utts)
        row = {**info, **summarize(utts)}
        if rtf:
            row["rtf"] = float(np.mean(rtf))
        print(json.dumps(row), flush=True)
        return row

    e = ds.items[idx[0]]
    synth(e["norm_text"], speaker=e["spk_id"], temperature=temps[0])  # warm-up, so that RTF excludes CUDA init
    rows = []
    if with_audio:
        rows.append(evaluate_system("recording", ((ds[i]["audio"], {}) for i in idx), {"system": "recording"}))
    if not args.no_gt:
        gt = ((synth.vocoder(ds.stats.denormalize(ds[i]["mel"])[None])[0].cpu(), {}) for i in idx)
        rows.append(evaluate_system("gt_vocoded", gt, {"system": "ground_truth_vocoded"}))
    for t in temps:
        for c in args.cfg:
            def generate(t=t, c=c):
                for i in idx:
                    e = ds.items[i]
                    wav, info = synth(e["norm_text"], speaker=e["spk_id"], cfg_scale=c, temperature=t,
                                      length_scale=args.length_scale, seed=i, attn_window=args.attn_window,
                                      steps=args.steps)
                    yield wav, {"seed": i, "seconds": info["seconds"], "rtf": info["rtf_total"]}

            rows.append(evaluate_system(f"T{t:g}_cfg{c:g}", generate(),
                                        {"system": "drifting_tts", "temperature": t, "cfg": c}))
    _write(out, {
        "model": str(args.model), "split": args.split, "offset": args.offset, "num_utterances": len(idx),
        "steps": args.steps, "attn_window": args.attn_window, "length_scale": args.length_scale,
        "duration_scale": synth.model.duration_scale, "vocoder": args.vocoder or "stock",
        "judges": {"asr": args.asr if judges.asr else None, "sv": args.sv if judges.sv else None,
                   "mos": args.mos if judges.mos else None},
        "speaker_reference": "recording" if with_audio else "ground_truth_vocoded" if not args.no_gt else None,
        "rows": rows,
    })


@torch.no_grad()
def run_harmonic(args, out: Path) -> None:
    from .finetune_vocoder import gta_mels
    from .synthesize import preferred_temperature
    from .train import load_tts

    model, cfg, stats = load_tts(args.model, args.device)
    temps = args.temperature or [preferred_temperature(model)]
    root = Path(cfg.data.root)
    with_f0 = (root / "f0.bin").exists()
    ds = MelDataset(root, args.split, min_frames=1, max_frames=10**9, with_f0=with_f0)
    idx = select(ds, args.split, args.offset, args.num)
    centers = mel_center_freqs(stats.get("backend", "vocos"))
    rows = []
    for t in temps:
        for c in args.cfg:
            acc = SpectralComparison(centers)
            for i in idx:
                batch = collate([ds[i]])  # one utterance at a time: no padding
                g = torch.Generator(device=args.device).manual_seed(i)
                gen = gta_mels(model, batch, t, args.device, cfg_scale=c, steps=args.steps, generator=g)[0]
                acc.add(gen.float().cpu() * stats["std"] + stats["mean"], ds.stats.denormalize(batch["mel"][0]),
                        batch["f0"][0] > 0 if with_f0 else None)
            rows.append({"system": "drifting_tts_gta", "temperature": t, "cfg": c, **acc.summary()})
            print(json.dumps(rows[-1]), flush=True)
    pitch = ("ground_truth" if with_f0 else "predicted") if model.pitch_enabled else None
    _write(out, {"model": str(args.model), "split": args.split, "offset": args.offset, "num_utterances": len(idx),
                 "steps": args.steps, "alignment": "ground_truth (MAS)", "pitch": pitch,
                 "voicing": "f0 > 0" if with_f0 else "frames louder than the utterance median", "rows": rows})
