"""Benchmark on an external text set: Freya-TR-Eval by default (495 everyday Turkish sentences, CC-BY-4.0).

Protocol of the FreyaTTS technical report (arXiv 2607.09530, Table 2), applied as published:

* every sentence is synthesised once (seed = sentence index) with a fixed voice (``--speaker``, default ``male``);
* the audio is band-matched: downsampled to 8 kHz before transcription, so wideband and telephony-band systems
  are scored alike (``--band 0`` keeps the full band);
* Whisper large-v3 (beam 5) transcribes it in the checkpoint's language (``model.text.language``); transcript and
  reference go through that language's normaliser without punctuation; WER / CER are corpus-level with 95% bootstrap
  intervals.

UTMOSv2 is reported on the full-band audio as a relative naturalness proxy. The report's MOS column comes from a
listening study, so it is not comparable. Texts: a Hugging Face dataset id (``<id>`` with a ``freya_tr_eval.jsonl``
file, or ``<id>:<file>`` for another ``.jsonl`` / ``.txt`` file of it), a local ``.jsonl`` (field ``text``) or a
``.txt`` file (one sentence per line). The default set is Turkish: a checkpoint of another language needs ``--texts``.
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
from .synthesize import add_vocoder_args, pause_arg
from .voices import DEFAULT_VOICE, VOICES

FREYA = "freyavoice/freya-tr-eval"
DEFAULT_TEXTS = {"tr": FREYA}  # per language of the checkpoint
WARMUP = {"tr": "Merhaba.", "en": "Hello."}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="TTS checkpoint")
    add_vocoder_args(p)
    p.add_argument("--texts", default=None,
                   help=f"HF dataset id (or <id>:<file>), .jsonl (field 'text') or .txt (default for a Turkish "
                        f"checkpoint: {FREYA})")
    p.add_argument("--num", type=int, default=0, help="first N sentences only (0: all)")
    p.add_argument("--speaker", nargs="+", default=[DEFAULT_VOICE],
                   help=f"voice(s): {', '.join(VOICES)} or speaker IDs; with several, sentence i uses the (i %% n)-th")
    p.add_argument("--temperature", type=float, default=None, help="default: the checkpoint's preferred one")
    p.add_argument("--cfg", type=float, default=None,
                   help="guidance scale (default: the checkpoint's preferred one, else 2 for the DiT, 1 for ARDrift)")
    p.add_argument("--band", type=int, default=8000, help="band-match: resample to this rate before ASR (0: off)")
    p.add_argument("--asr", default="large-v3")
    p.add_argument("--mos", default="utmosv2", choices=["utmosv2", "utmos22", "none"])
    p.add_argument("--save-wavs", type=int, default=20)
    p.add_argument("--stream", action="store_true", help="synthesise with Synthesizer.stream (streaming vocoder)")
    p.add_argument("--fast", action="store_true", help="CUDA graphs, Synthesizer(fast=True); implies --stream")
    p.add_argument("--prosody", default=None, help="stochastic prosody predictor: drift or a train-prosody checkpoint")
    p.add_argument("--prosody-temperature", type=float, default=None, help="default: the checkpoint's preferred one")
    p.add_argument("--prosody-spread", type=float, default=1.0)
    p.add_argument("--prosody-durations", choices=["sampled", "regressor", "pauses"], default="sampled")
    p.add_argument("--prosody-duration-temperature", type=float, default=None,
                   help="noise temperature of the sampled durations (default: the checkpoint's preferred one, else "
                        "--prosody-temperature)")
    p.add_argument("--prosody-pitch-model", default=None,
                   help="a second prosody predictor that samples the token pitch; the durations stay --prosody's")
    p.add_argument("--paragraph", type=int, nargs="?", const=180, default=None, metavar="CHARS",
                   help="paragraph mode (Synthesizer(paragraph=CHARS)): the sentences of an item in one pass; 0: "
                        "sentence by sentence (default: on for ARDrift, off for the DiT)")
    p.add_argument("--acoustic-context", type=int, default=None, metavar="FRAMES",
                   help="ARDrift: frames of the audio before a unit that it continues from "
                        "(Synthesizer(acoustic_context=FRAMES); 0: every unit starts from silence)")
    p.add_argument("--context", action="store_true",
                   help="Synthesizer(context=True): each sentence conditioned on the text before it")
    p.add_argument("--join", type=int, default=1,
                   help="paragraph test set: every K consecutive texts joined into one item (after --num)")
    p.add_argument("--pause", type=pause_arg, default=0.15,
                   help="silence between the sentences of a multi-sentence item: seconds or 'punct' (per voice)")
    p.add_argument("--chunked", action="store_true",
                   help="stream the DiT on frame windows (drifting_tts.chunked; implies --stream)")
    p.add_argument("--chunk-right", type=int, default=64, help="--chunked: the DiT's lookahead in frames")
    p.add_argument("--chunk-left", type=int, default=32, help="--chunked: left context of the later windows")
    p.add_argument("--chunk-size", type=int, default=256, help="--chunked: frames committed per later window")
    p.add_argument("--crossfade", type=int, default=16, help="--chunked: frames blended at each join")
    p.add_argument("--noise", choices=["torch", "philox"], default="torch",
                   help="--chunked: the noise scheme (philox: counter-based, drifting_tts.noise)")
    p.add_argument("--dit-dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="--batch: the DiT's precision (autocast)")
    p.add_argument("--prosody-dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="--batch: the prosody predictor network's precision (autocast)")
    p.add_argument("--fast-serving", action="store_true",
                   help="Serving.fast(): buckets, fp16 text encoder / prosody network / DiT / vocoder, all compiled")
    p.add_argument("--buckets", type=int, default=1, help="--batch: length buckets of the batched passes")
    p.add_argument("--compile-dit", action="store_true", help="--batch: torch.compile the batched DiT")
    p.add_argument("--text-dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="--batch: the text encoder's precision (autocast; the duration rounding stays fp32)")
    p.add_argument("--vocoder-dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="--batch: the batched vocoder's precision (autocast)")
    p.add_argument("--compile-text", action="store_true", help="--batch: torch.compile the text pass")
    p.add_argument("--min-bucket", type=int, default=64, help="--batch: rows per length bucket at least")
    p.add_argument("--pipeline", action="store_true", help="first round group by group, each yielded when ready")
    p.add_argument("--compile-vocoder", action="store_true", help="torch.compile the batched vocoder")
    p.add_argument("--batch", type=int, default=0,
                   help="synthesise the sentences N at a time with drifting_tts.batched.stream_batched (the serving "
                        "path; one voice; seed = sentence index as without it; 0: one at a time)")
    p.add_argument("--out", default="outputs/benchmark")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def load_texts(spec: str) -> list[dict]:
    """``[{"id", "text", ...}]`` from a .jsonl or a .txt file, or from a HF dataset: ``<id>`` (its
    ``freya_tr_eval.jsonl``) or ``<id>:<file>``."""
    path = Path(spec)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        repo, _, filename = spec.partition(":")
        path = Path(hf_hub_download(repo, filename or "freya_tr_eval.jsonl", repo_type="dataset"))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{"id": f"{i:04d}", "text": t.strip()} for i, t in enumerate(path.read_text().splitlines()) if t.strip()]


def join_items(items: list[dict], k: int) -> list[dict]:
    """A paragraph test set: every ``k`` consecutive items as one (texts joined with a space, the first one's id)."""
    return [{**items[i], "text": " ".join(it["text"].strip() for it in items[i: i + k])}
            for i in range(0, len(items) - k + 1, k)]


def band_match(wav: torch.Tensor, band: int) -> np.ndarray:
    """24 kHz waveform -> 16 kHz ASR input, through ``band`` Hz first when ``band`` > 0."""
    import torchaudio

    x = wav.float().cpu()
    if band:
        x = torchaudio.functional.resample(x, SAMPLE_RATE, band)
        return torchaudio.functional.resample(x, band, 16_000).numpy()
    return torchaudio.functional.resample(x, SAMPLE_RATE, 16_000).numpy()


def batch_synthesize(synth, items: list[dict], args, cfg: float, temperature: float, chunked,
                     serving) -> list[tuple[torch.Tensor, dict]]:
    """Every sentence through :func:`drifting_tts.batched.stream_batched`, ``args.batch`` at a time (seed = sentence
    index): ``(waveform, {"rtf_total"})`` per sentence, the RTF of its batch."""
    from .batched import stream_batched

    if len(args.speaker) != 1:
        raise ValueError("--batch synthesises one voice")
    out = []
    for s in range(0, len(items), args.batch):
        idx = list(range(s, min(s + args.batch, len(items))))
        pieces = {i: [] for i in idx}
        t0 = time.perf_counter()
        for rnd in stream_batched(synth, [items[i]["text"] for i in idx], speaker=args.speaker[0],
                                  cfg_scale=cfg, temperature=temperature, seeds=idx, pause=None, chunked=chunked,
                                  serving=serving):
            for j, p in rnd:
                pieces[idx[j]].append(p)
        wavs = [torch.cat(pieces[i]) for i in idx]
        rtf = (time.perf_counter() - t0) / max(sum(w.numel() for w in wavs) / SAMPLE_RATE, 1e-6)
        out += [(w, {"rtf_total": rtf}) for w in wavs]
    return out


def run(args) -> None:
    from .evaluate import _plain, format_table, summarize
    from .judges import load_judges
    from .metrics import error_counts
    from .synthesize import Synthesizer

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    synth = Synthesizer(args.model, args.device, vocoder=args.vocoder, cuda_kernel=args.cuda_kernel, fast=args.fast,
                        prosody=args.prosody, prosody_temperature=args.prosody_temperature,
                        prosody_spread=args.prosody_spread, prosody_durations=args.prosody_durations, pause=args.pause,
                        prosody_duration_temperature=args.prosody_duration_temperature,
                        prosody_pitch=args.prosody_pitch_model,
                        paragraph=None if args.paragraph is None else args.paragraph or False,
                        context=args.context, acoustic_context=args.acoustic_context)
    language = synth.language  # of the checkpoint: Whisper's language and the normaliser of both texts
    texts = args.texts or DEFAULT_TEXTS.get(language)
    if texts is None:
        raise SystemExit(f"--texts is required for a {language!r} checkpoint (the default set, {FREYA}, is Turkish)")
    items = load_texts(texts)
    items = items[: args.num] if args.num else items
    if args.join > 1:
        items = join_items(items, args.join)
    cfg, temperature = synth.operating_point(args.cfg, args.temperature)
    judges = load_judges(args.asr, None, args.mos, args.device, language=language)

    chunked = None
    if args.chunked:
        from .chunked import Chunking

        chunked = Chunking(right=args.chunk_right, left=args.chunk_left, chunk=args.chunk_size,
                           crossfade=args.crossfade, noise=args.noise)
    serving = None
    if args.batch:
        from .batched import Serving

        serving = Serving(buckets=args.buckets, min_bucket=args.min_bucket, dit_dtype=args.dit_dtype,
                          prosody_dtype=args.prosody_dtype, compile=args.compile_dit, text_dtype=args.text_dtype,
                          compile_text=args.compile_text, vocoder_dtype=args.vocoder_dtype,
                          compile_vocoder=args.compile_vocoder, pipeline=args.pipeline)
        if args.fast_serving:
            serving = Serving.fast(pipeline=args.pipeline)
    elif (args.dit_dtype, args.prosody_dtype, args.text_dtype, args.vocoder_dtype) != ("fp32",) * 4 \
            or args.buckets != 1 or args.compile_dit or args.compile_text or args.compile_vocoder \
            or args.fast_serving or args.pipeline:
        raise ValueError("--fast-serving, --buckets, --*-dtype, --compile-* and --pipeline apply to --batch")

    def generate(text: str, speaker: str, seed: int) -> tuple[torch.Tensor, dict]:
        if not (args.stream or args.fast or chunked):
            return synth(text, speaker=speaker, cfg_scale=cfg, temperature=temperature, seed=seed)
        t0 = time.perf_counter()
        wav = torch.cat(list(synth.stream(text, speaker=speaker, cfg_scale=cfg, temperature=temperature,
                                          seed=seed, chunked=chunked)))
        return wav, {"rtf_total": (time.perf_counter() - t0) / max(wav.numel() / SAMPLE_RATE, 1e-6)}

    generate(WARMUP.get(language, items[0]["text"]), args.speaker[0], 0)  # warm-up for the RTF
    batched = batch_synthesize(synth, items, args, cfg, temperature, chunked, serving) if args.batch else None

    rows = []
    for i, item in enumerate(items):
        spk = args.speaker[i % len(args.speaker)]
        wav, info = batched[i] if batched else generate(item["text"], spk, i)
        ref = _plain(item["text"], language)
        row = {"id": item.get("id", i), "speaker": spk, "ref": ref, "rtf": info["rtf_total"]}
        if judges.asr is not None:
            row["hyp"] = _plain(judges.asr(band_match(wav, args.band)), language)
            row.update(error_counts(ref, row["hyp"]))
        if judges.mos is not None:
            row["mos"] = judges.mos(band_match(wav, 0))
        rows.append(row)
        if i < args.save_wavs:
            sf.write(out / "wav" / f"{row['id']}.wav", wav.numpy(), SAMPLE_RATE)
    with open(out / "utterances.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = {"system": "drifting_tts", "temperature": temperature, "cfg": cfg, **summarize(rows),
           "rtf": float(np.mean([r["rtf"] for r in rows]))}
    results = {"texts": texts, "language": language, "sentences": len(rows), "band_hz": args.band,
               "voices": args.speaker,
               "stream": args.stream or args.fast or args.chunked or args.batch > 0, "fast": args.fast,
               "batch": args.batch, "serving": None if serving is None else vars(serving),
               "chunked": None if chunked is None else vars(chunked),
               "model": args.model, "vocoder": args.vocoder or "stock", "asr": args.asr, "rows": [res],
               "prosody": args.prosody, "prosody_temperature": synth.prosody_temperature if args.prosody else None,
               "prosody_spread": args.prosody_spread, "prosody_durations": args.prosody_durations, "pause": args.pause,
               "paragraph": synth.paragraph, "join": args.join, "context": args.context,
               "acoustic_context": synth.acoustic_context,
               "prosody_duration_temperature": synth.prosody_duration_temperature if args.prosody else None,
               "prosody_pitch_model": args.prosody_pitch_model}
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / "results.md").write_text(format_table([res]) + "\n")
    print(format_table([res]))
    print(f"-> {out / 'results.json'}")
