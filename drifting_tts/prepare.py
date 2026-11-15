"""Prepare an HF audio/text parquet dataset (columns ``audio`` and ``text``) into packed log-mels.

Output layout (``--out``)::

    mels.bin       float16 [total_frames, 100] log-mels, utterances concatenated
    f0.bin         float16 [total_frames] F0 in Hz at the mel frame rate (0 = unvoiced), with --f0
    audio.bin      float16 24 kHz waveforms the mels were computed from, with --save-audio
    audio_hr.bin   float16 waveforms of the same utterances at --hires-rate (48 kHz: what the 44.1 / 48 kHz codecs
                   of `extract-latents` encode; index: hr_offset / hr_samples, stats.json: hr_rate)
    index.jsonl    one line per utterance: offset, frames, text, norm_text, speaker, spk_id, split, ...
                   (split: train / val (reported results) / dev (tuning, with --dev-size))
    speakers.json  speaker name -> integer id
    stats.json     global log-mel mean / std (scalar and per bin), the backend and the text ``language``

``--parquet-glob`` also takes Hub paths (``hf://datasets/<repo>/data/train-0000*.parquet``), read without a local
copy; ``--val-glob`` names held-out files whose rows form the val split (the rest of them: dev), e.g. shards of a large
multi-speaker corpus past the ones used for training.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from .audio import BACKENDS, N_MELS, extract_f0, make_logmel, prepare_waveform
from .text import LANGUAGES, frontend

META_COLUMNS = ["text", "speaker", "quality_score"]  # optional except text; a missing speaker column is one speaker
_MEL, _NORMALIZE = None, None


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default=None, help="HF dataset repo id (or use --parquet-glob)")
    p.add_argument("--parquet-glob", default=None, help="use local parquet files instead of the HF hub")
    p.add_argument("--out", default="data/train")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-size", type=int, default=200, help="held-out utterances for reported results")
    p.add_argument("--dev-size", type=int, default=0,
                   help="a disjoint held-out 'dev' split for tuning (temperature, guidance, ...); val is unchanged")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="process only the first N rows (debugging)")
    p.add_argument("--no-trim", action="store_true", help="do not trim leading/trailing silence")
    p.add_argument("--backend", choices=BACKENDS, default="vocos", help="mel front end / vocoder pair")
    p.add_argument("--f0", action="store_true", help="also extract WORLD F0 (pitch conditioning)")
    p.add_argument("--f0-method", choices=("dio", "harvest"), default="dio", help="WORLD F0 estimator")
    p.add_argument("--save-audio", action="store_true", help="also store the 24 kHz waveforms (vocoder fine-tuning)")
    p.add_argument("--hires-rate", type=int, default=0,
                   help="with --save-audio: also store the waveforms at this rate, resampled from the source "
                        "(audio_hr.bin; 48000 for `extract-latents` with dacvae / voxcpm1.5, which otherwise get "
                        "audio band-limited to 12 kHz)")
    p.add_argument("--speaker-name", default=None,
                   help="one speaker name for every row (single-voice data whose speaker column means something else)")
    p.add_argument("--val-max-seconds", type=float, default=12.0, help="longest utterance eligible for val / dev")
    p.add_argument("--language", choices=LANGUAGES, default="tr", help="text frontend (normalisation, symbols)")
    p.add_argument("--speaker-column", default="speaker", help="the column naming each row's speaker")
    p.add_argument("--val-glob", default=None,
                   help="held-out parquet files: their rows are the val split (up to --val-size), the rest dev")


def _init_worker(backend: str = "vocos", language: str = "tr") -> None:
    global _MEL, _NORMALIZE
    torch.set_num_threads(1)
    _MEL = make_logmel(backend)
    _NORMALIZE = frontend(language).normalize


def _process(row: dict) -> dict | None:
    norm_text = _NORMALIZE(row["text"] or "")
    if len(norm_text) < 2:
        return None
    try:
        wav = prepare_waveform(row["audio"]["bytes"], trim=row["trim"], also=row.get("hires_rate") or None)
    except Exception as e:  # corrupted audio
        return {"error": f"{type(e).__name__}: {e}"}
    wav, hires = wav if isinstance(wav, tuple) else (wav, None)
    mel = _MEL(wav).T.contiguous().numpy()  # [T, n_mels]
    if not np.isfinite(mel).all():
        return {"error": "non-finite mel"}
    meta = {k: row.get(k) for k in META_COLUMNS}
    meta["speaker"] = meta["speaker"] or "speaker"  # no speaker column: one speaker
    if row.get("held_out"):
        meta["held_out"] = True
    meta.update(norm_text=norm_text, frames=int(mel.shape[0]), audio_seconds=round(len(wav) / 24_000, 3))
    res = {"meta": meta, "mel": mel.astype(np.float16), "sum": mel.sum(0, dtype=np.float64),
           "sumsq": (mel.astype(np.float64) ** 2).sum(0)}
    if row.get("save_audio"):
        res["audio"] = wav.numpy().clip(-1, 1).astype(np.float16)
    if hires is not None:
        res["audio_hr"] = hires.numpy().clip(-1, 1).astype(np.float16)
    if row.get("f0"):
        f0 = extract_f0(wav.numpy(), mel.shape[0], method=row["f0_method"], backend=row["backend"])
        lf0 = np.log(f0[f0 > 0])
        res.update(f0=f0.astype(np.float16), lf0_stats=(float(lf0.sum()), float((lf0**2).sum()), int(lf0.size)))
    return res


def _open(f: str):
    """A local path, or a Hub file (``hf://...``) opened for reading without a local copy."""
    if f.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        return HfFileSystem().open(f, "rb", block_size=64 * 2**20)
    return f


def _iter_rows(files: list[str], trim: bool, limit: int, f0: bool = False, save_audio: bool = False,
               f0_method: str = "dio", backend: str = "vocos", speaker_name: str | None = None,
               speaker_column: str = "speaker", held_out: frozenset[str] = frozenset(), hires_rate: int = 0):
    n = 0
    columns = [speaker_column if c == "speaker" else c for c in META_COLUMNS]
    for f in files:
        pf = pq.ParquetFile(_open(f))
        present = [c for c in columns if c in pf.schema_arrow.names]  # other datasets lack some metadata
        for rg in range(pf.num_row_groups):
            cols = pf.read_row_group(rg, columns=["audio", *present]).to_pylist()
            for row in cols:
                if speaker_column != "speaker" and speaker_column in row:
                    row["speaker"] = row.pop(speaker_column)
                if speaker_name is not None:
                    row["speaker"] = speaker_name
                row["held_out"] = f in held_out
                row["trim"] = trim
                row["f0"] = f0
                row["f0_method"], row["backend"] = f0_method, backend
                row["save_audio"], row["hires_rate"] = save_audio, hires_rate
                yield row
                n += 1
                if limit and n >= limit:
                    return


def _glob(pattern: str) -> list[str]:
    if pattern.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        return sorted(f"hf://{f}" for f in HfFileSystem().glob(pattern))
    return sorted(glob.glob(pattern))


def _resolve_files(args) -> list[str]:
    if args.parquet_glob:
        files = _glob(args.parquet_glob)
    else:
        from huggingface_hub import snapshot_download

        if not args.dataset:
            raise SystemExit("give --dataset <hf dataset id> or --parquet-glob")
        root = snapshot_download(args.dataset, repo_type="dataset", allow_patterns=["*.parquet"])
        files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError("no parquet files found")
    return files


def choose_validation(index: list[dict], size: int, seed: int,
                      exclude: frozenset[int] | set[int] = frozenset(), max_seconds: float = 12.0) -> set[int]:
    """Held-out utterances from *seen* speakers (>= 20 utterances), clean (quality >= 70, or no quality score) and
    of moderate length."""
    counts = Counter(e["speaker"] for e in index)
    quality = lambda e: 100 if e.get("quality_score") is None else e["quality_score"]  # noqa: E731
    pool = [i for i, e in enumerate(index) if i not in exclude and counts[e["speaker"]] >= 20
            and quality(e) >= 70 and 2.0 <= e["audio_seconds"] <= max_seconds]
    rng = random.Random(seed)
    return set(rng.sample(pool, min(size, len(pool))))


def run(args) -> None:
    hires = getattr(args, "hires_rate", 0)
    if hires and not args.save_audio:
        raise SystemExit("--hires-rate stores a second copy of the audio: it needs --save-audio")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = _resolve_files(args)
    held_out = frozenset(_glob(args.val_glob)) if args.val_glob else frozenset()
    if args.val_glob and not held_out:
        raise FileNotFoundError(f"no parquet files match --val-glob {args.val_glob}")
    files = [f for f in files if f not in held_out] + sorted(held_out)
    total = sum(pq.ParquetFile(_open(f)).metadata.num_rows for f in files)
    if args.limit:
        total = min(total, args.limit)

    index: list[dict] = []
    s = np.zeros(N_MELS, np.float64)
    ss = np.zeros(N_MELS, np.float64)
    offset = errors = skipped = 0
    lf0_sum = lf0_sq = 0.0
    lf0_n = 0
    f0_file = open(out / "f0.bin", "wb") if args.f0 else None
    audio_file = open(out / "audio.bin", "wb") if args.save_audio else None
    hr_file = open(out / "audio_hr.bin", "wb") if hires else None
    audio_offset = hr_offset = 0
    pool_args = dict(initializer=_init_worker, initargs=(args.backend, args.language))
    with open(out / "mels.bin", "wb") as fbin, Pool(args.workers, **pool_args) as pool:
        rows = _iter_rows(files, trim=not args.no_trim, limit=args.limit, f0=args.f0, save_audio=args.save_audio,
                          f0_method=args.f0_method, backend=args.backend, speaker_name=args.speaker_name,
                          speaker_column=args.speaker_column, held_out=held_out, hires_rate=hires)
        for res in tqdm(pool.imap(_process, rows, chunksize=4), total=total, desc="prepare"):
            if res is None:
                skipped += 1
                continue
            if "error" in res:
                errors += 1
                continue
            fbin.write(res["mel"].tobytes())
            if f0_file is not None:
                f0_file.write(res["f0"].tobytes())
                a, b, c = res["lf0_stats"]
                lf0_sum, lf0_sq, lf0_n = lf0_sum + a, lf0_sq + b, lf0_n + c
            if audio_file is not None:
                audio_file.write(res["audio"].tobytes())
                res["meta"]["audio_offset"] = audio_offset
                res["meta"]["audio_samples"] = int(res["audio"].size)
                audio_offset += int(res["audio"].size)
            if hr_file is not None:
                hr_file.write(res["audio_hr"].tobytes())
                res["meta"]["hr_offset"] = hr_offset
                res["meta"]["hr_samples"] = int(res["audio_hr"].size)
                hr_offset += int(res["audio_hr"].size)
            res["meta"]["offset"] = offset
            offset += res["meta"]["frames"]
            index.append(res["meta"])
            s += res["sum"]
            ss += res["sumsq"]

    speakers = {name: i for i, name in enumerate(sorted({e["speaker"] for e in index}))}
    if held_out:  # val from the held-out files' rows, the rest of them dev; nothing else is held out
        rows = [i for i, e in enumerate(index) if e.get("held_out")]
        pool = [i for i in rows if 2.0 <= index[i]["audio_seconds"] <= args.val_max_seconds]
        val = set(random.Random(args.seed).sample(pool, min(args.val_size, len(pool))))
        dev = set(rows) - val
    else:
        val = choose_validation(index, args.val_size, args.seed, max_seconds=args.val_max_seconds)
        dev = choose_validation(index, args.dev_size, args.seed + 1, exclude=val, max_seconds=args.val_max_seconds)
    for i, e in enumerate(index):
        e.pop("held_out", None)
        e["spk_id"] = speakers[e["speaker"]]
        e["split"] = "val" if i in val else "dev" if i in dev else "train"
    with open(out / "index.jsonl", "w") as f:
        for e in index:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=0))

    mean_bin = s / offset
    std_bin = np.sqrt(np.maximum(ss / offset - mean_bin**2, 1e-8))
    mean = float(s.sum() / (offset * N_MELS))
    std = float(np.sqrt(ss.sum() / (offset * N_MELS) - mean**2))
    stats = {"backend": args.backend, "language": args.language, "mean": mean, "std": std,
             "mean_bin": mean_bin.tolist(), "std_bin": std_bin.tolist(),
             "frames": int(offset), "utterances": len(index), "speakers": len(speakers),
             "errors": errors, "skipped": skipped}
    if audio_file is not None:
        audio_file.close()
    if hr_file is not None:
        hr_file.close()
        stats["hr_rate"] = hires
    if f0_file is not None:
        f0_file.close()
        lf0_mean = lf0_sum / max(lf0_n, 1)
        lf0_std = float(np.sqrt(max(lf0_sq / max(lf0_n, 1) - lf0_mean**2, 1e-8)))
        stats.update(f0_method=args.f0_method, lf0_mean=lf0_mean, lf0_std=lf0_std, voiced_frames=lf0_n)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {len(index)} utterances ({offset / 93.75 / 3600:.1f} h), {len(speakers)} speakers, "
          f"{errors} decode errors, {skipped} empty transcripts -> {out}")
