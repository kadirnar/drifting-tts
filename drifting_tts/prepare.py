"""Prepare an HF audio/text parquet dataset (columns ``audio`` and ``text``) into packed log-mels.

Output layout (``--out``)::

    mels.bin       float16 [total_frames, 100] log-mels, utterances concatenated
    f0.bin         float16 [total_frames] F0 in Hz at the mel frame rate (0 = unvoiced), with --f0
    audio.bin      float16 24 kHz waveforms the mels were computed from, with --save-audio
    index.jsonl    one line per utterance: offset, frames, text, norm_text, speaker, spk_id, split, ...
                   (split: train / val (reported results) / dev (tuning, with --dev-size))
    speakers.json  speaker name -> integer id
    stats.json     global log-mel mean / std (scalar and per bin)
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
from .text import normalize

META_COLUMNS = ["text", "speaker", "quality_score"]  # optional except text; a missing speaker column is one speaker
_MEL = None


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
    p.add_argument("--speaker-name", default=None,
                   help="one speaker name for every row (single-voice data whose speaker column means something else)")
    p.add_argument("--val-max-seconds", type=float, default=12.0, help="longest utterance eligible for val / dev")


def _init_worker(backend: str = "vocos") -> None:
    global _MEL
    torch.set_num_threads(1)
    _MEL = make_logmel(backend)


def _process(row: dict) -> dict | None:
    norm_text = normalize(row["text"] or "")
    if len(norm_text) < 2:
        return None
    try:
        wav = prepare_waveform(row["audio"]["bytes"], trim=row["trim"])
    except Exception as e:  # corrupted audio
        return {"error": f"{type(e).__name__}: {e}"}
    mel = _MEL(wav).T.contiguous().numpy()  # [T, n_mels]
    if not np.isfinite(mel).all():
        return {"error": "non-finite mel"}
    meta = {k: row.get(k) for k in META_COLUMNS}
    meta["speaker"] = meta["speaker"] or "speaker"  # no speaker column: one speaker
    meta.update(norm_text=norm_text, frames=int(mel.shape[0]), audio_seconds=round(len(wav) / 24_000, 3))
    res = {"meta": meta, "mel": mel.astype(np.float16), "sum": mel.sum(0, dtype=np.float64),
           "sumsq": (mel.astype(np.float64) ** 2).sum(0)}
    if row.get("save_audio"):
        res["audio"] = wav.numpy().clip(-1, 1).astype(np.float16)
    if row.get("f0"):
        f0 = extract_f0(wav.numpy(), mel.shape[0], method=row["f0_method"], backend=row["backend"])
        lf0 = np.log(f0[f0 > 0])
        res.update(f0=f0.astype(np.float16), lf0_stats=(float(lf0.sum()), float((lf0**2).sum()), int(lf0.size)))
    return res


def _iter_rows(files: list[str], trim: bool, limit: int, f0: bool = False, save_audio: bool = False,
               f0_method: str = "dio", backend: str = "vocos", speaker_name: str | None = None):
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        present = [c for c in META_COLUMNS if c in pf.schema_arrow.names]  # other datasets lack some metadata
        for rg in range(pf.num_row_groups):
            cols = pf.read_row_group(rg, columns=["audio", *present]).to_pylist()
            for row in cols:
                if speaker_name is not None:
                    row["speaker"] = speaker_name
                row["trim"] = trim
                row["f0"] = f0
                row["f0_method"], row["backend"] = f0_method, backend
                row["save_audio"] = save_audio
                yield row
                n += 1
                if limit and n >= limit:
                    return


def _resolve_files(args) -> list[str]:
    if args.parquet_glob:
        files = sorted(glob.glob(args.parquet_glob))
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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = _resolve_files(args)
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
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
    audio_offset = 0
    pool_args = dict(initializer=_init_worker, initargs=(args.backend,))
    with open(out / "mels.bin", "wb") as fbin, Pool(args.workers, **pool_args) as pool:
        rows = _iter_rows(files, trim=not args.no_trim, limit=args.limit, f0=args.f0, save_audio=args.save_audio,
                          f0_method=args.f0_method, backend=args.backend, speaker_name=args.speaker_name)
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
            res["meta"]["offset"] = offset
            offset += res["meta"]["frames"]
            index.append(res["meta"])
            s += res["sum"]
            ss += res["sumsq"]

    speakers = {name: i for i, name in enumerate(sorted({e["speaker"] for e in index}))}
    val = choose_validation(index, args.val_size, args.seed, max_seconds=args.val_max_seconds)
    dev = choose_validation(index, args.dev_size, args.seed + 1, exclude=val, max_seconds=args.val_max_seconds)
    for i, e in enumerate(index):
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
    stats = {"backend": args.backend, "mean": mean, "std": std,
             "mean_bin": mean_bin.tolist(), "std_bin": std_bin.tolist(),
             "frames": int(offset), "utterances": len(index), "speakers": len(speakers),
             "errors": errors, "skipped": skipped}
    if audio_file is not None:
        audio_file.close()
    if f0_file is not None:
        f0_file.close()
        lf0_mean = lf0_sum / max(lf0_n, 1)
        lf0_std = float(np.sqrt(max(lf0_sq / max(lf0_n, 1) - lf0_mean**2, 1e-8)))
        stats.update(f0_method=args.f0_method, lf0_mean=lf0_mean, lf0_std=lf0_std, voiced_frames=lf0_n)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {len(index)} utterances ({offset / 93.75 / 3600:.1f} h), {len(speakers)} speakers, "
          f"{errors} decode errors, {skipped} empty transcripts -> {out}")
