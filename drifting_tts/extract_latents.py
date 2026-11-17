"""Extract audio-VAE latents for a prepared dataset: a new data root whose frames are VAE latents (issue #17).

The source is a ``prepare --save-audio --f0`` root. Its waveforms are encoded with an audio backend
(:mod:`drifting_tts.latents`, posterior means), and every latent frame is repeated ``--repeat`` times (default 4:
25 Hz -> 100 frames per second). Characters with interspersed blanks come at ~29 tokens per second, more than the
25 Hz latent rate, so monotonic alignment needs the repeats. A generator with patch size 4 then works at the latent
rate, and :class:`drifting_tts.latents.vocoder.LatentVocoder` averages the repeats back before decoding.

Output layout (``--out``), as ``prepare`` (the training code reads either)::

    mels.bin      float16 [total_frames, dim] latents (utterances concatenated)
    f0.bin        float16 [total_frames] the source F0, resampled to the latent frame rate
    index.jsonl   the source index with this root's offsets and frame counts (line order kept)
    stats.json    per-channel mean / std, dim, frame_rate, latent_repeat, backend, and the source log-F0 stats
    audio.bin, speakers.json, scores.jsonl   linked / copied from the source

    drifting-tts extract-latents --data data/train --backend voxcpm2 --out data/train_voxcpm2

**The encoder's input rate.** The prepared waveforms are 24 kHz, so an encoder that takes 44.1 / 48 kHz audio
(``dacvae``, ``voxcpm1.5``) would get it band-limited to 12 kHz. A root made with ``prepare --save-audio
--hires-rate 48000`` also holds the waveforms at that rate (``audio_hr.bin``), and those are what such an encoder
gets; ``--audio-rate 24000`` keeps the 24 kHz ones (the roots made before #94). ``stats.json`` records the rate
used (``audio_rate``).

**From parquet files** (``--parquet-glob``, local or ``hf://``) the corpus is encoded without a prepared root or a
copy of its audio: the shards download in parallel (``--prefetch``), worker processes turn each row into the waveform
``prepare`` would (decode, trim, loudness; at the encoder's input rate if that is above 24 kHz, else 24 kHz, or at
``--audio-rate``) and its text into the language's normalised form, and the encoder
runs on batches of similar length on the GPU. ``--val-glob`` names held-out files whose rows form the val split (the
rest of them: dev), as in ``prepare``; no F0 and no audio are stored. For 10,000 h with Mimi (12.5 Hz x 32, stored
at 50 Hz with ``--repeat 4``) that is ~115 GB::

    drifting-tts extract-latents --parquet-glob "hf://datasets/VoiceHub/echo-clones-10kh-en/data/*.parquet" \
        --val-glob "hf://datasets/SynDataLab-EN/echo-clones-4m-en/data/train-07960.parquet" --backend mimi \
        --language en --speaker-column speaker_id --out data/en_mimi
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .audio import FRAME_RATE, SAMPLE_RATE, resample


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=None, help="a prepared root with audio.bin (`prepare --save-audio`)")
    p.add_argument("--parquet-glob", default=None, help="or: parquet files (local or hf://) encoded directly")
    p.add_argument("--backend", required=True, help="dacvae, voxcpm2, voxcpm1.5 or mimi")
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=4, help="copies of every latent frame (alignment resolution)")
    p.add_argument("--native", action="store_true",
                   help="store every latent frame once (a quarter of the disk with --repeat 4); the data loader "
                        "makes the copies. For a model that ties them (model.ar.tied, #92)")
    p.add_argument("--save-std", action="store_true",
                   help="also stds.bin: the posterior's standard deviation, for training on posterior samples "
                        "(drift.target_noise: posterior, #92); a backend with a posterior")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--audio-rate", type=int, default=None,
                   help="rate of the audio given to the encoder (default: its input rate if that is above 24 kHz and "
                        "the audio is there at it, see the module docstring; 24000: as before #94)")
    # --parquet-glob only
    p.add_argument("--val-glob", default=None,
                   help="held-out parquet files: their rows are val (--val-size), the rest dev")
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--val-max-seconds", type=float, default=12.0)
    p.add_argument("--language", default="tr", help="text frontend (normalisation)")
    p.add_argument("--speaker-column", default="speaker")
    p.add_argument("--max-seconds", type=float, default=30.0, help="longer utterances are skipped")
    p.add_argument("--workers", type=int, default=12, help="processes preparing the waveforms")
    p.add_argument("--prefetch", type=int, default=8, help="Hub shards downloaded at once")
    p.add_argument("--batch-seconds", type=float, default=640.0, help="padded audio per encoder batch")
    p.add_argument("--max-pending", type=int, default=2048,
                   help="waveforms prepared ahead of the encoder (bounds the memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true",
                   help="TF32 matmuls / convolutions (Mimi latents differ from fp32 by ~2e-4, below fp16 storage)")


def source_rate(be, requested: int | None, hires: int | None = None) -> int:
    """The rate of the waveforms the encoder ``be`` is given: ``requested`` (``--audio-rate``), else the best there
    is for it. ``hires``: the rate of a prepared root's ``audio_hr.bin`` (``None``: 24 kHz audio only); without a
    root (parquet files) any rate can be made from the source."""
    if requested:
        return requested
    if be.input_rate <= SAMPLE_RATE:
        return SAMPLE_RATE
    return be.input_rate if hires is None else hires


def encode_batch(be, wavs: list[np.ndarray], sr: int, with_std: bool = False) -> list:
    """The latents ``[dim, T_k]`` of waveforms of different lengths, encoded as one right-padded batch; with
    ``with_std`` pairs of the posterior's mean and standard deviation.

    Zeros after an utterance do not reach the frames of a causal encoder that zero-pads to whole frames (VoxCPM,
    Mimi), so its rows are what it returns for each utterance alone. An encoder that reflect-pads to whole frames
    (``be.frame_pad == "reflect"``: DAC-VAE) gets that padding per row here, as it would make it for the utterance
    alone; what comes after it is zeros, where alone its convolutions pad the features, so the last frames within its
    receptive field can still differ from single-utterance encoding (exactly equal for rows of equal frame count)."""
    frames = [be.num_frames(len(w), sr) for w in wavs]
    x = torch.zeros(len(wavs), max(len(w) for w in wavs))
    for k, w in enumerate(wavs):
        x[k, : len(w)] = torch.from_numpy(w)
    x = x.to(be.device)
    if getattr(be, "frame_pad", "constant") == "reflect":
        x, hop = resample(x, sr, be.input_rate), be.hop_in
        x = torch.nn.functional.pad(x, (0, max(0, max(frames) * hop - x.shape[1])))
        for k, w in enumerate(wavs):
            n, end = -(-len(w) * be.input_rate // sr), frames[k] * hop  # the utterance's samples; its whole frames
            if 0 < end - n < n:
                x[k, n:end] = x[k, 2 * n - end - 1: n - 1].flip(0)  # F.pad(..., "reflect")
            x[k, end:] = 0  # incl. the resampler's ringing past the utterance
        sr = be.input_rate
    if with_std:
        z, std = (t.float() for t in be.posterior(x, sr))
        return [(z[k, :, :t], std[k, :, :t]) for k, t in enumerate(frames)]
    z = be.encode(x, sr).float()
    return [z[k, :, :t] for k, t in enumerate(frames)]


def frame_bytes(z: torch.Tensor, copies: int) -> bytes:
    """Frames ``[dim, T]`` as they are stored: frame-major float16, every frame ``copies`` times."""
    z = z.repeat_interleave(copies, dim=1) if copies > 1 else z
    return z.T.contiguous().cpu().numpy().astype(np.float16).tobytes()


def check_posterior(be, args) -> bool:
    """Whether the backend has a posterior (its statistics then go to ``stats.json``; ``--save-std`` needs one)."""
    if args.save_std and not hasattr(be, "posterior"):
        raise SystemExit(f"--save-std: the {args.backend} backend has no posterior (its encoder gives one latent)")
    return hasattr(be, "posterior")


def resample_f0(f0: np.ndarray, src_rate: float, frames: int, rate: float) -> np.ndarray:
    """F0 per source frame (frame ``i`` spans ``[i, i + 1) / src_rate``) -> per new frame, by the source frame that
    contains each new frame's centre."""
    centres = (np.arange(frames) + 0.5) / rate
    idx = np.clip(np.floor(centres * src_rate).astype(np.int64), 0, max(len(f0) - 1, 0))
    return f0[idx] if len(f0) else np.zeros(frames, f0.dtype)


def storage_stats(args, sd, n_native: int, std: np.ndarray) -> dict:
    """``stats.json`` entries of ``--native`` and of a codec with a posterior: how the frames are stored, and
    ``active``: the channels whose spread over the data exceeds the posterior's own standard deviation. The other
    channels are unused by the VAE (their posterior is as wide as the prior) and are left out of the normalised
    frames (:class:`drifting_tts.data.MelStats`). ``posterior_rel``: the posterior's standard deviation over the
    data's, median of the active channels: what a posterior sample adds, in normalised units."""
    out = {"stored_repeat": 1 if args.native else args.repeat}
    if sd is not None:
        post = (sd / n_native).cpu().numpy()
        active = std > post
        out.update(posterior_std=bool(args.save_std), active=[bool(a) for a in active],
                   posterior_rel=float(np.median((post / np.maximum(std, 1e-3))[active])) if active.any() else None)
    return out


@torch.inference_mode()
def run(args) -> None:
    if args.parquet_glob:
        return run_parquet(args)
    if not args.data:
        raise SystemExit("give --data <prepared root> or --parquet-glob")
    from .latents import load_backend

    src, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st_src = json.loads((src / "stats.json").read_text())
    hires = st_src.get("hr_rate") if (src / "audio_hr.bin").exists() else None
    f0_src = np.memmap(src / "f0.bin", dtype=np.float16, mode="r") if (src / "f0.bin").exists() else None
    src_rate = float(st_src.get("frame_rate", FRAME_RATE))
    index = [json.loads(line) for line in open(src / "index.jsonl")]
    be = load_backend(args.backend, args.device)
    rate = be.frame_rate * args.repeat
    sr = source_rate(be, getattr(args, "audio_rate", None), hires or SAMPLE_RATE)
    if sr not in (SAMPLE_RATE, hires):
        raise SystemExit(f"--audio-rate {sr}: {src} holds audio at {SAMPLE_RATE} Hz"
                         + (f" and {hires} Hz" if hires else " only (prepare --save-audio --hires-rate)"))
    name, k_off, k_len = ("audio.bin", "audio_offset", "audio_samples") if sr == SAMPLE_RATE else \
        ("audio_hr.bin", "hr_offset", "hr_samples")
    audio = np.memmap(src / name, dtype=np.float16, mode="r")
    if be.input_rate > sr:
        print(f"the {args.backend} encoder takes {be.input_rate} Hz audio and gets {sr} Hz audio (nothing above "
              f"{sr // 2000} kHz): `prepare --save-audio --hires-rate {be.input_rate}` stores it", flush=True)

    post = check_posterior(be, args)
    copies = 1 if args.native else args.repeat  # stored frames per latent frame
    total, done = 0, {}  # (audio offset, samples) -> (offset, frames): merged roots repeat entries
    s = ss = sd = None
    n_native = 0
    with open(out / "mels.bin", "wb") as fm, open(out / "f0.bin", "wb") as ff, \
            open(out / "stds.bin" if args.save_std else os.devnull, "wb") as fs:
        for e in tqdm(index, desc=f"extract {args.backend}"):
            key = (e[k_off], e[k_len])
            if key not in done:
                wav = torch.from_numpy(np.array(audio[key[0]: key[0] + key[1]], dtype=np.float32))
                if post:
                    z, std = (t[0].float() for t in be.posterior(wav.to(args.device), sr))
                    if args.save_std:
                        fs.write(frame_bytes(std, copies))
                    sd = std.double().sum(1) if sd is None else sd + std.double().sum(1)
                else:
                    z = be.encode(wav.to(args.device), sr)[0].float()  # [dim, T]
                zs = z.double()
                s = zs.sum(1) if s is None else s + zs.sum(1)
                ss = (zs * zs).sum(1) if ss is None else ss + (zs * zs).sum(1)
                n_native += z.shape[1]
                frames = z.shape[1] * args.repeat
                fm.write(frame_bytes(z, copies))
                if f0_src is not None and not args.native:  # per stored frame: not with one frame per latent
                    f0 = np.array(f0_src[e["offset"]: e["offset"] + e["frames"]], dtype=np.float32)
                    ff.write(resample_f0(f0, src_rate, frames, rate).astype(np.float16).tobytes())
                done[key] = (total, frames)
                total += z.shape[1] * copies
            e["offset"], e["frames"] = done[key]
    with open(out / "index.jsonl", "w") as f:
        for e in index:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    mean = (s / n_native).cpu().numpy()
    std = np.sqrt(np.maximum((ss / n_native).cpu().numpy() - mean**2, 0.0))
    stats = {"backend": args.backend, "dim": int(be.dim), "frame_rate": rate, "latent_repeat": args.repeat,
             "latent_rate": be.frame_rate, "output_rate": be.output_rate, "audio_rate": sr, "mean": mean.tolist(),
             "std": std.tolist(), "frames": int(total), "utterances": len(index),
             **storage_stats(args, sd, n_native, std)}
    stats.update({k: st_src[k] for k in ("f0_method", "lf0_mean", "lf0_std", "language") if k in st_src})
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    for name in ("speakers.json", "scores.jsonl"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    link = out / "audio.bin"
    if not link.exists():
        os.symlink((src / "audio.bin").resolve(), link)
    print(f"{len(done)} utterances encoded ({n_native / be.frame_rate / 3600:.1f} h of latents at "
          f"{be.frame_rate:g} Hz, stored at {rate:g} Hz), {len(index)} index entries; channel std "
          f"{std.min():.3f}-{std.max():.3f}{' (collapsed channels present)' if std.min() < 1e-3 else ''} -> {out}")
    if not math.isfinite(float(std.sum())):
        raise RuntimeError("non-finite latent statistics")


_NORMALIZE, _RATE = None, SAMPLE_RATE


def _init_worker(language: str, rate: int = SAMPLE_RATE) -> None:
    global _NORMALIZE, _RATE
    from .text import frontend

    torch.set_num_threads(1)
    _NORMALIZE, _RATE = frontend(language).normalize, rate


def _prepare_row(row: dict) -> tuple[dict, np.ndarray] | None:
    """A parquet row -> (its index entry, the waveform ``prepare`` makes, at the worker's rate), or None (no text,
    bad audio)."""
    from .audio import prepare_waveform

    norm = _NORMALIZE(row["text"] or "")
    if len(norm) < 2:
        return None
    try:
        wav = prepare_waveform(row["audio"]["bytes"], sample_rate=_RATE).numpy()
    except Exception:  # corrupted audio
        return None
    meta = {"text": row["text"], "norm_text": norm, "speaker": str(row.get("speaker") or "speaker"),
            "audio_seconds": round(len(wav) / _RATE, 3)}
    if row.get("held_out"):
        meta["held_out"] = True
    return meta, wav


def _shard_rows(files: list[str], held_out: frozenset[str], speaker_column: str, prefetch: int, cache: Path):
    """The rows of ``files`` in order; Hub files are downloaded ``prefetch`` at a time and deleted after use."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq

    def fetch(f: str) -> tuple[str, bool]:
        if not f.startswith("hf://"):
            return f, False
        from huggingface_hub import hf_hub_download

        from .stream import _hub_path

        repo, path = _hub_path(f)
        return hf_hub_download(repo, path, repo_type="dataset", local_dir=cache), True

    with ThreadPoolExecutor(prefetch) as pool:
        ahead = [pool.submit(fetch, f) for f in files[:prefetch]]
        for i, f in enumerate(files):
            path, temporary = ahead[i].result()
            if i + prefetch < len(files):
                ahead.append(pool.submit(fetch, files[i + prefetch]))
            pf = pq.ParquetFile(path)
            cols = [c for c in ("audio", "text", speaker_column) if c in pf.schema_arrow.names]
            for rg in range(pf.num_row_groups):
                for row in pf.read_row_group(rg, columns=cols).to_pylist():
                    if speaker_column != "speaker":
                        row["speaker"] = row.pop(speaker_column, None)
                    row["held_out"] = f in held_out
                    yield row
            if temporary:
                os.remove(path)


@torch.inference_mode()
def run_parquet(args) -> None:
    import random
    import threading
    from multiprocessing import Pool

    from .latents import load_backend
    from .prepare import _glob

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    files = _glob(args.parquet_glob)
    held_out = frozenset(_glob(args.val_glob)) if args.val_glob else frozenset()
    files = [f for f in files if f not in held_out] + sorted(held_out)
    if not files:
        raise FileNotFoundError(f"no parquet files match {args.parquet_glob}")
    be = load_backend(args.backend, args.device)
    rate = be.frame_rate * args.repeat
    sr = source_rate(be, getattr(args, "audio_rate", None))
    post = check_posterior(be, args)
    copies = 1 if args.native else args.repeat  # stored frames per latent frame
    index: list[dict] = []
    s = ss = sd = None
    n_native = total = skipped = 0
    buffer: list[tuple[dict, np.ndarray]] = []

    def encode(batch: list[tuple[dict, np.ndarray]]) -> None:
        nonlocal s, ss, sd, n_native, total
        for (meta, _), zk in zip(batch, encode_batch(be, [w for _, w in batch], sr, post)):
            if post:
                zk, std = zk
                if args.save_std:
                    fs.write(frame_bytes(std, copies))
                sd = std.double().sum(1) if sd is None else sd + std.double().sum(1)
            zs = zk.double()
            s = zs.sum(1) if s is None else s + zs.sum(1)
            ss = (zs * zs).sum(1) if ss is None else ss + (zs * zs).sum(1)
            n_native += zk.shape[1]
            fm.write(frame_bytes(zk, copies))
            index.append({**meta, "offset": total, "frames": zk.shape[1] * args.repeat})
            total += zk.shape[1] * copies

    def flush(final: bool = False) -> None:
        """Encode the buffer in batches of similar length (each within --batch-seconds of padded audio)."""
        nonlocal buffer
        if not buffer or (not final and sum(len(w) for _, w in buffer) < 4 * args.batch_seconds * sr):
            return
        buffer.sort(key=lambda item: len(item[1]))
        batch: list = []
        for item in buffer:
            if batch and (len(batch) + 1) * len(item[1]) > args.batch_seconds * sr:
                encode(batch)
                batch = []
            batch.append(item)
        encode(batch)
        buffer = []

    # Pool.imap keeps every finished result until it is consumed: workers that outrun the encoder would fill the
    # memory (10k h: the 60 GB limit after 6 min), so at most --max-pending rows are handed out ahead of it.
    pending = threading.Semaphore(args.max_pending)

    def throttled(rows):
        for row in rows:
            pending.acquire()
            yield row

    rows = _shard_rows(files, held_out, args.speaker_column, args.prefetch, out / "_shards")
    with open(out / "mels.bin", "wb") as fm, open(out / "stds.bin" if args.save_std else os.devnull, "wb") as fs, \
            Pool(args.workers, initializer=_init_worker, initargs=(args.language, sr)) as pool:
        for res in tqdm(pool.imap(_prepare_row, throttled(rows), chunksize=8), desc=f"extract {args.backend}"):
            pending.release()
            if res is None or len(res[1]) > args.max_seconds * sr:
                skipped += 1
                continue
            buffer.append(res)
            flush()
        flush(final=True)

    speakers = {name: i for i, name in enumerate(sorted({e["speaker"] for e in index}))}
    rows_out = [i for i, e in enumerate(index) if e.get("held_out")]
    pool_val = [i for i in rows_out if 2.0 <= index[i]["audio_seconds"] <= args.val_max_seconds]
    val = set(random.Random(args.seed).sample(pool_val, min(args.val_size, len(pool_val))))
    with open(out / "index.jsonl", "w") as f:
        for i, e in enumerate(index):
            e.pop("held_out", None)
            e["spk_id"] = speakers[e["speaker"]]
            e["split"] = "val" if i in val else "dev" if i in set(rows_out) else "train"
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=0))
    mean = (s / n_native).cpu().numpy()
    std = np.sqrt(np.maximum((ss / n_native).cpu().numpy() - mean**2, 0.0))
    stats = {"backend": args.backend, "language": args.language, "dim": int(be.dim), "frame_rate": rate,
             "latent_repeat": args.repeat, "latent_rate": be.frame_rate, "output_rate": be.output_rate,
             "audio_rate": sr, "mean": mean.tolist(), "std": std.tolist(), "frames": int(total),
             "utterances": len(index), "speakers": len(speakers), "skipped": skipped,
             **storage_stats(args, sd, n_native, std)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    shutil.rmtree(out / "_shards", ignore_errors=True)
    print(f"{len(index)} utterances ({n_native / be.frame_rate / 3600:.1f} h of latents at {be.frame_rate:g} Hz, "
          f"stored at {rate:g} Hz), {len(speakers)} speakers, {skipped} skipped, val {len(val)} / dev "
          f"{len(rows_out) - len(val)} -> {out}")
    if not math.isfinite(float(std.sum())):
        raise RuntimeError("non-finite latent statistics")
