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

from .audio import FRAME_RATE, SAMPLE_RATE


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="a prepared root with audio.bin (`prepare --save-audio`)")
    p.add_argument("--backend", required=True, help="dacvae, voxcpm2 or voxcpm1.5")
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=4, help="copies of every latent frame (alignment resolution)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def resample_f0(f0: np.ndarray, src_rate: float, frames: int, rate: float) -> np.ndarray:
    """F0 per source frame (frame ``i`` spans ``[i, i + 1) / src_rate``) -> per new frame, by the source frame that
    contains each new frame's centre."""
    centres = (np.arange(frames) + 0.5) / rate
    idx = np.clip(np.floor(centres * src_rate).astype(np.int64), 0, max(len(f0) - 1, 0))
    return f0[idx] if len(f0) else np.zeros(frames, f0.dtype)


@torch.inference_mode()
def run(args) -> None:
    from .latents import load_backend

    src, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st_src = json.loads((src / "stats.json").read_text())
    audio = np.memmap(src / "audio.bin", dtype=np.float16, mode="r")
    f0_src = np.memmap(src / "f0.bin", dtype=np.float16, mode="r") if (src / "f0.bin").exists() else None
    src_rate = float(st_src.get("frame_rate", FRAME_RATE))
    index = [json.loads(line) for line in open(src / "index.jsonl")]
    be = load_backend(args.backend, args.device)
    rate = be.frame_rate * args.repeat

    total, done = 0, {}  # (audio offset, samples) -> (offset, frames): merged roots repeat entries
    s = ss = None
    n_native = 0
    with open(out / "mels.bin", "wb") as fm, open(out / "f0.bin", "wb") as ff:
        for e in tqdm(index, desc=f"extract {args.backend}"):
            key = (e["audio_offset"], e["audio_samples"])
            if key not in done:
                wav = torch.from_numpy(np.array(audio[key[0]: key[0] + key[1]], dtype=np.float32))
                z = be.encode(wav.to(args.device), SAMPLE_RATE)[0].float()  # [dim, T]
                zs = z.double()
                s = zs.sum(1) if s is None else s + zs.sum(1)
                ss = (zs * zs).sum(1) if ss is None else ss + (zs * zs).sum(1)
                n_native += z.shape[1]
                frames = z.shape[1] * args.repeat
                fm.write(z.repeat_interleave(args.repeat, dim=1).T.contiguous().cpu().numpy()
                         .astype(np.float16).tobytes())
                if f0_src is not None:
                    f0 = np.array(f0_src[e["offset"]: e["offset"] + e["frames"]], dtype=np.float32)
                    ff.write(resample_f0(f0, src_rate, frames, rate).astype(np.float16).tobytes())
                done[key] = (total, frames)
                total += frames
            e["offset"], e["frames"] = done[key]
    with open(out / "index.jsonl", "w") as f:
        for e in index:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    mean = (s / n_native).cpu().numpy()
    std = np.sqrt(np.maximum((ss / n_native).cpu().numpy() - mean**2, 0.0))
    stats = {"backend": args.backend, "dim": int(be.dim), "frame_rate": rate, "latent_repeat": args.repeat,
             "latent_rate": be.frame_rate, "output_rate": be.output_rate, "mean": mean.tolist(), "std": std.tolist(),
             "frames": int(total), "utterances": len(index)}
    stats.update({k: st_src[k] for k in ("f0_method", "lf0_mean", "lf0_std") if k in st_src})
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
