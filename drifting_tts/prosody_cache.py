"""Prosody targets of a trained TTS model: MAS durations and token pitch of every utterance (``prosody-cache``).

The frozen text encoder of a trained model (eval mode, fp32) is run over the training corpus (the training split
with the model's ``data.filters``, plus ``val`` / ``dev``). For every utterance it stores, per token:

* ``dur``: the MAS duration in frames (``align(mu, x_mask, y, y_mask)``, as in training; always >= 1);
* ``pitch``: the token pitch the generator was trained on (:func:`token_pitch` with the model's ``lf0_stats``,
  0 for unvoiced tokens) and ``voiced``, the number of voiced frames of the token;
* ``logw_det`` / ``pitch_det``: the deterministic predictions of the model's duration and pitch regressors.

Utterances listed several times in the training split (``merge-data --repeat``) are stored once with their
``repeat`` count. The output is one ``torch.save`` file of packed arrays plus a per-utterance index.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .alignment import sequence_mask
from .data import MelDataset, collate
from .models.text_encoder import align, token_pitch

SPLITS = ("train", "val", "dev")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="TTS checkpoint (model_ema.pt / released .pt)")
    p.add_argument("--data", default=None, help="data root (default: the model's data.root)")
    p.add_argument("--out", required=True, help="output .pt file")
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--batch-frames", type=int, default=40000, help="mel frames per batch")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="first N utterances per split (0: all; for tests)")
    p.add_argument("--word-model", default=None,
                   help="add contextual word features to an existing cache --out (e.g. dbmdz/bert-base-turkish-cased; "
                        "drifting_tts.word_features) instead of building one")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


class _Unique(Dataset):
    """The items of ``ds`` at ``indices`` (one per distinct utterance)."""

    def __init__(self, ds: MelDataset, indices: list[int]):
        self.ds, self.indices = ds, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        item = self.ds[self.indices[i]]
        item["index"] = i
        return item


def _batches(frames: list[int], budget: int) -> list[list[int]]:
    order = sorted(range(len(frames)), key=lambda i: frames[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * frames[i] > budget:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


@torch.no_grad()
def build_cache(model, root: str | Path, splits, filters: dict | None, max_frames: int, min_frames: int,
                device, batch_frames: int = 40000, num_workers: int = 2, limit: int = 0) -> dict:
    """Run the frozen encoder over ``splits`` of ``root``; see the module docstring for the stored fields."""
    lf0_mean, lf0_std = (float(v) for v in model.lf0_stats)
    toks = {k: [] for k in ("ids", "dur", "pitch", "voiced", "logw_det", "pitch_det")}
    utts, start = [], 0
    for split in splits:
        if split == "train":
            ds = MelDataset(root, split, min_frames=min_frames, max_frames=max_frames, with_f0=True, filters=filters)
        else:
            ds = MelDataset(root, split, min_frames=1, max_frames=10**9, with_f0=True)
        repeat = Counter(e["offset"] for e in ds.items)
        seen, unique = set(), []
        for i, e in enumerate(ds.items):
            if e["offset"] not in seen:
                seen.add(e["offset"])
                unique.append(i)
        unique = unique[:limit] if limit else unique
        sub = _Unique(ds, unique)
        batches = _batches([ds.items[i]["frames"] for i in unique], batch_frames)
        loader = DataLoader(sub, batch_sampler=batches, collate_fn=collate, num_workers=num_workers)
        t0, done = time.time(), 0
        rows: dict[int, dict] = {}
        for b in loader:
            text, text_len, y, y_len, spk = (b[k].to(device) for k in ("text", "text_len", "mel", "mel_len", "spk"))
            h, mu, logw, x_mask = model.encoder(text, text_len, spk)
            y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
            attn, _ = align(mu, x_mask, y, y_mask)
            f0 = b["f0"].to(device)
            pitch = token_pitch(f0, attn, lf0_mean, lf0_std)[:, 0]
            voiced = torch.bmm(attn, (f0 > 0).float()[:, :, None])[:, :, 0]
            dur = attn.sum(-1)
            s = model.encoder.spk(spk)[:, :, None].expand(-1, -1, h.shape[-1])
            pitch_det = model.pitch_predictor(torch.cat([h, s], 1), x_mask)[:, 0]
            for j in range(text.shape[0]):
                n = int(text_len[j])
                rows[int(b["index"][j])] = {
                    "ids": text[j, :n].cpu().numpy().astype(np.uint8),
                    "dur": dur[j, :n].round().cpu().numpy().astype(np.int16),
                    "pitch": pitch[j, :n].cpu().numpy().astype(np.float32),
                    "voiced": voiced[j, :n].round().cpu().numpy().astype(np.int16),
                    "logw_det": logw[j, 0, :n].cpu().numpy().astype(np.float32),
                    "pitch_det": pitch_det[j, :n].cpu().numpy().astype(np.float32),
                    "frames": int(y_len[j]),
                }
            done += text.shape[0]
            if done % 2000 < text.shape[0]:
                print(f"{split}: {done}/{len(sub)} ({done / (time.time() - t0):.1f} utt/s)", flush=True)
        for k in range(len(sub)):  # back to the dataset order
            r, e = rows[k], ds.items[unique[k]]
            assert int(r["dur"].sum()) == r["frames"], "MAS durations must sum to the mel length"
            for key in toks:
                toks[key].append(r[key])
            utts.append({"split": split, "spk": e["spk_id"], "start": start, "n": len(r["ids"]),
                         "frames": r["frames"], "repeat": repeat[e["offset"]], "offset": e["offset"],
                         "audio_offset": e.get("audio_offset"), "audio_samples": e.get("audio_samples"),
                         "norm_text": e["norm_text"]})
            start += len(r["ids"])
        print(f"{split}: {len(sub)} utterances ({sum(repeat[ds.items[i]['offset']] for i in unique)} with repeats)",
              flush=True)
    packed = {k: torch.from_numpy(np.concatenate(v)) for k, v in toks.items()}
    return {"tokens": packed, "utts": utts, "lf0_stats": [lf0_mean, lf0_std]}


@torch.no_grad()
def add_word_features(cache: dict, name: str, device, batch_size: int = 64) -> None:
    """Store ``word_feats`` ``[total_words, dim]`` (fp16) and per-utterance ``word_start`` / ``n_words``."""
    from .word_features import WordEncoder

    enc = WordEncoder(name, device)
    feats, start = [], 0
    utts = cache["utts"]
    for s in range(0, len(utts), batch_size):
        chunk = utts[s: s + batch_size]
        for u, f in zip(chunk, enc([u["norm_text"] for u in chunk])):
            u["word_start"], u["n_words"] = start, f.shape[0]
            start += f.shape[0]
            feats.append(f.half())
        if (s // batch_size) % 100 == 0:
            print(f"word features: {s + len(chunk)}/{len(utts)}", flush=True)
    cache["word_feats"], cache["word_model"] = torch.cat(feats), name


def run(args) -> None:
    from .train import load_tts

    if args.word_model:
        cache = torch.load(args.out, map_location="cpu", weights_only=False)
        add_word_features(cache, args.word_model, args.device)
        torch.save(cache, args.out)
        print(f"-> {args.out}: word features {tuple(cache['word_feats'].shape)} from {args.word_model}")
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg, _ = load_tts(args.model, args.device)
    if not model.pitch_enabled:
        raise SystemExit("the prosody cache needs a pitch-conditioned model (model.pitch.enabled)")
    d = cfg.data
    root = args.data or d.root
    cache = build_cache(model, root, args.splits, d.get("filters"), d.max_frames, cfg.drift.crop_frames,
                        args.device, args.batch_frames, args.num_workers, args.limit)
    cache.update(model=str(args.model), data=str(root))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    n = cache["tokens"]["ids"].numel()
    print(f"-> {args.out}: {len(cache['utts'])} utterances, {n} tokens")
