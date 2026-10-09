"""Checkpoints for publishing: keep what inference loads, drop training state and local paths, scan the result.

Training checkpoints carry their config with absolute paths, data roots and run names (docs/EXPERIMENTS.md §6). The
``export_*`` functions keep only what the loaders read; :func:`scan` lists the strings of a ``torch.save`` file (the
pickle's own strings, and printable runs of the whole file, as ``strings`` does) and counts the ones that match local
paths or any of the given private terms (dataset, show or speaker names), without printing the terms themselves.
"""

from __future__ import annotations

import io
import pickletools
import re
import zipfile
from pathlib import Path

import torch

# acoustic model: what drifting_tts.train.load_tts and Synthesizer read
TTS_KEYS = ("ema", "config", "num_speakers", "n_mels", "step", "stats", "duration_scale", "temperature",
            "duration_scales")
# Vocos fine-tune: what drifting_tts.vocoder._from_checkpoint reads
VOCOS_KEYS = ("vocos", "init", "step", "mel", "head_padding", "noise_channels")
PATH_RE = re.compile(r"(/workspace|/root|/home/|/tmp/|/mnt/|/data/|/Users/|[A-Za-z]:\\)")  # absolute paths
GENERIC_DATA_ROOT = "data/train"


def _tensors_to_cpu(state: dict) -> dict:
    return {k: v.detach().cpu().contiguous().clone() if torch.is_tensor(v) else v for k, v in state.items()}


def export_vocos(ck: dict) -> dict:
    """A ``finetune-vocoder`` Vocos checkpoint -> ``{vocos, init, step, mel, head_padding}``. ``init`` must be a Hub
    repo id (a local YAML config would not load elsewhere)."""
    out = {k: ck[k] for k in VOCOS_KEYS if k in ck}
    if "vocos" not in out:
        raise ValueError("not a Vocos checkpoint (no 'vocos' state dict)")
    init = str(out.get("init", ""))
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", init) or Path(init).exists():
        raise ValueError(f"Vocos 'init' must be a Hub repo id such as charactr/vocos-mel-24khz, got {init!r}")
    out["vocos"] = _tensors_to_cpu(out["vocos"])
    return out


def _clean_config(node, key: str = ""):
    """Config with local paths replaced: ``data.root`` -> :data:`GENERIC_DATA_ROOT`, other path strings -> their
    file name under ``runs/`` (as the released v3.1's ``mae.path``)."""
    if isinstance(node, dict):
        return {k: _clean_config(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_clean_config(v, key) for v in node]
    if isinstance(node, str) and (node.startswith("/") or "/" in node and Path(node).suffix in (".pt", ".yaml")):
        return f"runs/{Path(node).parent.name}/{Path(node).name}" if Path(node).suffix else GENERIC_DATA_ROOT
    return node


def export_tts(ck: dict) -> dict:
    """An exported acoustic model (``model_ema.pt``) -> what ``load_tts`` reads, the config's paths made generic
    (``data.root``, ``mae.path``, any absolute path) and training-only state (optimiser, ``taus``, discriminators)
    dropped. Speaker names are not part of a checkpoint (``speakers.json`` lives in the data root)."""
    out = {k: ck[k] for k in TTS_KEYS if k in ck}
    out["ema"] = _tensors_to_cpu(out["ema"])
    cfg = _clean_config(dict(out["config"]))
    if "data" in cfg:
        cfg["data"] = {**cfg["data"], "root": GENERIC_DATA_ROOT}
    out["config"] = cfg
    return out


def pickle_strings(path: str | Path) -> list[str]:
    """Every string of the pickle(s) inside a ``torch.save`` zip file (keys, config values, class names)."""
    out: list[str] = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".pkl"):
                for op, arg, _ in pickletools.genops(io.BytesIO(z.read(name))):
                    if isinstance(arg, str) and "UNICODE" in op.name or op.name in ("STRING", "BINSTRING",
                                                                                      "SHORT_BINSTRING"):
                        out.append(arg if isinstance(arg, str) else str(arg))
    return out


def printable_runs(path: str | Path, min_len: int = 6) -> list[str]:
    """ASCII runs of at least ``min_len`` printable characters in the whole file, as ``strings -n``."""
    return [m.group().decode("ascii") for m in re.finditer(rb"[\x20-\x7e]{%d,}" % min_len, Path(path).read_bytes())]


def scan(path: str | Path, terms: dict[str, list[str]] | None = None) -> dict[str, int]:
    """Hits per category in ``path``: ``paths`` (local path patterns in the pickle's strings) and, per category of
    ``terms`` (e.g. speaker or dataset names; case-insensitive, terms shorter than 4 characters ignored), matches in
    the pickle's strings or in the file's printable runs. All zero: nothing private found."""
    strings = pickle_strings(path)
    runs = printable_runs(path)
    hits = {"paths": sum(bool(PATH_RE.search(s)) for s in strings)}
    text = "\n".join(strings + runs).lower()
    for cat, words in (terms or {}).items():
        hits[cat] = sum(text.count(w.lower()) for w in {w for w in words if len(w) >= 4})
    return hits


def private_terms(data_root: str | Path) -> dict[str, list[str]]:
    """The names a published file must not contain, from a prepared data root: the data root's own name, the
    speaker names (``speakers.json``) and the source fields of ``index.jsonl`` (``show_name``, ``episode_name``,
    ``dataset``, ``source``)."""
    import json

    root = Path(data_root)
    terms: dict[str, list[str]] = {"data root": [root.name, *root.name.split("_")]}
    spk = root / "speakers.json"
    if spk.exists():
        terms["speaker names"] = list(json.loads(spk.read_text()))
    sources: set[str] = set()
    index = root / "index.jsonl"
    if index.exists():
        with open(index) as f:
            for line in f:
                row = json.loads(line)
                sources.update(str(row[k]) for k in ("show_name", "episode_name", "dataset", "source") if row.get(k))
    terms["source names"] = sorted(sources)
    return terms
