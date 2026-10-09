"""Prosody evaluation: held-out recordings against the model's renditions of their texts, with oracle prosody.

For every utterance of a split and speaker (default: ``val``, the studio voice 722, 100 utterances) the recording's
own text is synthesised (T = 0.3, α = 2, ``vocos-ft``, seed = dataset index as in ``evaluate``) by each system:

* ``recording``: the recording (the reference); ``copy``: its mel through the vocoder (copy-synthesis ceiling);
* ``predicted``: the released pipeline as is (:class:`Synthesizer`: sentence by sentence, 0.15 s pauses);
  ``pause-punct`` / ``pause-punct-j<J>``: the same with :class:`drifting_tts.prosody.PausePolicy` (jitter J);
* ``release``: the release pipeline (v3.2): sentence by sentence, durations and token pitch sampled by the
  ``--release-prosody`` predictor, punctuation pauses, vocoded by ``--release-vocoder``;
* one pass over the whole text (no sentence split), the prosody combined from ``+``-joined parts:
  ``onepass`` (predicted durations and pitch), ``oracle-dur`` (ground-truth MAS durations: the mel has the
  recording's frame count), ``oracle-pitch`` (ground-truth token pitch), ``oracle-both``, ``pitch-gain-<g>``
  (predicted token pitch scaled around its utterance mean on voiced tokens), ``dur-gain-<g>`` (predicted letter
  log-durations scaled around their mean, total length kept), ``dur-mix-<λ>`` (``(1 - λ)`` predicted + ``λ``
  ground-truth log-durations), e.g. ``oracle-dur+pitch-gain-1.4``.

Suffixes: ``@cfg<α>``, ``@t<T>``, ``@win<N>`` (sampling) and ``@voc=<registry name>`` (another vocoder for that
system, e.g. ``predicted@voc=bigvgan-v2-ft`` next to ``predicted`` with ``--vocoder vocos-ft``).

The DiT was trained with ground-truth durations and token pitch, so the oracle rows are in distribution: the gap
between ``onepass`` and ``oracle-both`` is what the deterministic prosody predictors cost.

Metrics per system (:mod:`drifting_tts.prosody`): audio-level F0 statistics, pauses and rate; DTW log-F0 correlation /
RMSE and duration ratio against the recording; token-level agreement of the durations and pitch it used with the
ground truth; guard rails: Whisper large-v3 CER / WER (``--band``), UTMOSv2 and WavLM-ECAPA similarity to the
recording. ``--diversity K`` adds the spread over K seeds. ``--texts`` (e.g. the Freya-TR-Eval id) runs text-only
systems on an external text set (no recordings: no paired or token metrics, ASR band-matched to 8 kHz).
Feature extraction (harvest F0) runs in ``--workers`` CPU processes while the GPU synthesises and judges.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .audio import FRAME_RATE, SAMPLE_RATE
from .synthesize import add_vocoder_args
from .voices import DEFAULT_VOICE

DEFAULT_SYSTEMS = ["recording", "copy", "predicted", "onepass", "oracle-dur", "oracle-pitch", "oracle-both"]
AUDIO_COLS = [("f0_std", "F0 std", 2), ("f0_range", "F0 range", 1), ("f0_skew", "skew", 2), ("f0_kurt", "kurt", 2),
              ("f0_cv", "F0 CV", 3), ("f0_move", "move", 2), ("f0_micro", "micro", 2),
              ("f0_reversals", "reversals/s", 2), ("voiced_pct", "voiced %", 1), ("pauses", "pauses/utt", 2),
              ("pause_mean_s", "pause s", 3), ("rate_sps", "syl/s", 2)]
PAIRED_COLS = [("f0_corr", "DTW F0 r", 3), ("f0_rmse", "F0 RMSE", 2), ("dur_ratio", "dur ratio", 3),
               ("f0_tok_std", "in-token F0 std", 2), ("f0_within", "in-token share", 3), ("render_r", "render r", 3),
               ("render_flat", "render flat", 3), ("cer", "CER", None), ("wer", "WER", None), ("mos", "UTMOSv2", 3),
               ("speaker_sim", "SIM", 3)]
TOKEN_COLS = [("dur_r", "dur r", 3), ("dur_flat", "dur flat", 3), ("dur_mae", "dur MAE", 3),
              ("dur_cv", "dur CV", 3), ("dur_cv_gt", "dur CV (GT)", 3), ("len_ratio", "len ratio", 3),
              ("pitch_r", "pitch r", 3), ("pitch_flat", "pitch flat", 3), ("pitch_mae", "pitch MAE st", 2),
              ("pitch_bias", "pitch bias st", 2)]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="runs/release/drifting_tts_v3.1.pt")
    p.add_argument("--data", default=None, help="prepared data with audio.bin and f0.bin (default: data.root)")
    p.add_argument("--split", default="val", choices=["val", "dev", "train"],
                   help="train only for voices without held-out utterances (389 / 323): in-sample, say so")
    p.add_argument("--speaker", default="722", help=f"voice name or speaker ID (default: 722 = {DEFAULT_VOICE})")
    p.add_argument("--offset", type=int, default=0, help="skip the first N utterances of the speaker")
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--longest", type=int, default=0, help="keep only the N longest of the selected utterances")
    p.add_argument("--texts", default=None, help="text-only mode: HF dataset id, .jsonl or .txt (benchmark)")
    p.add_argument("--systems", nargs="+", default=None, help=f"default: {' '.join(DEFAULT_SYSTEMS)}")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--f0", default="harvest", choices=["harvest", "dio"], help="F0 tracker of the audio metrics")
    p.add_argument("--asr", default="large-v3", help="'none' disables")
    p.add_argument("--sv", default="wavlm-large-ecapa", help="'none' disables")
    p.add_argument("--mos", default="utmosv2", choices=["utmosv2", "utmos22", "none"])
    p.add_argument("--band", type=int, default=None,
                   help="resample to this rate before ASR (default: 0 = full band, as evaluate; 8000 with --texts)")
    p.add_argument("--diversity", type=int, default=0, help="seeds per text for the diversity table (0: off)")
    p.add_argument("--diversity-num", type=int, default=10, help="texts for the diversity table")
    p.add_argument("--diversity-systems", nargs="+", default=["onepass"])
    p.add_argument("--workers", type=int, default=3, help="CPU processes for F0 extraction (<= 3 on a shared box)")
    p.add_argument("--overwrite", action="store_true", help="recompute systems whose results already exist")
    p.add_argument("--out", default="outputs/prosody")
    p.add_argument("--release-prosody", default=None,
                   help="system 'release': the stochastic prosody predictor (drift or a checkpoint)")
    p.add_argument("--release-prosody-temperature", type=float, default=None, help="default: the checkpoint's")
    p.add_argument("--release-vocoder", default=None, help="system 'release': its vocoder (default: --vocoder)")
    p.add_argument("--release-model", default=None, help="system 'release': its acoustic model (default: --model)")
    add_vocoder_args(p, default="vocos-ft")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


# --- systems -----------------------------------------------------------------------------------------------------


@dataclass
class System:
    name: str
    kind: str  # recording | copy | split | onepass
    dur: tuple = ("pred",)  # ("pred",) | ("gt",) | ("gain", g) | ("mix", lam)
    pitch: tuple = ("pred",)  # ("pred",) | ("gt",) | ("gain", g)
    jitter: float | None = None  # split: None = fixed 0.15 s, else a PausePolicy with this jitter
    cfg: float | None = None  # overrides of --cfg / --temperature, and a DiT attention window (@cfg, @t, @win)
    temperature: float | None = None
    attn_window: int | None = None
    vocoder: str | None = None  # another vocoder (@voc=<name>; 'release': --release-vocoder)
    sampled: bool = False  # 'release': durations and token pitch from --release-prosody

    @property
    def needs_recording(self) -> bool:
        return self.kind in ("recording", "copy") or "gt" in (self.dur[0], self.pitch[0]) or self.dur[0] == "mix"


def parse_system(name: str) -> System:
    """A system name (see the module docstring) -> :class:`System`. Suffixes ``@cfg<α>``, ``@t<T>`` and
    ``@win<N>`` (DiT attention radius in tokens of 2 frames; training crops are 128 tokens) change the sampling."""
    base, *mods = name.split("@")
    s = _parse_base(name, base)
    for mod in mods:
        if m := re.fullmatch(r"(cfg|t|win)([\d.]+)", mod):
            key = {"cfg": "cfg", "t": "temperature", "win": "attn_window"}[m.group(1)]
            setattr(s, key, int(m.group(2)) if key == "attn_window" else float(m.group(2)))
        elif m := re.fullmatch(r"voc=([\w.-]+)", mod):
            from .vocoder import VOCODER_ALIASES, VOCODERS

            if m.group(1) not in VOCODERS and m.group(1) not in VOCODER_ALIASES:
                raise ValueError(f"{name!r}: @voc= takes a vocoder registry name ({', '.join(VOCODERS)})")
            s.vocoder = m.group(1)
        else:
            raise ValueError(f"unknown system suffix @{mod} in {name!r}")
    if s.kind == "recording" and mods or s.kind == "copy" and any(not m.startswith("voc=") for m in mods):
        raise ValueError(f"{name!r}: the recording rows take no sampling options")
    return s


def _parse_base(name: str, base: str) -> System:
    if base in ("recording", "copy"):
        return System(name, base)
    if base == "predicted":
        return System(name, "split")
    if base == "release":
        return System(name, "split", jitter=0.0, vocoder="release", sampled=True)
    if m := re.fullmatch(r"pause-punct(?:-j([\d.]+))?", base):
        return System(name, "split", jitter=float(m.group(1) or 0.0))
    s = System(name, "onepass")
    for part in base.split("+"):
        if part == "onepass":
            continue
        if part in ("oracle-dur", "oracle-both"):
            s.dur = ("gt",)
        if part in ("oracle-pitch", "oracle-both"):
            s.pitch = ("gt",)
        if part in ("oracle-dur", "oracle-pitch", "oracle-both"):
            continue
        if m := re.fullmatch(r"(pitch-gain|dur-gain|dur-mix)-([\d.]+)", part):
            v = float(m.group(2))
            if m.group(1) == "pitch-gain":
                s.pitch = ("gain", v)
            else:
                s.dur = ("gain", v) if m.group(1) == "dur-gain" else ("mix", v)
            continue
        raise ValueError(f"unknown system part {part!r} in {name!r}")
    return s


@dataclass
class Utterance:
    index: int  # dataset index (or text index) = seed
    text: str  # normalised
    ids: torch.Tensor  # [1, N] on the device
    mel: torch.Tensor | None = None  # normalised [1, C, T] on the device
    f0: torch.Tensor | None = None  # [1, T] Hz on the mel frames
    audio: torch.Tensor | None = None  # recording, 24 kHz
    targets: dict | None = None  # prosody.token_targets


class Runner:
    """Synthesises the systems of :func:`parse_system` for one speaker."""

    def __init__(self, synth, speaker: int, temperature: float, cfg_scale: float, release: dict | None = None):
        """``release``: ``vocoder`` and ``prosody`` / ``prosody_temperature`` of the ``release`` system."""
        self.synth, self.model, self.device = synth, synth.model, synth.device
        self.spk = torch.tensor([speaker], device=self.device)
        self.speaker = speaker
        self.tempo = getattr(self.model, "duration_scales", {}).get(speaker, self.model.duration_scale)
        self.temperature, self.cfg_scale = temperature, cfg_scale
        self.release = release or {}
        self._synths: dict[tuple, object] = {}

    def synth_for(self, s: System):
        """The Synthesizer of a system: the shared one, or a variant (same acoustic model) with another vocoder
        and / or the release's prosody predictor (with ``--release-model``: a Synthesizer of that model)."""
        voc = self.release.get("vocoder") if s.vocoder == "release" else s.vocoder
        key = (voc, s.sampled)
        if key == (None, False):
            return self.synth
        if key not in self._synths:
            if s.sampled and not self.release.get("prosody"):
                raise SystemExit(f"system {s.name!r} needs --release-prosody")
            base = self.synth
            if s.sampled and self.release.get("model"):  # another acoustic model
                from .synthesize import Synthesizer

                base = Synthesizer(self.release["model"], self.device, vocoder=voc or self.synth.vocoder.name)
                voc = None
            kw = {} if voc is None else {"vocoder": voc}
            if s.sampled:
                kw.update(prosody=self.release["prosody"], prosody_temperature=self.release.get("prosody_temperature"))
            self._synths[key] = base.variant(**kw)
        return self._synths[key]

    def prosody(self, u: Utterance, s: System, seed_offset: int = 0) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Durations (frames ``[1, N]``, ``None``: predicted) and token pitch (``[1, 1, N]``, ``None``: predicted)."""
        from .prosody import frames_from_logw, letter_tokens, scale_deviations, voiced_tokens

        t, durations, pitch = u.targets, None, None
        if s.dur[0] == "gt":
            durations = t["durations_gt"][None]
        elif s.dur[0] == "gain":
            logw = t["logw"] + math.log(self.tempo)
            durations = frames_from_logw(scale_deviations(logw, s.dur[1], letter_tokens(t["ids"]), keep_total=True))
            durations = durations[None]
        elif s.dur[0] == "mix":
            lam = s.dur[1]
            logd = (1 - lam) * (t["logw"] + math.log(self.tempo)) + lam * t["durations_gt"].float().log()
            durations = torch.ceil(torch.exp(logd) - 1e-4).clamp_min(1)[None]
        if s.pitch[0] == "gt":
            pitch = t["pitch_gt"][None, None]
        elif s.pitch[0] == "gain":
            pitch = scale_deviations(t["pitch"], s.pitch[1], voiced_tokens(t["ids"]))[None, None]
        return durations, pitch

    @torch.no_grad()
    def __call__(self, u: Utterance, s: System, seed: int | None = None) -> torch.Tensor:
        seed = u.index if seed is None else seed
        synth = self.synth_for(s)
        cfg = self.cfg_scale if s.cfg is None else s.cfg
        temperature = self.temperature if s.temperature is None else s.temperature
        if s.kind == "recording":
            return u.audio
        if s.kind == "copy":
            return synth.vocoder(synth.stats.denormalize(u.mel))[0].cpu()
        if s.kind == "split":
            pause = 0.15
            if s.jitter is not None:
                from .prosody import PausePolicy

                pause = PausePolicy.for_voice(self.speaker, jitter=s.jitter)
            return synth(u.text, speaker=self.speaker, cfg_scale=cfg, temperature=temperature, seed=seed,
                         pause=pause, attn_window=s.attn_window)[0]
        durations, pitch = self.prosody(u, s)
        g = torch.Generator(device=self.device).manual_seed(seed)
        mel, _ = self.model.synthesize(u.ids, torch.tensor([u.ids.shape[1]], device=self.device), self.spk,
                                       cfg_scale=cfg, temperature=temperature, length_scale=self.tempo, generator=g,
                                       durations=durations, pitch=pitch, attn_window=s.attn_window)
        if s.dur[0] == "gt" and u.mel is not None:
            assert mel.shape[-1] == u.mel.shape[-1], "oracle durations must reproduce the recording's frame count"
        return synth.vocoder(synth.stats.denormalize(mel))[0].cpu()

    def alignment(self, u: Utterance, s: System) -> dict | None:
        """Mel frames per token, the token pitch the audio was conditioned on (absolute semitones re 1 Hz) and the
        tokens to compare (voiced in the recording, else :func:`voiced_tokens`): for :func:`prosody.token_f0`."""
        from .prosody import ST_PER_LN, frames_from_logw, voiced_tokens

        t = u.targets
        if s.kind == "split" or t is None or (s.kind == "copy" and "durations_gt" not in t):
            return None
        if s.kind in ("recording", "copy"):
            frames, pitch = t["durations_gt"], t.get("pitch_gt")
        else:
            d, p = self.prosody(u, s)
            frames = frames_from_logw(t["logw"], self.tempo) if d is None else d[0]
            pitch = t.get("pitch") if p is None else p[0, 0]
        mask = t["voiced_gt"] if "voiced_gt" in t else voiced_tokens(t["ids"])
        out = {"frames": frames.float().cpu().numpy(), "mask": mask.cpu().numpy()}
        if pitch is not None:
            lf0_mean, lf0_std = self.model.lf0_stats.tolist()
            out["pitch_st"] = ((pitch.float() * lf0_std + lf0_mean) * ST_PER_LN).cpu().numpy()
        return out

    def token_report(self, u: Utterance, s: System) -> dict:
        """Agreement of the durations / pitch a one-pass system uses with the ground truth."""
        from .prosody import frames_from_logw, token_report

        if s.kind != "onepass" or u.targets is None or "durations_gt" not in u.targets:
            return {}
        durations, pitch = self.prosody(u, s)
        frames = frames_from_logw(u.targets["logw"], self.tempo) if durations is None else durations[0]
        return token_report(u.targets, FRAME_RATE, frames=frames, pitch=None if pitch is None else pitch[0, 0],
                            lf0_std=float(self.model.lf0_stats[1]))


# --- CPU feature workers -----------------------------------------------------------------------------------------


def _init_worker() -> None:
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    torch.set_num_threads(1)


def _features(path: str, text: str, method: str, ref_path: str | None, ref_f0: np.ndarray | None,
              align: dict | None = None) -> dict:
    from .prosody import f0_contour, paired_metrics, prosody_features, token_f0

    wav, sr = sf.read(path, dtype="float64")
    f0 = f0_contour(wav, sr, method)
    row = prosody_features(wav, sr, f0=f0, text=text)
    if ref_path is not None and ref_path != path:
        ref, _ = sf.read(ref_path, dtype="float64")
        row.update(paired_metrics(ref, wav, sr, ref_f0=ref_f0, gen_f0=f0))
    if align is not None:
        row.update(token_f0(f0, align["frames"], align.get("pitch_st"), align["mask"]))
    row["_f0"] = f0.astype(np.float32)
    return row


def _diversity(paths: list[str], method: str, lengths: list[float], durations: list | None) -> dict:
    from .prosody import f0_contour, mfcc, seed_diversity

    wavs = [sf.read(p, dtype="float64")[0] for p in paths]
    return seed_diversity([f0_contour(w, SAMPLE_RATE, method) for w in wavs], lengths, durations,
                          [mfcc(w) for w in wavs])


# --- judges ------------------------------------------------------------------------------------------------------


def retry_oom(fn, *args, tries: int = 30, wait: float = 30.0, **kw):
    """``fn(*args, **kw)``, retried after a CUDA out-of-memory error (a GPU shared with other jobs)."""
    import time

    for k in range(tries):
        try:
            return fn(*args, **kw)
        except RuntimeError as e:  # torch.OutOfMemoryError, CTranslate2's "CUDA failed with error out of memory"
            if "out of memory" not in str(e) or k == tries - 1:
                raise
            print(f"CUDA out of memory, retry {k + 1}/{tries} in {wait:g} s", flush=True)
            torch.cuda.empty_cache()
            time.sleep(wait)


def judge(judges, wav: torch.Tensor, text: str, band: int, spk_ref: torch.Tensor | None):
    """CER / WER counts, UTMOSv2 and speaker similarity of one utterance; returns the row and its embedding."""
    from .benchmark import band_match
    from .evaluate import _plain
    from .metrics import error_counts

    row: dict = {}
    if judges.asr is not None:
        ref = _plain(text)
        row["hyp"] = _plain(judges.asr(band_match(wav, band)))
        row.update(error_counts(ref, row["hyp"]))
    full = band_match(wav, 0)
    emb = judges.sv(full) if judges.sv is not None else None
    if emb is not None and spk_ref is not None:
        row["speaker_sim"] = float(emb @ spk_ref)
    if judges.mos is not None:
        row["mos"] = judges.mos(full)
    return row, emb


def missing_judges(row: dict, judges, system: str) -> bool:
    """Whether a cached utterance row lacks a score that an enabled judge gives."""
    return ((judges.asr is not None and "hyp" not in row) or (judges.mos is not None and "mos" not in row)
            or (judges.sv is not None and system != "recording" and "speaker_sim" not in row))


# --- aggregation -------------------------------------------------------------------------------------------------


def aggregate(rows: list[dict]) -> dict:
    """Means over utterances (pause lengths pooled), corpus CER / WER and the judges' intervals."""
    from .evaluate import summarize

    out: dict = {"utterances": len(rows)}
    skip = {"index", "hyp", "pause_lengths", "punct", "char_errors", "chars", "word_errors", "words"}
    keys = sorted({k for r in rows for k, v in r.items() if k not in skip and isinstance(v, (int, float))})
    for k in keys:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not math.isnan(r[k])]
        if v:
            out[k] = float(np.mean(v))
    lengths = [x for r in rows for x in r.get("pause_lengths", [])]
    out["pause_mean_s"] = float(np.mean(lengths)) if lengths else float("nan")
    punct: dict[str, list] = {}
    for r in rows:
        for mark, pairs in r.get("punct", {}).items():
            punct.setdefault(mark, []).extend(pairs)
    if punct:
        out["punct"] = {m: {"n": len(p), "pred_mean": float(np.mean([a for a, _ in p])),
                            "pred_std": float(np.std([a for a, _ in p])), "gt_mean": float(np.mean([b for _, b in p])),
                            "gt_std": float(np.std([b for _, b in p]))} for m, p in sorted(punct.items())}
    out.update(summarize(rows))
    return out


def format_table(rows: list[dict], cols: list[tuple]) -> str:
    cols = [c for c in cols if any(c[0] in r for r in rows)]

    def fmt(r, key, digits):
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "–"
        return f"{100 * v:.2f}%" if digits is None else f"{v:.{digits}f}"

    lines = ["| system | " + " | ".join(c[1] for c in cols) + " |", "|---" + "|---:" * len(cols) + "|"]
    lines += [f"| {r['system']} | " + " | ".join(fmt(r, k, d) for k, _, d in cols) + " |" for r in rows]
    return "\n".join(lines)


# --- main --------------------------------------------------------------------------------------------------------


def load_utterances(args, synth, speaker: int) -> list[Utterance]:
    from .data import MelDataset
    from .prosody import token_targets
    from .text import normalize, text_to_ids

    dev = synth.device
    if args.texts:
        from .benchmark import load_texts

        items = load_texts(args.texts)[args.offset: args.offset + args.num]
        out = []
        for k, item in enumerate(items):
            text = normalize(item["text"])
            ids = torch.tensor([text_to_ids(text, normalized=True)], device=dev)
            u = Utterance(args.offset + k, text, ids)
            u.targets = token_targets(synth.model, ids, torch.tensor([speaker], device=dev))
            out.append(u)
        return out
    root = Path(args.data or synth.cfg.data.root)
    ds = MelDataset(root, args.split, min_frames=1, max_frames=10**9, with_f0=True, with_audio=True)
    idx = [i for i, e in enumerate(ds.items) if e["spk_id"] == speaker][args.offset: args.offset + args.num]
    if args.longest:
        idx = sorted(sorted(idx, key=lambda i: -ds.items[i]["frames"])[: args.longest])
    if not idx:
        raise SystemExit(f"speaker {speaker} has no utterances in the '{args.split}' split of {root}")
    out = []
    for i in idx:
        it = ds[i]
        ids, mel, f0 = it["text"][None].to(dev), it["mel"][None].to(dev), it["f0"][None].to(dev)
        u = Utterance(i, ds.items[i]["norm_text"], ids, mel, f0, it["audio"])
        u.targets = token_targets(synth.model, ids, torch.tensor([speaker], device=dev), mel, f0)
        assert int(u.targets["durations_gt"].sum()) == mel.shape[-1]
        out.append(u)
    return out


def run(args) -> None:
    from multiprocessing import get_context

    from .judges import load_judges
    from .synthesize import Synthesizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.systems or (["predicted", "onepass"] if args.texts else DEFAULT_SYSTEMS)
    systems = [parse_system(n) for n in names]
    if args.texts and any(s.needs_recording for s in systems):
        raise SystemExit("--texts has no recordings: only predicted / onepass / *-gain / pause-punct systems")
    synth = Synthesizer(args.model, args.device, vocoder=args.vocoder, cuda_kernel=args.cuda_kernel)
    speaker = synth.speaker_id(args.speaker)
    release = {"vocoder": args.release_vocoder, "prosody": args.release_prosody,
               "prosody_temperature": args.release_prosody_temperature,
               "model": args.release_model if args.release_model not in (None, args.model) else None}
    runner = Runner(synth, speaker, args.temperature, args.cfg, release)
    utts = load_utterances(args, synth, speaker)
    band = (8000 if args.texts else 0) if args.band is None else args.band
    judges = retry_oom(load_judges, args.asr, args.sv if not args.texts else None, args.mos, args.device)
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init_worker)
    with_rec = not args.texts
    if with_rec and systems[0].name != "recording":  # the reference for the paired metrics and similarity
        systems = [parse_system("recording")] + [s for s in systems if s.name != "recording"]
    rec_f0: dict[int, np.ndarray] = {}
    spk_ref: dict[int, torch.Tensor] = {}
    results: dict[str, dict] = {}
    pending: list[tuple[System, list[dict], list[Future]]] = []

    def wav_path(s: System, u: Utterance) -> Path:
        return out / "wav" / s.name / f"{u.index:05d}.wav"

    def finish(s: System, rows: list[dict], futs: list[Future]) -> None:
        for r, f in zip(rows, futs):
            feats = f.result()
            f0 = feats.pop("_f0")
            if s.name == "recording":
                rec_f0[r["index"]] = f0
            r.update(feats)
        if s.name == "recording":
            np.savez_compressed(out / "wav" / s.name / "f0.npz", **{str(k): v for k, v in rec_f0.items()})
        with open(out / f"utterances_{s.name}.jsonl", "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        results[s.name] = {"system": s.name, **aggregate(rows)}
        print(json.dumps({k: v for k, v in results[s.name].items() if k != "punct"}), flush=True)

    for s in systems:
        cached = out / f"utterances_{s.name}.jsonl"
        if cached.exists() and not args.overwrite:
            rows = [json.loads(line) for line in cached.read_text().splitlines()]
            if len(rows) == len(utts):
                if s.name == "recording":
                    z = np.load(out / "wav" / s.name / "f0.npz")
                    rec_f0.update({int(k): z[k] for k in z.files})
                    if judges.sv is not None:  # the speaker references
                        from .benchmark import band_match

                        spk_ref.update({u.index: retry_oom(judges.sv, band_match(u.audio, 0)) for u in utts})
                if missing_judges(rows[0], judges, s.name):  # e.g. a first pass with --asr / --mos / --sv none
                    for r, u in zip(rows, utts):  # only the scores a row lacks
                        need = replace(judges, asr=None if "hyp" in r else judges.asr,
                                       mos=None if "mos" in r else judges.mos,
                                       sv=None if s.name == "recording" or "speaker_sim" in r else judges.sv)
                        wav = torch.from_numpy(sf.read(wav_path(s, u), dtype="float32")[0])
                        r.update(retry_oom(judge, need, u.audio if s.name == "recording" else wav, u.text, band,
                                           spk_ref.get(u.index))[0])
                    cached.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                    print(f"{s.name}: judges added", flush=True)
                results[s.name] = {"system": s.name, **aggregate(rows)}
                print(f"{s.name}: cached ({cached})", flush=True)
                continue
        if s.name != "recording" and with_rec:  # paired metrics need every recording's F0
            for prev in [p for p in pending if p[0].name == "recording"]:
                finish(*prev)
                pending.remove(prev)
        (out / "wav" / s.name).mkdir(parents=True, exist_ok=True)
        rows, futs = [], []
        for u in utts:
            wav = retry_oom(runner, u, s).float().cpu()
            path = wav_path(s, u)
            sf.write(path, wav.numpy(), SAMPLE_RATE, subtype="FLOAT")  # lossless: judges may read it later
            ref = wav_path(parse_system("recording"), u) if with_rec else None
            futs.append(pool.submit(_features, str(path), u.text, args.f0, None if ref is None else str(ref),
                                    rec_f0.get(u.index), runner.alignment(u, s)))
            row, emb = retry_oom(judge, judges, wav, u.text, band, spk_ref.get(u.index))
            if s.name == "recording":
                spk_ref[u.index] = emb
            row.update(index=u.index, text=u.text, seconds=wav.numel() / SAMPLE_RATE, **runner.token_report(u, s))
            rows.append(row)
        pending.append((s, rows, futs))
        print(f"{s.name}: synthesised and judged {len(rows)} utterances", flush=True)
        for p in [p for p in pending if all(f.done() for f in p[2])]:
            finish(*p)
            pending.remove(p)
    for p in pending:
        finish(*p)

    diversity = run_diversity(args, runner, utts, pool, out) if args.diversity else None
    pool.shutdown()
    rows = [results[s.name] for s in systems if s.name in results]
    report = {"model": args.model, "vocoder": args.vocoder, "release": release,
              "split": None if args.texts else args.split,
              "texts": args.texts, "speaker": speaker, "num_utterances": len(utts), "offset": args.offset,
              "longest": args.longest or None,
              "temperature": args.temperature, "cfg": args.cfg, "duration_scale": runner.tempo, "f0": args.f0,
              "band_hz": band, "judges": {"asr": args.asr, "sv": None if args.texts else args.sv, "mos": args.mos},
              "rows": rows, "diversity": diversity}
    (out / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    md = format_table(rows, AUDIO_COLS) + "\n\n" + format_table(rows, PAIRED_COLS)
    tok = [r for r in rows if "dur_r" in r or "pitch_r" in r]
    if tok:
        md += "\n\nToken level (predictors against MAS durations / ground-truth token pitch):\n\n"
        md += format_table(tok, TOKEN_COLS)
    if diversity:
        md += "\n\nSeed diversity:\n\n" + format_table(diversity["rows"], [
            ("f0_spread", "F0 spread st", 3), ("f0_spread_mean", "mean", 3), ("f0_std_cv", "F0 std CV", 3),
            ("len_cv", "length CV", 4),
            ("token_dur_std", "token log-dur std", 4)])
    (out / "results.md").write_text(md + "\n")
    print(md)
    print(f"-> {out / 'results.json'}")


def run_diversity(args, runner: Runner, utts: list[Utterance], pool, out: Path) -> dict:
    """F0 / duration spread over ``--diversity`` seeds for the first ``--diversity-num`` texts."""
    from .prosody import frames_from_logw

    tmp = out / "wav" / "_diversity"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.diversity_systems:
        s = parse_system(name)
        jobs = []
        for u in utts[: args.diversity_num]:
            paths, lengths, durs = [], [], []
            for k in range(args.diversity):
                wav = retry_oom(runner, u, s, seed=1000 * k + u.index).float().cpu()
                path = tmp / f"{s.name}_{u.index:05d}_{k}.wav"
                sf.write(path, wav.numpy(), SAMPLE_RATE)
                paths.append(path)
                lengths.append(wav.numel() / SAMPLE_RATE)
                if s.kind == "onepass":
                    d, _ = runner.prosody(u, s)
                    durs.append(frames_from_logw(u.targets["logw"], runner.tempo) if d is None else d[0])
            jobs.append(pool.submit(_diversity, [str(p) for p in paths], args.f0, lengths,
                                    [d.cpu() for d in durs] or None))
        per_text = [j.result() for j in jobs]
        row = {"system": s.name, "texts": len(per_text), "seeds": args.diversity}
        row.update({k: float(np.mean([d[k] for d in per_text])) for k in per_text[0]})
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"rows": rows}
