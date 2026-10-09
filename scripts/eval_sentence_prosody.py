"""Intonation by sentence type: final pitch movement, polar-question peaks, wh-word peaks and phrase breaks (#40).

Per sentence (:mod:`drifting_tts.sentence_features` decides the type), on the token pitch in semitones (the continuous
contour, on the tokens of voiced letters, :func:`drifting_tts.prosody.voiced_tokens`) and the frames per token:

* ``final``: mean pitch of the last word minus the sentence mean; ``final_step``: last word minus the word before;
* polar questions (mI): ``pre_mi`` (the host word before the last mI minus the sentence mean), ``fall`` (last word
  minus the host) and ``ends_low`` (share with ``fall`` < 0);
* wh-questions: ``wh_peak`` (the first wh-word minus the sentence mean);
* ``breaks``: word boundaries inside the sentence whose non-letter tokens (blanks, punctuation, space) last at least
  ``--break-frames`` frames; ``breaks_nc`` counts them only at boundaries without a comma;
* ``final_len``: frames per letter of the last word over the sentence's mean.

Two modes:

* ``--heldout``: the held-out recordings (``val`` + ``dev`` of a prosody cache; MAS durations and token pitch) against
  every system, each utterance sampled in one pass (as in training); ``--reference train`` adds the recordings of the
  training split (no model) as the reference distribution.
* ``--texts <jsonl>`` (``scripts/prosody_diagnostic_tr.jsonl``: ``id``, ``type``, ``group``, ``text``): each sentence
  alone, as ``Synthesizer`` sends it, for ``--speakers`` and ``--seeds``. ``--audio`` also synthesises it (T 0.3,
  α 2) with ``--vocoder`` and measures the same quantities on the harvest F0 of the audio (word spans from the frames
  that were used), the sentence-final F0 (median of the last 25 voiced, non-silent frames against the sentence
  median, as in docs/POCKET_TTS_GATE.md) and the internal pauses (>= 100 ms, 35 dB below the loud level).

Systems: ``name=regressors`` (the TTS model's own), ``name=<prosody_ema.pt>@<T>[,<T pitch>][:<spread>[,<pitch
spread>]]``.

    python scripts/eval_sentence_prosody.py --heldout --reference train --cache runs/pm_cache/targets_v31.pt \\
        --systems v3.1=regressors v3.2=runs/pm_drift_final/prosody_ema.pt@0.5 --out runs/p2_eval/heldout.json
    python scripts/eval_sentence_prosody.py --texts scripts/prosody_diagnostic_tr.jsonl --speakers 722 --seeds 2 \\
        --systems v3.1=regressors v3.2=runs/pm_drift_final/prosody_ema.pt@0.5 --audio --vocoder vocos-ft \\
        --out runs/p2_eval/diag.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from drifting_tts.models.prosody_net import ProsodyPredictor, word_index
from drifting_tts.prosody import voiced_tokens
from drifting_tts.sentence_features import SENTENCE_FEATURES, sentence_features, sentence_type
from drifting_tts.text import SYMBOLS, ids_to_text, normalize, split_sentences, text_to_ids
from drifting_tts.train_prosody import ProsodyData, load_frozen_tts, sample_split

COL = {k: i for i, k in enumerate(SENTENCE_FEATURES)}
LETTER = np.array([len(s) == 1 and s.isalpha() for s in SYMBOLS])
MEL_FPS = 24000 / 256


def word_spans(ids: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Word index per token and per word the token range ``[first letter, last letter]``."""
    word = word_index(torch.from_numpy(ids)[None])[0].numpy()
    spans = []
    for w in range(word.max() + 1):
        let = np.flatnonzero((word == w) & LETTER[ids])
        spans.append((int(let[0]), int(let[-1])) if len(let) else (-1, -1))
    return word, spans


def sentence_metrics(text: str, ids: np.ndarray, frames: np.ndarray, pitch_st: np.ndarray, voiced: np.ndarray,
                     break_frames: int) -> list[dict]:
    """Per sentence of ``text`` (normalised; ``ids`` its tokens): see the module docstring. ``word_values`` (the mean
    pitch per word, relative to the sentence mean) is kept for the placement series."""
    words = text.split(" ")
    feats = sentence_features(text)
    word, spans = word_spans(ids)
    wp = np.full(len(words), np.nan)
    for w in range(len(words)):
        sel = (word == w) & voiced
        if sel.any():
            wp[w] = pitch_st[sel].mean()
    nlet = np.array([max(1, ((word == w) & LETTER[ids]).sum()) for w in range(len(words))])
    wfr = np.array([frames[(word == w) & LETTER[ids]].sum() for w in range(len(words))])
    out, a = [], 0
    for b in np.flatnonzero(feats[:, COL["sentence_final"]]) + 1:
        sw = range(a, b)
        tok = np.isin(word, list(sw)) & voiced
        if tok.sum() < 3 or b - a < 2:
            a = b
            continue
        mean = pitch_st[tok].mean()
        kind = sentence_type(words[a:b])
        r = {"type": kind, "n_words": int(b - a), "final": wp[b - 1] - mean, "final_step": wp[b - 1] - wp[b - 2],
             "word_values": [float(v - mean) for v in wp[a:b]]}
        hosts = [w for w in sw if feats[w, COL["mi_host"]]]
        if kind == "polar_q" and hosts:
            h = hosts[-1]
            r.update(pre_mi=wp[h] - mean, fall=wp[b - 1] - wp[h], host=h - a)
        whs = [w for w in sw if feats[w, COL["wh_word"]]]
        if kind == "wh_q" and whs:
            r["wh_peak"] = wp[whs[0]] - mean
        brk, brk_nc = 0, 0
        for w in range(a, b - 1):
            (_, last), (first, _) = spans[w], spans[w + 1]
            if last < 0 or first < 0:
                continue
            gap = frames[last + 1: first].sum()
            if gap >= break_frames:
                brk += 1
                brk_nc += not words[w].endswith(",")
        r.update(breaks=brk, breaks_nc=brk_nc, has_comma=any(words[w].endswith(",") for w in range(a, b - 1)),
                 final_len=float((wfr[b - 1] / nlet[b - 1]) / (wfr[a:b].sum() / nlet[a:b].sum())))
        out.append(r)
        a = b
    return out


def summarize(rows: list[dict]) -> dict:
    """Mean of every measure per sentence type (and ``all``)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
        groups["all"].append(r)
    out = {}
    for t, rs in groups.items():
        s = {"n": len(rs)}
        for k in ("final", "final_step", "pre_mi", "fall", "wh_peak", "breaks", "breaks_nc", "final_len",
                  "audio_final", "audio_final_word", "audio_pre_mi", "audio_fall", "audio_wh_peak", "audio_pauses",
                  "audio_pauses_nc"):
            v = np.array([r[k] for r in rs if k in r and r[k] is not None and not np.isnan(r[k])])
            if len(v):
                s[k] = float(v.mean())
                if k in ("fall", "audio_fall"):
                    s[k.replace("fall", "ends_low")] = float((v < 0).mean())
        nc = [r["breaks"] for r in rs if not r["has_comma"]]
        if nc:
            s["breaks_no_comma_sent"] = float(np.mean(nc))
        out[t] = s
    return out


def parse_system(spec: str):
    name, _, rest = spec.partition("=")
    if rest == "regressors":
        return name, None, None
    path, _, opts = rest.partition("@")
    temps, _, spreads = (opts or "1").partition(":")
    t = [float(x) for x in temps.split(",")]
    s = [float(x) for x in spreads.split(",")] if spreads else [1.0]
    return name, path, {"temperature": t[0], "pitch_temperature": t[-1], "spread": s[0], "pitch_spread": s[-1]}


def heldout(args, tts, st: float) -> dict:
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    res = {}

    def measure(data: ProsodyData, samples) -> list[dict]:
        rows = []
        for i, (it, u) in enumerate(zip(data.items, data.utts)):
            ids = it["ids"]
            v = voiced_tokens(torch.from_numpy(ids)).numpy()
            if samples is None:
                rows += sentence_metrics(u["norm_text"], ids, it["dur"], it["pcont"] * st, v, args.break_frames)
            else:
                for k in range(samples[i]["frames"].shape[0]):
                    rows += sentence_metrics(u["norm_text"], ids, samples[i]["frames"][k], samples[i]["pcont"][k] * st,
                                             v, args.break_frames)
        return rows

    if args.reference:
        ref = ProsodyData(cache, (args.reference,), None if args.speaker is None else [args.speaker])
        res[f"recordings ({args.reference})"] = summarize(measure(ref, None))
    data = ProsodyData(cache, ("val", "dev"), None if args.speaker is None else [args.speaker])
    res["recordings"] = summarize(measure(data, None))
    for spec in args.systems:
        name, path, kw = parse_system(spec)
        if path is None:
            smp = sample_split(None, tts, data, [0], 1.0, args.device)
            for s, it in zip(smp, data.items):  # regressors: their own (continuous) pitch prediction
                s["pcont"] = it["pitch_det"][None]
        else:
            pred = ProsodyPredictor.load(path, args.device, tts=tts)
            smp = sample_split(pred, tts, data, list(range(args.seeds)), kw["temperature"], args.device,
                               apply_scales=True, spread=kw["spread"], pitch_temperature=kw["pitch_temperature"],
                               pitch_spread=kw["pitch_spread"])
        res[name] = summarize(measure(data, smp))
        print(name, json.dumps(res[name]), flush=True)
    return res


@torch.no_grad()
def sample_sentence(tts, pred, kw, text: str, spk: int, seed: int, device):
    """Frames and token pitch (normalised; continuous for samplers) of one sentence as ``Synthesizer`` draws them
    (prosody first from the generator seeded with ``seed``)."""
    ids = torch.tensor([text_to_ids(text, normalized=True)], device=device)
    n = torch.tensor([ids.shape[1]], device=device)
    s = torch.tensor([spk], device=device)
    if pred is None:
        h, _, logw, x_mask = tts.encoder(ids, n, s)
        tempo = tts.duration_scales.get(spk, tts.duration_scale)
        frames = torch.ceil(torch.exp(logw[:, 0]) * tempo)
        se = tts.encoder.spk(s)[:, :, None].expand(-1, -1, h.shape[-1])
        pitch = tts.pitch_predictor(torch.cat([h, se], 1), x_mask)[:, 0]
        return ids[0].cpu().numpy(), frames[0].cpu().numpy(), pitch[0].cpu().numpy(), None
    g = torch.Generator(device=device).manual_seed(seed)
    h, _, logw, x_mask = tts.encoder(ids, n, s)
    cond, base = pred.condition(tts, h, x_mask, s, logw, pred.stats, pred.word_tokens(ids, n), pred.sent_tokens(ids, n))
    y, vlogit = pred.sample(cond, base, x_mask, kw["temperature"], generator=g, spread=kw["spread"],
                            pitch_temperature=kw["pitch_temperature"], pitch_spread=kw["pitch_spread"])
    frames, pitch = pred.frames_and_pitch(y, vlogit, x_mask, pred.duration_scales.get(spk, 1.0))
    _, pc = pred.stats.denorm(y)
    return ids[0].cpu().numpy(), frames[0].cpu().numpy(), pc[0].cpu().numpy(), pitch[0, 0].cpu().numpy()


def audio_metrics(wav: np.ndarray, text: str, ids: np.ndarray, frames: np.ndarray) -> dict:
    """Harvest-F0 measures of one synthesised sentence (word spans from the frames it was generated with)."""
    from drifting_tts.prosody import FRAME_MS, MIN_PAUSE_S, f0_contour, frame_level_db, pauses, silent_frames

    f0 = f0_contour(wav)
    sil = silent_frames(frame_level_db(wav))
    n = min(len(f0), len(sil))
    f0, sil = f0[:n], sil[:n]
    v = (f0 > 0) & ~sil
    if v.sum() < 10:
        return {}
    st = 12 * np.log2(np.where(v, f0, 1.0) / np.median(f0[v]))
    out = {"audio_final": float(np.median(st[np.flatnonzero(v)[-25:]]))}
    # word spans in 10 ms frames from the token frames
    edges = np.concatenate([[0], np.cumsum(frames)]) / MEL_FPS / (FRAME_MS / 1000)
    words = text.split(" ")
    word, spans = word_spans(ids)
    wv = np.full(len(words), np.nan)
    for w, (a, b) in enumerate(spans):
        if a < 0:
            continue
        lo, hi = int(edges[a]), int(math.ceil(edges[b + 1]))
        sel = v[lo:hi]
        if sel.sum() >= 2:
            wv[w] = st[lo:hi][sel].mean()
    mean = st[v].mean()
    out["audio_final_word"] = float(wv[-1] - mean)
    feats = sentence_features(text)
    hosts = np.flatnonzero(feats[:, COL["mi_host"]])
    if sentence_type(words) == "polar_q" and len(hosts):
        out["audio_pre_mi"] = float(wv[hosts[-1]] - mean)
        out["audio_fall"] = float(wv[-1] - wv[hosts[-1]])
    whs = np.flatnonzero(feats[:, COL["wh_word"]])
    if sentence_type(words) == "wh_q" and len(whs):
        out["audio_wh_peak"] = float(wv[whs[0]] - mean)
    runs, _ = pauses(sil, int(MIN_PAUSE_S * 1000 / FRAME_MS))
    out["audio_pauses"] = len(runs)
    # pauses not at a comma: the word boundary nearest to the pause's centre has none (or it lies inside a word)
    bounds = [(edges[spans[k][1] + 1], edges[spans[k + 1][0]], words[k].endswith(","))
              for k in range(len(spans) - 1) if spans[k][1] >= 0 and spans[k + 1][0] >= 0]
    nc = 0
    for a, b in runs:
        c = (a + b) / 2
        if bounds:
            dist = [0.0 if lo <= c <= hi else min(abs(c - lo), abs(c - hi)) for lo, hi, _ in bounds]
            nc += not bounds[int(np.argmin(dist))][2]
    out["audio_pauses_nc"] = nc
    out["word_audio"] = [None if np.isnan(x) else float(x - mean) for x in wv]
    return out


def diagnostic(args, tts, st: float) -> dict:
    items = [json.loads(line) for line in Path(args.texts).read_text().splitlines() if line.strip()]
    synths = {}
    res, per_item = {}, []
    for spec in args.systems:
        name, path, kw = parse_system(spec)
        pred = None if path is None else ProsodyPredictor.load(path, args.device, tts=tts)
        rows = []
        if args.audio:
            from drifting_tts.synthesize import Synthesizer

            synths[name] = Synthesizer(args.model, args.device, vocoder=args.vocoder, prosody=path,
                                       prosody_temperature=None if kw is None else kw["temperature"],
                                       prosody_spread=1.0 if kw is None else kw["spread"],
                                       prosody_pitch_temperature=None if kw is None else kw["pitch_temperature"],
                                       prosody_pitch_spread=None if kw is None else kw["pitch_spread"])
        for spk in args.speakers:
            for seed in range(args.seeds):
                for k, it in enumerate(items):
                    text = split_sentences(normalize(it["text"]))[0]
                    ids, frames, pc, _ = sample_sentence(tts, pred, kw, text, spk, seed * 1000 + k, args.device)
                    v = voiced_tokens(torch.from_numpy(ids)).numpy()
                    ms = sentence_metrics(text, ids, frames, pc * st, v, args.break_frames)
                    if not ms:
                        continue
                    m = ms[0]
                    m["type"] = it["type"]
                    if args.audio:
                        wav, _ = synths[name](it["text"], speaker=spk, cfg_scale=2.0, temperature=0.3,
                                              seed=seed * 1000 + k)
                        m.update(audio_metrics(wav.numpy(), text, ids, frames))
                    rows.append(m)
                    per_item.append({"system": name, "speaker": spk, "seed": seed, "id": it["id"],
                                     "group": it.get("group"), **{a: (float(b) if isinstance(b, (float, np.floating))
                                                                      else b) for a, b in m.items()}})
        res[name] = {str(spk): summarize([r for r, p in zip(rows, per_item[-len(rows):]) if p["speaker"] == spk])
                     for spk in args.speakers}
        res[name]["all"] = summarize(rows)
        print(name, json.dumps(res[name]["all"]), flush=True)
        synths.pop(name, None)
        torch.cuda.empty_cache()
    Path(args.out).with_suffix(".items.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in per_item) + "\n")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="runs/release/drifting_tts_v3.1.pt")
    p.add_argument("--systems", nargs="*", default=[])
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--cache", default="runs/pm_cache/targets_v31.pt")
    p.add_argument("--reference", default=None, help="heldout: also the recordings of this split (e.g. train)")
    p.add_argument("--speaker", type=int, default=None, help="heldout: one speaker only (default: all)")
    p.add_argument("--texts", default=None)
    p.add_argument("--speakers", type=int, nargs="+", default=[722])
    p.add_argument("--audio", action="store_true")
    p.add_argument("--vocoder", default="vocos-ft")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--break-frames", type=int, default=16,
                   help="boundary frames that count as a phrase break (16 = 0.17 s: on the studio `val` recordings "
                        "this gives 1.6 breaks per utterance, against 1.39 pauses of >= 0.1 s in their audio)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    tts = load_frozen_tts(args.model, args.device)
    st = float(tts.lf0_stats[1]) * 12 / math.log(2)  # normalised log-F0 -> semitones
    res = heldout(args, tts, st) if args.heldout else diagnostic(args, tts, st)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
