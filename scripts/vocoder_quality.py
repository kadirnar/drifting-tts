"""Vocoder quality beyond MOS: the Freya-100 judges plus pitch and periodicity diagnostics, and copy-synthesis of
held-out recordings (docs/VOCODERS.md, "Training Vocos further").

* ``freya``: the v3.1 mels of the first ``--num`` Freya-TR-Eval sentences (``compare_vocoders.py``'s cache: seed =
  sentence index, ``studio`` voice, T = 0.3, alpha = 2), vocoded and joined with 0.15 s pauses. Judges: WER / CER
  (Whisper large-v3, 8 kHz band-matched), UTMOSv2, DNSMOS P.835 OVRL and P.808 (``compare_vocoders.score``). Pitch:
  the F0 micro-variation of docs/EXPERIMENTS.md §5 and the mean periodicity of voiced frames, plus the F0 and
  periodicity of each output against ``--ref`` (default ``bigvgan-v2-ft``) on the same mels.
* ``copy``: recorded mel -> vocoder against the recording, on the first ``--copy-num`` utterances of the ``val``
  split (``val``: all speakers, the set of ``resynthesis_benchmark.py``) or of speaker 722 in it (``studio``):
  log-mel L1 (the BigVGAN front end), multi-resolution log-STFT L1, UTMOSv2, DNSMOS OVRL, and against the
  recording's WORLD harvest F0: voicing decision error (VDE), gross pitch error (GPE, > 20% off), the RMS of the
  other frames in cents, periodicity RMSE and bias, and the F0 micro-variation of output and recording.

Pitch: WORLD harvest, 10 ms frames, 60-500 Hz. Periodicity: 1 - the minimum of YIN's cumulative mean normalised
difference over 2-16.7 ms lags (a 32 ms window per 10 ms frame), in [0, 1]; frames within 40 dB of an utterance's
loudest count. F0 extraction runs in ``--workers`` processes (CPU). Results accumulate in ``<out>/results.json``;
``--table`` prints markdown tables.

    python scripts/vocoder_quality.py --vocoders vocos-ft bigvgan-v2-ft --extra p0=runs/voc_p0/vocos_ft.pt \\
        --mels runs/compare_vocoders/mels_495_studio_T0.3_cfg2.pt --out outputs/vocoder_quality
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

SR, FRAME_MS, F0_MIN, F0_MAX = 24_000, 10.0, 60.0, 500.0
HOP = int(SR * FRAME_MS / 1000)


def harvest(x: np.ndarray) -> np.ndarray:
    """WORLD harvest F0 (Hz, 0 = unvoiced) every 10 ms (the settings of the prosody diagnosis, EXPERIMENTS.md §5)."""
    import pyworld

    f0, _ = pyworld.harvest(np.ascontiguousarray(x, dtype=np.float64), SR, f0_floor=F0_MIN, f0_ceil=F0_MAX,
                            frame_period=FRAME_MS)
    return f0.astype(np.float32)


def prosody_stats(f0: np.ndarray) -> dict:
    """EXPERIMENTS.md §5: semitones about the median over the voiced frames, ``micro``: mean |F0 - its 5-frame moving
    average| (over the concatenated voiced frames), ``movement``: mean |change| between voiced neighbours."""
    v = f0 > 0
    if v.sum() < 10:
        return {}
    st = 12 * np.log2(f0[v] / np.median(f0[v]))
    d = np.abs(np.diff(12 * np.log2(np.where(v, f0, 1))))[v[1:] & v[:-1]]
    sm = np.convolve(st, np.ones(5) / 5, mode="same")
    return {"f0_std": float(st.std()), "movement": float(d.mean()), "micro": float(np.abs(st - sm)[2:-2].mean()),
            "voiced": float(v.mean())}


def periodicity(wav: torch.Tensor, win: int = 768, fmin: float = F0_MIN, fmax: float = F0_MAX) -> tuple[np.ndarray,
                                                                                                         np.ndarray]:
    """Per 10 ms frame (centred on ``k * 240``, harvest's grid): YIN periodicity ``1 - min_tau d'(tau)`` over lags
    of ``fmax`` .. ``fmin``, and the frame level in dB. ``wav``: ``[samples]`` at 24 kHz."""
    lo, hi = int(SR / fmax), int(SR / fmin)
    length = win + hi
    x = torch.nn.functional.pad(wav.float()[None, None], (win // 2, win // 2 + hi))[0, 0]
    frames = x.unfold(0, length, HOP)  # frame k: window [k * hop - win / 2, k * hop + win / 2) and its lags
    a = frames[:, :win]
    n = 1 << (length + win - 1).bit_length()
    r = torch.fft.irfft(torch.fft.rfft(a, n).conj() * torch.fft.rfft(frames, n), n)[:, : hi + 1]
    c = torch.nn.functional.pad(frames.square().cumsum(1), (1, 0))
    e0 = c[:, win]
    e_tau = c[:, win: win + hi + 1] - c[:, : hi + 1]
    d = (e0[:, None] + e_tau - 2 * r).clamp_min(0)[:, 1:]  # tau = 1 .. hi
    tau = torch.arange(1, hi + 1, device=d.device, dtype=d.dtype)
    cmnd = d * tau / d.cumsum(1).clamp_min(1e-12)
    p = (1 - cmnd[:, lo - 1:].min(1).values).clamp(0, 1)
    level = 10 * torch.log10(e0 / win + 1e-10)
    return p.cpu().numpy(), level.cpu().numpy()


def f0_errors(f0: np.ndarray, ref: np.ndarray) -> dict:
    """Against ``ref``: voicing decision error, gross pitch error (> 20% off, frames voiced in both) and the RMS in
    cents of the other frames voiced in both."""
    n = min(len(f0), len(ref))
    f0, ref = f0[:n], ref[:n]
    vo, vr = f0 > 0, ref > 0
    both = vo & vr
    ratio = f0[both] / ref[both]
    gross = np.abs(ratio - 1) > 0.2
    cents = 1200 * np.log2(ratio[~gross])
    return {"vde": float((vo != vr).mean()), "gpe": float(gross.mean()) if both.any() else 0.0,
            "cents_sq": float((cents**2).sum()), "cents_n": int(len(cents))}


def periodicity_errors(p: np.ndarray, p_ref: np.ndarray, level_ref: np.ndarray, voiced_ref: np.ndarray) -> dict:
    """Periodicity RMSE on frames within 40 dB of the reference's loudest, and the mean difference on voiced ones."""
    n = min(len(p), len(p_ref), len(voiced_ref))
    p, p_ref, level_ref, voiced_ref = p[:n], p_ref[:n], level_ref[:n], voiced_ref[:n]
    loud = level_ref > level_ref.max() - 40
    v = loud & voiced_ref
    return {"per_sq": float(((p - p_ref)[loud] ** 2).sum()), "per_n": int(loud.sum()),
            "per_bias_sum": float((p - p_ref)[v].sum()), "per_v_n": int(v.sum())}


class LogStft(torch.nn.Module):
    """Multi-resolution log-magnitude STFT L1 (n_fft 512 / 1024 / 2048, hop n_fft / 4) and the BigVGAN log-mel L1."""

    def __init__(self):
        from drifting_tts.audio import BigVGANLogMel

        super().__init__()
        self.mel = BigVGANLogMel()

    def forward(self, y: torch.Tensor, ref: torch.Tensor) -> dict:
        n = min(y.shape[-1], ref.shape[-1])
        y, ref = y[..., :n].float(), ref[..., :n].float()
        out = {"mel_l1": float((self.mel(y) - self.mel(ref)).abs().mean())}
        dist = []
        for fft in (512, 1024, 2048):
            w = torch.hann_window(fft, device=y.device)
            s = [torch.stft(x, fft, fft // 4, fft, w, return_complex=True).abs().clamp_min(1e-5).log()
                 for x in (y, ref)]
            dist.append(float((s[0] - s[1]).abs().mean()))
        out["stft_l1"] = float(np.mean(dist))
        return out


def mean_of(rows: list[dict], key: str) -> float:
    v = [r[key] for r in rows if key in r]
    return float(np.mean(v)) if v else float("nan")


def pool_f0(pool, wavs: list[torch.Tensor]) -> list[np.ndarray]:
    return list(pool.map(harvest, [w.double().numpy() for w in wavs], chunksize=4))


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def pitch_summary(f0s: list[np.ndarray], pers: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """F0 micro-variation and the mean periodicity of voiced, non-silent frames."""
    stats = [prosody_stats(f) for f in f0s]
    voiced = []
    for f, (p, level) in zip(f0s, pers):
        k = min(len(f), len(p))
        voiced.append(p[:k][(f[:k] > 0) & (level[:k] > level.max() - 40)])
    return {"micro": mean_of(stats, "micro"), "movement": mean_of(stats, "movement"),
            "f0_std": mean_of(stats, "f0_std"), "voiced_frac": mean_of(stats, "voiced"),
            "periodicity_voiced": float(np.concatenate(voiced).mean())}


def against(f0s, pers, ref_f0s, ref_pers) -> dict:
    """Corpus-level F0 and periodicity errors of outputs against references (same frames)."""
    fe = [f0_errors(f, r) for f, r in zip(f0s, ref_f0s)]
    pe = [periodicity_errors(p[0], rp[0], rp[1], rf > 0) for p, rp, rf in zip(pers, ref_pers, ref_f0s)]
    n_c, n_p, n_v = (sum(x[k] for x in xs) for xs, k in ((fe, "cents_n"), (pe, "per_n"), (pe, "per_v_n")))
    return {"vde": mean_of(fe, "vde"), "gpe": mean_of(fe, "gpe"),
            "cents": float(np.sqrt(sum(x["cents_sq"] for x in fe) / max(n_c, 1))),
            "per_rmse": float(np.sqrt(sum(x["per_sq"] for x in pe) / max(n_p, 1))),
            "per_bias": float(sum(x["per_bias_sum"] for x in pe) / max(n_v, 1))}


def run_freya(args, specs: dict, res: dict, pool, out: Path) -> None:
    from compare_vocoders import cached_mels, score, vocode

    from drifting_tts.benchmark import FREYA, load_texts
    from drifting_tts.evaluate import _plain
    from drifting_tts.vocoder import load_vocoder

    texts = [t["text"] for t in load_texts(FREYA)][: args.num]
    if args.mels:
        mels = torch.load(args.mels)[: args.num]
    else:
        mels = cached_mels(args, texts, out / f"mels_{len(texts)}_{args.speaker}_T{args.temperature:g}_"
                           f"cfg{args.cfg:g}.pt")
    refs = [_plain(t) for t in texts]
    table = res.setdefault("freya", {})
    names = list(dict.fromkeys([args.ref, *specs]))
    pitch_cache: dict[str, tuple] = {}
    judges = dnsmos = None
    for name in names:
        spec = specs.get(name, name)
        cache = out / "pitch" / f"freya{args.num}_{name}.pt"
        if name in table and not args.force and cache.exists():
            pitch_cache[name] = torch.load(cache, weights_only=False)
            continue
        t0 = time.perf_counter()
        voc = load_vocoder(spec, args.device, backend="bigvgan")
        wavs = vocode(voc, mels)
        del voc
        free()
        if args.save_wavs:
            import soundfile as sf

            (out / "wav" / name).mkdir(parents=True, exist_ok=True)
            for i, w in enumerate(wavs[: args.save_wavs]):
                sf.write(out / "wav" / name / f"{i:04d}.wav", w.numpy(), SR)
        f0s = pool_f0(pool, wavs)
        pers = [periodicity(w.to(args.device)) for w in wavs]
        pitch_cache[name] = (f0s, pers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save((f0s, pers), cache)
        if judges is None:
            from drifting_tts.judges import load_judges
            from drifting_tts.score import DnsMos

            judges, dnsmos = load_judges("large-v3", None, "utmosv2", args.device), DnsMos(args.device)
        q = score(wavs, refs, judges, dnsmos, 8000, out / f"freya{args.num}_{name}.jsonl")
        table[name] = {"spec": spec, "wer": q["wer"], "wer_ci": q["wer_ci"], "cer": q["cer"], "utmosv2": q["mos"],
                       "dnsmos_ovrl": q["dnsmos_ovrl"], "dnsmos_p808": q["dnsmos_p808"],
                       **pitch_summary(f0s, pers)}
        print(f"freya {name}: " + json.dumps({k: round(v, 4) for k, v in table[name].items()
                                               if isinstance(v, float)}) + f" ({time.perf_counter() - t0:.0f} s)",
              flush=True)
        save(res, out)
    for name in names:  # against the reference vocoder on the same mels
        if name != args.ref:
            table[name][f"vs_{args.ref}"] = against(*pitch_cache[name], *pitch_cache[args.ref])
    save(res, out)
    del judges, dnsmos
    free()


def run_copy(args, specs: dict, res: dict, pool, out: Path, which: str) -> None:
    import torchaudio

    from drifting_tts.data import MelDataset
    from drifting_tts.judges import UTMOSv2
    from drifting_tts.score import DnsMos
    from drifting_tts.vocoder import load_vocoder

    ds = MelDataset(args.data, "val", min_frames=1, max_frames=10**9, with_audio=True)
    idx = [i for i, e in enumerate(ds.items) if which == "val" or e["spk_id"] == 722][: args.copy_num]
    items = [ds[i] for i in idx]
    rec = [it["audio"] for it in items]
    mels = [ds.stats.denormalize(it["mel"]) for it in items]
    key = f"copy_{which}"
    table = res.setdefault(key, {})
    rec_cache = out / "pitch" / f"{key}{len(idx)}_recording.pt"
    if rec_cache.exists():
        rec_f0, rec_per = torch.load(rec_cache, weights_only=False)
    else:
        rec_f0, rec_per = pool_f0(pool, rec), [periodicity(w.to(args.device)) for w in rec]
        rec_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save((rec_f0, rec_per), rec_cache)
    dist = LogStft().to(args.device)
    mos = dns = None

    def judge(wavs: list[torch.Tensor]) -> dict:
        nonlocal mos, dns
        if mos is None:
            mos, dns = UTMOSv2(args.device), DnsMos(args.device)
        w16 = [torchaudio.functional.resample(w.float(), SR, 16_000).numpy() for w in wavs]
        d = dns.score([w.clip(-1, 1) for w in w16])
        return {"utmosv2": float(np.mean([mos(w) for w in w16])), "dnsmos_ovrl": float(d[:, 2].mean())}

    if "recording" not in table or args.force:
        table["recording"] = {**judge(rec), **pitch_summary(rec_f0, rec_per)}
        save(res, out)
    for name, spec in specs.items():
        if name in table and not args.force:
            continue
        t0 = time.perf_counter()
        voc = load_vocoder(spec, args.device, backend="bigvgan")
        wavs = [voc(m[None])[0, : len(r)].cpu() for m, r in zip(mels, rec)]
        del voc
        free()
        d = [dist(w.to(args.device), r.to(args.device)) for w, r in zip(wavs, rec)]
        f0s = pool_f0(pool, wavs)
        pers = [periodicity(w.to(args.device)) for w in wavs]
        table[name] = {"spec": spec, "mel_l1": mean_of(d, "mel_l1"), "stft_l1": mean_of(d, "stft_l1"),
                       **judge(wavs), **against(f0s, pers, rec_f0, rec_per), **pitch_summary(f0s, pers)}
        print(f"{key} {name}: " + json.dumps({k: round(v, 4) for k, v in table[name].items()
                                               if isinstance(v, float)}) + f" ({time.perf_counter() - t0:.0f} s)",
              flush=True)
        save(res, out)
    del mos, dns
    free()


def save(res: dict, out: Path) -> None:
    (out / "results.json").write_text(json.dumps(res, indent=1))


def tables(res: dict, order: list[str]) -> str:
    out = ""
    if "freya" in res:
        f = res["freya"]
        ref = next((k[3:] for v in f.values() for k in v if k.startswith("vs_")), None)
        out += (f"| vocoder | WER | CER | UTMOSv2 | DNSMOS OVRL | P.808 | F0 micro-var (st) | periodicity (voiced) | "
                f"vs {ref}: VDE | GPE | cents | periodicity RMSE |\n|---" + "|---" * 11 + "|\n")
        for n in [n for n in order if n in f] + [n for n in f if n not in order]:
            r, v = f[n], f[n].get(f"vs_{ref}", {})
            out += (f"| `{n}` | {100 * r['wer']:.2f}% | {100 * r['cer']:.2f}% | {r['utmosv2']:.3f} | "
                    f"{r['dnsmos_ovrl']:.3f} | {r['dnsmos_p808']:.3f} | {r['micro']:.3f} | "
                    f"{r['periodicity_voiced']:.3f} | "
                    + (f"{100 * v['vde']:.1f}% | {100 * v['gpe']:.2f}% | {v['cents']:.1f} | {v['per_rmse']:.3f} |"
                       if v else "– | – | – | – |") + "\n")
    for key in ("copy_studio", "copy_val"):
        if key not in res:
            continue
        c = res[key]
        out += (f"\n{key}:\n\n| system | mel L1 | log-STFT L1 | UTMOSv2 | DNSMOS OVRL | VDE | GPE | cents | "
                "periodicity RMSE | periodicity bias | F0 micro-var (st) |\n|---" + "|---" * 10 + "|\n")
        for n in [n for n in ["recording", *order] if n in c] + [n for n in c if n not in order and n != "recording"]:
            r = c[n]
            if n == "recording":
                out += (f"| recording | – | – | {r['utmosv2']:.3f} | {r['dnsmos_ovrl']:.3f} | – | – | – | – | – | "
                        f"{r['micro']:.3f} |\n")
                continue
            out += (f"| `{n}` | {r['mel_l1']:.4f} | {r['stft_l1']:.4f} | {r['utmosv2']:.3f} | {r['dnsmos_ovrl']:.3f} | "
                    f"{100 * r['vde']:.1f}% | {100 * r['gpe']:.2f}% | {r['cents']:.1f} | {r['per_rmse']:.4f} | "
                    f"{r['per_bias']:+.4f} | {r['micro']:.3f} |\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--vocoders", nargs="*", default=["vocos-ft"], help="registry names or checkpoint paths")
    p.add_argument("--extra", nargs="+", default=[], metavar="NAME=PATH", help="more vocoders: checkpoint paths")
    p.add_argument("--phases", nargs="+", default=["freya", "copy"], choices=["freya", "copy"])
    p.add_argument("--copy-sets", nargs="+", default=["studio"], choices=["studio", "val"])
    p.add_argument("--copy-num", type=int, default=100)
    p.add_argument("--num", type=int, default=100, help="Freya sentences")
    p.add_argument("--ref", default="bigvgan-v2-ft", help="reference vocoder for the Freya F0 / periodicity")
    p.add_argument("--mels", default=None, help="cached Freya mels of compare_vocoders.py (the first --num are used)")
    p.add_argument("--model", default=None, help="TTS checkpoint (only to generate the mels without --mels)")
    p.add_argument("--speaker", default="studio")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--data", default="data/train", help="prepared data with audio (copy-synthesis)")
    p.add_argument("--workers", type=int, default=3, help="processes for WORLD harvest")
    p.add_argument("--save-wavs", type=int, default=0, help="Freya outputs to keep per vocoder")
    p.add_argument("--force", action="store_true", help="recompute rows already in results.json")
    p.add_argument("--table", action="store_true", help="only print the tables")
    p.add_argument("--out", default="outputs/vocoder_quality")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    specs = {n: n for n in args.vocoders} | dict(e.split("=", 1) for e in args.extra)
    res["order"] = list(dict.fromkeys(res.get("order", []) + list(specs)))
    if not args.table:
        import multiprocessing as mp

        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
            if "freya" in args.phases:
                run_freya(args, specs, res, pool, out)
            if "copy" in args.phases:
                for which in args.copy_sets:
                    run_copy(args, specs, res, pool, out, which)
        save(res, out)
    print(tables(res, res["order"]))


if __name__ == "__main__":
    main()
