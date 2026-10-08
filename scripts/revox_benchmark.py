"""Revox Vocoder 1.0 on held-out recordings: mel conversion, F0 sources, copy-synthesis and speed (docs/VOCODERS.md).

Revox Vocoder 1.0 is by Minori Live (https://huggingface.co/minori-live/revox-vocoder-1), CC BY-NC-SA 4.0:
non-commercial use only. Its weights are downloaded at runtime and never stored here; see
:class:`drifting_tts.vocoder.Revox`.

* ``convert``: the Revox mel converted from the recording's BigVGAN mel against the Revox mel of the recording
  upsampled to 48 kHz, in dB below 11.5 kHz, with linear- and log-domain time interpolation.
* ``f0``: WORLD dio / harvest on the Griffin-Lim and BigVGAN-v2-ft audio of the mel, as the ``revox`` vocoder runs it
  (``revox_pitch``), against the recording on the frames where harvest and dio agree: voicing decision error (VDE),
  gross pitch error (GPE, > 20%), fine pitch error (cents RMS).
* ``quality``: copy-synthesis rows scored by ``resynthesis_benchmark.score`` (same utterances, judges and columns),
  plus the share of energy above 12 kHz and the output's F0 against the recording's.
* ``speed``: real-time factor of each stage of the ``revox`` registry entry (GPU for the torch stages, ONNX Runtime
  and WORLD on the CPU), over the first ``--speed-utts`` utterances.

    python scripts/revox_benchmark.py --data data/train --out outputs/revox
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from resynthesis_benchmark import pct, score

from drifting_tts.audio import SAMPLE_RATE, BigVGANLogMel, world_f0
from drifting_tts.data import MelDataset
from drifting_tts.evaluate import _plain, select
from drifting_tts.vocoder import (
    REVOX_HOP,
    REVOX_RATE,
    Revox,
    load_vocoder,
    resample_sharp,
    revox_frames,
    revox_pitch,
    to_revox_frames,
)

DB = 20 / math.log(10)  # natural-log magnitude -> dB
PERIOD = 1000 * REVOX_HOP / REVOX_RATE  # 10 ms


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def f0_job(job: tuple[np.ndarray, int, str, int, bool]) -> np.ndarray:
    wav, sr, method, frames, source = job
    if source:
        return revox_pitch(torch.from_numpy(wav), sr, frames, method)[0].numpy()
    f0 = world_f0(wav, sr, PERIOD, method).astype(np.float32)[:frames]
    return np.pad(f0, (0, frames - len(f0)))


def f0_all(pool, wavs: list[torch.Tensor], frames: list[int], method: str, source: bool = False,
           sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """WORLD F0 at Revox's frame centres (``frames`` each, 0: unvoiced): as the ``revox`` vocoder computes it
    (``source``), else plain WORLD."""
    return list(pool.map(f0_job, [(w.double().cpu().numpy(), sr, method, k, source) for w, k in zip(wavs, frames)]))


def f0_reference(harvest: list[np.ndarray], dio: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reference F0 and its frames: where harvest and dio on the recording agree (both unvoiced, or both voiced
    within 10%: harvest's F0). Harvest alone extends voicing into pauses, where no F0 is defined."""
    ref, keep = [], []
    for h, d in zip(harvest, dio):
        voiced = (h > 0) & (d > 0) & (np.abs(d - h) < 0.1 * h)
        ref.append(np.where(voiced, h, 0).astype(np.float32))
        keep.append(voiced | ((h == 0) & (d == 0)))
    return ref, keep


def f0_errors(ref: tuple[list[np.ndarray], list[np.ndarray]], est: list[np.ndarray]) -> dict:
    """VDE, GPE (> 20% off on frames voiced in both) and the cents RMS of the rest, on the reference's frames."""
    keep = np.concatenate(ref[1])
    r, e = np.concatenate(ref[0])[keep], np.concatenate([x[: len(y)] for x, y in zip(est, ref[0])])[keep]
    both = (r > 0) & (e > 0)
    ratio = e[both] / r[both]
    gross = np.abs(ratio - 1) > 0.2
    cents = 1200 * np.log2(ratio[~gross])
    return {"vde": float(np.mean((r > 0) != (e > 0))), "gpe": float(gross.mean()),
            "fpe_cents": float(np.sqrt(np.mean(cents**2))), "voiced_ref": float(np.mean(r > 0)),
            "voiced_est": float(np.mean(e > 0))}


def high_band(wav: torch.Tensor, sr: int, cut: float = 12_000) -> float:
    """Share of the energy above ``cut`` Hz."""
    p = torch.fft.rfft(wav.double()).abs().square()
    f = torch.fft.rfftfreq(wav.numel(), 1 / sr)
    return float(p[f > cut].sum() / p.sum().clamp_min(1e-30))


def timed(fn, timings: dict, key: str):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - t
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="prepared dataset with audio.bin (prepare --save-audio)")
    p.add_argument("--split", default="val")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--phases", nargs="+", default=["convert", "f0", "quality", "speed"],
                   choices=["convert", "f0", "quality", "speed"])
    p.add_argument("--vocoder", default="bigvgan-v2-ft", help="the BigVGAN row and F0 source")
    p.add_argument("--f0-methods", nargs="+", default=["dio", "harvest"],
                   help="WORLD methods on vocoded audio (the recording: harvest)")
    p.add_argument("--band", type=int, default=8000)
    p.add_argument("--asr", default="large-v3")
    p.add_argument("--asr-batch", type=int, default=16)
    p.add_argument("--no-judges", action="store_true", help="quality: update the F0 / band diagnostics only")
    p.add_argument("--speed-utts", type=int, default=20)
    p.add_argument("--workers", type=int, default=3, help="processes for WORLD")
    p.add_argument("--out", default="outputs/revox")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    ds = MelDataset(args.data, args.split, min_frames=1, max_frames=10**9, with_audio=True)
    idx = select(ds, args.split, args.offset, args.num)
    wavs = [ds[i]["audio"].float() for i in idx]
    refs = [_plain(ds.items[i]["norm_text"]) for i in idx]
    seconds = sum(len(w) for w in wavs) / SAMPLE_RATE
    res["setup"] = {"utterances": len(idx), "seconds": seconds, "split": args.split, "vocoder": args.vocoder,
                    "f0_methods": args.f0_methods, "asr": args.asr, "band_hz": args.band,
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "torch": torch.__version__, "threads": torch.get_num_threads()}
    dev = args.device
    front, rv = BigVGANLogMel().to(dev), Revox("none", device=dev).to(dev)
    voc = load_vocoder(args.vocoder, dev)
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))

    mels = [front(w[None].to(dev)) for w in wavs]
    frames = [revox_frames(m.shape[-1]) for m in mels]
    mags = [rv.gl.magnitude(m) for m in mels]
    conv = [rv.front.convert(g) for g in mags]
    truth = [rv.front(resample_sharp(w[None].to(dev), SAMPLE_RATE, REVOX_RATE))[..., :k] for w, k in zip(wavs, frames)]
    rec_f0 = f0_all(pool, wavs, frames, "harvest")  # Revox's input for the "recording F0" rows
    ref = f0_reference(rec_f0, f0_all(pool, wavs, frames, "dio"))
    res["f0_reference"] = {"frames_kept": float(np.concatenate(ref[1]).mean()),
                           "voiced_of_kept": float(np.mean(np.concatenate(ref[0])[np.concatenate(ref[1])] > 0))}

    def save() -> None:
        res_path.write_text(json.dumps(res, indent=1))

    if "convert" in args.phases:
        fb = rv.front.fb
        top = (fb.shape[-1] - 1 - (fb.flip(-1) > 0).float().argmax(-1)) * REVOX_RATE / 2 / (fb.shape[-1] - 1)
        low = top < 11_500  # below the upsampler's transition band

        def lin(g: torch.Tensor) -> torch.Tensor:  # 24 kHz magnitude -> linear Revox mel, same frames
            return fb[:, : g.shape[1]] @ (2 * g)

        def log(x: torch.Tensor) -> torch.Tensor:
            return x.clamp_min(1e-5).log()

        def at_bigvgan_frames(w: torch.Tensor, t: int) -> torch.Tensor:  # 48 kHz frames centred at 512 i + 256
            y = F.pad(resample_sharp(w[None].to(dev), SAMPLE_RATE, REVOX_RATE)[:, None], (768, 1024), mode="reflect")
            spec = torch.stft(y[:, 0], 2048, 512, 2048, rv.front.window, center=False, return_complex=True)
            return log(fb @ spec.abs())[..., :t]

        exact = [rv.gl.stft(w[None].to(dev))[..., : m.shape[-1]].abs() for w, m in zip(wavs, mels)]
        variants = {
            "NNLS + linear interpolation (used)": (conv, truth),
            "NNLS + log interpolation": ([to_revox_frames(log(lin(g))) for g in mags], truth),
            "NNLS alone (BigVGAN frames)": ([log(lin(g)) for g in mags],
                                            [at_bigvgan_frames(w, m.shape[-1]) for w, m in zip(wavs, mels)]),
            "exact magnitude + linear interpolation": ([log(to_revox_frames(lin(g))) for g in exact], truth),
            "exact magnitude + log interpolation": ([to_revox_frames(log(lin(g))) for g in exact], truth),
        }
        stats = {}
        for name, (mats, refs_) in variants.items():
            err, act, bias = [], [], []
            for c, t in zip(mats, refs_):
                c, t = c[0, low, 2:-2], t[0, low, 2:-2]  # the edge frames see different padding
                d = DB * (c - t)
                active = t > t.max() - 60 / DB
                err.append(d.abs().flatten())
                act.append(d[active].abs())
                bias.append(d[active])
            stats[name] = {"mae_db": float(torch.cat(err).mean()), "mae_db_active": float(torch.cat(act).mean()),
                           "bias_db_active": float(torch.cat(bias).mean()),
                           "p95_db_active": float(torch.cat(act).quantile(0.95))}
            print("convert", name, json.dumps(stats[name]), flush=True)
        res["convert"] = {"variants": stats, "bands": int(low.sum())}
        save()

    gl_wavs = [rv.gl.reconstruct(g)[0] for g in mags]
    voc_wavs = [voc(m)[0] for m in mels]
    if "f0" in args.phases:
        res["f0"] = {}
        for src, ws in (("griffin-lim", gl_wavs), (args.vocoder, voc_wavs)):
            for method in ("dio", "harvest"):
                res["f0"][f"{src}/{method}"] = e = f0_errors(ref, f0_all(pool, ws, frames, method, True))
                print("f0", src, method, json.dumps(e), flush=True)
        save()

    if "quality" in args.phases:
        def run(mel, f0, voiced=None, valid=None) -> torch.Tensor:
            f0 = torch.from_numpy(f0)[None]
            voiced = f0 > 0 if voiced is None else voiced
            valid = torch.ones_like(voiced) if valid is None else valid
            return rv.generate(mel, f0, voiced, valid)[0].cpu()

        systems = {"revox: recording mel, recording F0": [run(t, f) for t, f in zip(truth, rec_f0)],
                   "revox: converted mel, recording F0": [run(c, f) for c, f in zip(conv, rec_f0)]}
        for src, ws in (("griffin-lim", gl_wavs), (args.vocoder, voc_wavs)):
            for method in args.f0_methods:
                systems[f"revox: converted mel, {src} F0 ({method})"] = [
                    run(c, f) for c, f in zip(conv, f0_all(pool, ws, frames, method, True))]
        off = [torch.zeros(1, k, dtype=torch.bool) for k in frames]
        systems["revox: converted mel, no F0"] = [run(c, np.zeros(k, np.float32), o, o)
                                                  for c, k, o in zip(conv, frames, off)]
        audio = {"recording": ([w.numpy() for w in wavs], SAMPLE_RATE)}
        audio |= {k: ([y[: 2 * len(w)].numpy() for y, w in zip(v, wavs)], REVOX_RATE) for k, v in systems.items()}
        audio[args.vocoder] = ([y[: len(w)].cpu().numpy() for y, w in zip(voc_wavs, wavs)], SAMPLE_RATE)
        diag = {}
        for s, (ws, sr) in audio.items():
            t = [torch.from_numpy(w) for w in ws]
            d24 = t if sr == SAMPLE_RATE else [resample_sharp(w, sr, SAMPLE_RATE) for w in t]
            diag[s] = {"energy_above_12k": float(np.mean([high_band(w, sr) for w in t])),
                       "f0_vs_recording": f0_errors(ref, f0_all(pool, d24, frames, "harvest")), "rate": sr}
            print(s, json.dumps(diag[s]), flush=True)
        q = {} if args.no_judges else score(audio, refs, argparse.Namespace(**vars(args)))
        res["quality"] = {s: {**res.get("quality", {}).get(s, {}), **q.get(s, {}), **diag[s]} for s in audio}
        save()

    if "speed" in args.phases:
        n = args.speed_utts
        sub = sum(len(w) for w in wavs[:n]) / SAMPLE_RATE
        t: dict[str, float] = {}
        regs = {s: load_vocoder(s, dev) for s in ("revox", "revox:none")}
        for m in mels[:2]:  # warm-up
            for r in regs.values():
                r(m)
            voc(m)
        for m in mels[:n]:
            g = timed(lambda m=m: rv.gl.magnitude(m), t, "nnls")
            c = timed(lambda g=g: rv.front.convert(g), t, "mel conversion")
            y = timed(lambda g=g: rv.gl.reconstruct(g), t, "griffin-lim audio")
            timed(lambda y=y, c=c: revox_pitch(y[0], SAMPLE_RATE, c.shape[-1], "dio"), t, "dio")
            timed(lambda y=y, c=c: revox_pitch(y[0], SAMPLE_RATE, c.shape[-1], "harvest"), t, "harvest")
            timed(lambda m=m: voc(m), t, args.vocoder)
            k = c.shape[-1]
            f0, vv = torch.full((1, k), 120.0), torch.ones(1, k, dtype=torch.bool)
            feeds = {"mel": c.cpu().numpy(), "f0_hz": f0.numpy(), "voiced": vv.numpy(), "pitch_valid": vv.numpy(),
                     "noise": np.random.default_rng(0).standard_normal((1, REVOX_HOP * k), dtype=np.float32)}
            real, imag = timed(lambda f=feeds: rv.session.run(None, f), t, "onnx runtime")
            spec = torch.complex(torch.from_numpy(real), torch.from_numpy(imag)).to(dev)
            y48 = timed(lambda s=spec, k=k: torch.istft(s, 2048, REVOX_HOP, 2048, rv.front.window, center=True,
                                                         length=REVOX_HOP * k), t, "istft")
            timed(lambda y=y48: resample_sharp(y, REVOX_RATE, SAMPLE_RATE), t, "resample 48 -> 24 kHz")
            for name, r in regs.items():
                timed(lambda m=m, r=r: r(m), t, f"{name} (whole call)")
        res["speed"] = {"seconds": sub, "providers": rv.session.get_providers(), "threads": torch.get_num_threads(),
                        "rtf": {k: v / sub for k, v in t.items()}}
        print("speed", json.dumps(res["speed"]), flush=True)
        save()
    pool.shutdown()
    print(tables(res))
    (out / "results.md").write_text(tables(res) + "\n")


def tables(res: dict) -> str:
    out = []
    if "convert" in res:
        out += ["| conversion | mean abs. error | within 60 dB of the peak | its bias | its 95th percentile |",
                "|---|---|---|---|---|"]
        for k, c in res["convert"]["variants"].items():
            out.append(f"| {k} | {c['mae_db']:.2f} dB | {c['mae_db_active']:.2f} dB | {c['bias_db_active']:+.2f} dB | "
                       f"{c['p95_db_active']:.2f} dB |")
        out.append("")
    if "f0_reference" in res:
        r = res["f0_reference"]
        out += [f"F0 reference: the {100 * r['frames_kept']:.0f}% of frames where harvest and dio agree on the "
                f"recording ({100 * r['voiced_of_kept']:.0f}% of them voiced).", ""]
    if "f0" in res:
        out += ["| F0 source | VDE | GPE | fine error |", "|---|---|---|---|"]
        for k, e in res["f0"].items():
            out.append(f"| {k} | {100 * e['vde']:.1f}% | {100 * e['gpe']:.2f}% | {e['fpe_cents']:.0f} cents |")
        out.append("")
    if "quality" in res and all("band" in q for q in res["quality"].values()):
        out += ["| system | WER 8 kHz [95% CI] | CER 8 kHz | UTMOSv2 | DNSMOS OVRL | speaker sim. | bandwidth | "
                "energy > 12 kHz | output F0 vs recording: VDE / GPE / cents |", "|" + "---|" * 9]
        for s, q in res["quality"].items():
            sim = f"{q['speaker_sim']['mean']:.3f}" if "speaker_sim" in q else "–"
            f = q["f0_vs_recording"]
            out.append(f"| {s} | {pct(q['band'], 'wer')} | {pct(q['band'], 'cer')} | {q['utmosv2']['mean']:.3f} | "
                       f"{q['dnsmos_ovrl']['mean']:.3f} | {sim} | {q['bandwidth_hz'] / 1000:.1f} kHz | "
                       f"{q['energy_above_12k']:.1e} | {100 * f['vde']:.1f}% / {100 * f['gpe']:.2f}% / "
                       f"{f['fpe_cents']:.0f} |")
        out.append("")
    if "speed" in res:
        out += ["| stage | RTF |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in res["speed"]["rtf"].items()]
    return "\n".join(out)


if __name__ == "__main__":
    main()
