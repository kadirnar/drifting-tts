"""Resynthesis benchmark of the audio backends (:mod:`drifting_tts.latents`): encode -> decode held-out recordings and
score the result against the recordings. This is the quality ceiling of a TTS model that generates these frames.

* intelligibility: Whisper large-v3 (``score.AsrScorer``, Turkish, beam 5) WER / CER against the transcripts, on audio
  band-matched to 8 kHz (``--band``, as in ``benchmark``) and on the full band; corpus-level, 95% bootstrap intervals;
* naturalness: UTMOSv2 and DNSMOS P.835 (OVRL / SIG / BAK), at 16 kHz;
* speaker similarity: cosine of WavLM-Large + ECAPA-TDNN embeddings with the original recording;
* speed (GPU, batch 1, after warm-up): encode and decode real-time factors over the whole utterances, and the decode
  time of a first streaming window (``--first-ms`` of audio plus the right context it needs);
* streaming: SNR of window-by-window decoding (``stream_decode``, fp32 without TF32) against decoding whole, as a
  function of the context; the smallest context above ``--match-db`` is the one used for the first window;
* frames: per-channel mean / std ranges, frames per second, the VAEs' posterior std, the delay of the output.

Rows: the recording (reference), BigVGAN-v2 on the recording's mel (``--vocoder``: fine-tuned weights) and each VAE.
The prepared recordings are 24 kHz, so the 44.1 / 48 kHz encoders see audio band-limited to 12 kHz. Only aggregate
numbers are written to ``--out`` (no audio unless ``--save-wavs``).

    python scripts/resynthesis_benchmark.py --data data/train --vocoder runs/vocoder/bigvgan_ft.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.data import MelDataset
from drifting_tts.evaluate import _plain, select, summarize
from drifting_tts.latents import AudioBackend, LatentStats, load_backend, resample, stream_decode
from drifting_tts.metrics import bootstrap_ci, error_counts

SYSTEMS = ("recording", "bigvgan", "dacvae", "voxcpm2", "voxcpm1.5")
CONTEXT_MS = (0, 40, 80, 160, 240, 320, 480, 640, 960, 1280)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t


@contextlib.contextmanager
def no_tf32():
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def lag(ref: torch.Tensor, out: torch.Tensor, max_lag: int) -> int:
    """Lag (samples) that maximises the cross-correlation of ``out`` with ``ref``; positive: ``out`` is late."""
    n = min(len(ref), len(out))
    f = torch.fft.rfft(ref[:n].double(), 2 * n).conj() * torch.fft.rfft(out[:n].double(), 2 * n)
    cc = torch.fft.irfft(f, 2 * n)
    k = torch.cat([cc[-max_lag:], cc[: max_lag + 1]]).argmax().item()
    return int(k - max_lag)


def frames(ms: float, rate: float) -> int:
    """Frames that cover ``ms`` milliseconds at ``rate`` frames per second."""
    return math.ceil(ms * rate / 1000 - 1e-9)


def resynthesize(backend: AudioBackend, wavs: list[torch.Tensor], args) -> tuple[list[np.ndarray], dict]:
    """Encode and decode every recording; returns the decoded audio (cut to the input length) and the measurements."""
    sr_out, rate = backend.output_rate, backend.frame_rate
    warm = backend.encode(wavs[0], SAMPLE_RATE)
    for _ in range(3):
        backend.decode(warm)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    outs, latents, post, lags, t_enc, t_dec = [], [], [], [], 0.0, 0.0
    for k, w in enumerate(wavs):
        z, te = timed(lambda w=w: backend.encode(w, SAMPLE_RATE))
        y, td = timed(lambda z=z: backend.decode(z))
        assert z.shape[-1] == backend.num_frames(len(w), SAMPLE_RATE) and y.shape[-1] == z.shape[-1] * backend.hop_out
        t_enc, t_dec = t_enc + te, t_dec + td
        y = y[0, : -(-len(w) * sr_out // SAMPLE_RATE)].float().cpu()
        outs.append(y.numpy())
        latents.append(z[0].float().cpu())
        if hasattr(backend, "posterior"):
            post.append(backend.posterior(w, SAMPLE_RATE)[1][0].float().cpu())
        if k < args.stream_utts:
            lags.append(lag(resample(w, SAMPLE_RATE, sr_out), y, sr_out // 100))
    seconds = sum(len(w) for w in wavs) / SAMPLE_RATE
    stats = LatentStats.from_latents(latents)
    res = {"input_rate": backend.input_rate, "output_rate": sr_out, "frame_rate": rate, "dim": backend.dim,
           "hop_in": backend.hop_in, "hop_out": backend.hop_out, "causal": backend.causal,
           "frames_per_second": sum(z.shape[-1] for z in latents) / seconds,
           "encode_rtf": t_enc / seconds, "decode_rtf": t_dec / seconds,
           "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
           "lag_samples": lags, "channel_mean_range": [stats.mean.min().item(), stats.mean.max().item()],
           "channel_std_range": [stats.std.min().item(), stats.std.max().item()],
           "global_mean": float(torch.cat([z.flatten() for z in latents]).mean()),
           "global_std": float(torch.cat([z.flatten() for z in latents]).std()), "stats": stats.to_dict()}
    active = stats.std > 0.01 * stats.std.median()  # some VAE channels are unused: constant mean, prior std
    res["active_channels"] = int(active.sum())
    if post:
        p = torch.cat(post, -1).mean(-1)  # mean posterior std per channel
        res["posterior_std_mean"] = p[active].mean().item()
        res["posterior_std_over_channel_std"] = (p / stats.std)[active].median().item()
    return outs, {**res, **streaming(backend, latents[: args.stream_utts], args)}


def snr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    return 10 * math.log10(ref.double().pow(2).sum().item() / max((ref - est).double().pow(2).sum().item(), 1e-30))


def streaming(backend: AudioBackend, latents: list[torch.Tensor], args) -> dict:
    """Window-by-window vs whole decoding (exact fp32) per context, and the decode time of the first window."""
    rate, dev = backend.frame_rate, backend.device
    first, chunk = frames(args.first_ms, rate), frames(args.chunk_ms, rate)
    sig, err = dict.fromkeys(CONTEXT_MS, 0.0), dict.fromkeys(CONTEXT_MS, 0.0)
    with no_tf32():
        for z in latents:
            whole = backend.decode(z[None].to(dev))[0].double()
            for ms in CONTEXT_MS:
                piece = torch.cat(list(stream_decode(backend, z[None].to(dev), first, chunk, frames(ms, rate))))
                sig[ms] += whole.pow(2).sum().item()
                err[ms] += (whole - piece).pow(2).sum().item()
        exact = backend.decode(latents[0][None].to(dev))
    default = backend.decode(latents[0][None].to(dev))  # PyTorch's default precision (TF32 convolutions), as timed
    snr = {ms: 10 * math.log10(sig[ms] / max(err[ms], 1e-30)) for ms in CONTEXT_MS}
    match = next((ms for ms in CONTEXT_MS if snr[ms] > args.match_db), CONTEXT_MS[-1])
    right = 0 if backend.causal else frames(match, rate)
    z0 = latents[0][None, :, : first + right].to(dev)
    for _ in range(3):
        backend.decode(z0)
    ttfa = sorted(timed(lambda: backend.decode(z0))[1] for _ in range(args.timing_runs))[args.timing_runs // 2]
    return {"stream_snr_db": {str(ms): round(v, 1) for ms, v in snr.items()},
            "default_vs_fp32_db": snr_db(exact, default), "stream_first_frames": first, "stream_chunk_frames": chunk,
            "stream_context_ms": match, "stream_context_frames": frames(match, rate),
            "first_window_frames": first + right, "first_window_audio_ms": 1000 * first / rate,
            "first_window_decode_ms": 1000 * ttfa}


def to16k(wav: np.ndarray, sr: int, band: int = 0) -> np.ndarray:
    """ASR / judge input: 16 kHz, through ``band`` Hz first when ``band`` > 0 (``benchmark.band_match``)."""
    x = torch.from_numpy(wav).float()
    if band:
        x = resample(x, sr, band)
        sr = band
    return resample(x, sr, 16_000).numpy()


def mean_ci(v: list[float]) -> dict:
    return {"mean": float(np.mean(v)), "ci": bootstrap_ci(v)}


def score(audio: dict[str, tuple[list[np.ndarray], int]], refs: list[str], args) -> dict[str, dict]:
    """Judges one at a time (each freed before the next) over every system."""
    from drifting_tts.judges import SpeakerEmbedder, UTMOSv2
    from drifting_tts.score import AsrScorer, DnsMos, bandwidth_hz

    res: dict[str, dict] = {s: {} for s in audio}
    asr = AsrScorer(args.asr, args.device, beam_size=5, batch_size=args.asr_batch)
    for s, (wavs, sr) in audio.items():
        for tag, band in (("band", args.band), ("full", 0)):
            hyps = asr.transcribe([to16k(w, sr, band) for w in wavs])
            rows = [{"hyp": _plain(h), **error_counts(r, _plain(h))} for r, h in zip(refs, hyps)]
            res[s][tag] = {k: v for k, v in summarize(rows).items() if k.startswith(("cer", "wer"))}
        print(s, json.dumps({k: res[s][k] for k in ("band", "full")}), flush=True)
    del asr
    torch.cuda.empty_cache()
    sv = SpeakerEmbedder("wavlm-large-ecapa", args.device)
    rec, rec_sr = audio["recording"]
    orig = [sv(to16k(w, rec_sr)) for w in rec]
    for s, (wavs, sr) in audio.items():
        if s != "recording":
            res[s]["speaker_sim"] = mean_ci([float(sv(to16k(w, sr)) @ e) for w, e in zip(wavs, orig)])
    del sv
    torch.cuda.empty_cache()
    mos = UTMOSv2(args.device)
    for s, (wavs, sr) in audio.items():
        res[s]["utmosv2"] = mean_ci([mos(to16k(w, sr)) for w in wavs])
    del mos
    torch.cuda.empty_cache()
    dns = DnsMos(args.device)
    for s, (wavs, sr) in audio.items():
        m = dns.score([to16k(w, sr).clip(-1, 1) for w in wavs])
        res[s].update({f"dnsmos_{k}": mean_ci(m[:, j].tolist()) for j, k in enumerate(("sig", "bak", "ovrl"))})
        bw = bandwidth_hz([torch.from_numpy(w).to(args.device) for w in wavs], 50.0, sr, round(1024 * sr / 24_000))
        res[s]["bandwidth_hz"] = float(np.median(bw))
    del dns
    torch.cuda.empty_cache()
    return res


def pct(d: dict, k: str) -> str:
    lo, hi = d[f"{k}_ci"]
    return f"{100 * d[k]:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]"


def tables(results: dict) -> str:
    rows, out = results["systems"], []
    if all("quality" in r for r in rows.values()):
        out += ["| system | in → out | WER 8 kHz [95% CI] | CER 8 kHz | WER full band | CER full band | UTMOSv2 | "
                "DNSMOS OVRL | SIG | BAK | speaker sim. | bandwidth |", "|" + "---|" * 12]
        for s, r in rows.items():
            q = r["quality"]
            rates = "24 → 24" if s == "recording" else f"{r['input_rate'] / 1000:g} → {r['output_rate'] / 1000:g}"
            sim = f"{q['speaker_sim']['mean']:.3f}" if "speaker_sim" in q else "–"
            out.append(f"| {s} | {rates} kHz | {pct(q['band'], 'wer')} | {pct(q['band'], 'cer')} | "
                       f"{pct(q['full'], 'wer')} | {pct(q['full'], 'cer')} | {q['utmosv2']['mean']:.3f} | "
                       f"{q['dnsmos_ovrl']['mean']:.3f} | {q['dnsmos_sig']['mean']:.3f} | "
                       f"{q['dnsmos_bak']['mean']:.3f} | {sim} | {q['bandwidth_hz'] / 1000:.1f} kHz |")
        out.append("")
    backends = {s: r for s, r in rows.items() if s != "recording"}
    out += ["| system | frame rate | dim | samples / frame (in → out) | delay | active channels | channel means | "
            "channel stds | posterior std / channel std |", "|" + "---|" * 9]
    for s, r in backends.items():
        post = f"{r['posterior_std_over_channel_std']:.4f}" if "posterior_std_over_channel_std" in r else "–"
        delay = "–" if s == "bigvgan" else f"{max(map(abs, r['lag_samples']))} samples"  # BigVGAN makes its own phase
        (m0, m1), (s0, s1) = r["channel_mean_range"], r["channel_std_range"]
        out.append(f"| {s} | {r['frame_rate']:g} Hz | {r['dim']} | {r['hop_in']} → {r['hop_out']} | {delay} | "
                   f"{r['active_channels']} | {m0:.2f} … {m1:.2f} | {s0:.2f} … {s1:.2f} | {post} |")
    out += ["", "| system | encode RTF | decode RTF | first window | its decode time | context for > 55 dB | "
            "default precision vs fp32 | peak GPU memory |", "|" + "---|" * 8]
    for s, r in backends.items():
        out.append(f"| {s} | {r['encode_rtf']:.4f} | {r['decode_rtf']:.4f} | {r['first_window_audio_ms']:.0f} ms "
                   f"({r['first_window_frames']} frames) | {r['first_window_decode_ms']:.1f} ms | "
                   f"{r['stream_context_ms']} ms ({r['stream_context_frames']} frames"
                   f"{', left only' if r['causal'] else ''}) | {r['default_vs_fp32_db']:.1f} dB | "
                   f"{r['peak_gpu_mb']:.0f} MB |")
    out += ["", "| context | " + " | ".join(backends) + " |", "|" + "---|" * (len(backends) + 1)]
    for ms in CONTEXT_MS:
        cells = [f"{r['stream_snr_db'][str(ms)]:.1f} dB ({frames(ms, r['frame_rate'])})" for r in backends.values()]
        out.append(f"| {ms} ms | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="prepared dataset with audio.bin (prepare --save-audio)")
    p.add_argument("--split", default="val")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    p.add_argument("--vocoder", default=None, help="fine-tuned BigVGAN-v2 (bigvgan_ft.pt); default: released")
    p.add_argument("--band", type=int, default=8000, help="band-match rate for the first WER / CER columns")
    p.add_argument("--asr", default="large-v3")
    p.add_argument("--asr-batch", type=int, default=16)
    p.add_argument("--stream-utts", type=int, default=20, help="utterances for the streaming sweep and the delay")
    p.add_argument("--first-ms", type=float, default=320, help="first streaming window (audio)")
    p.add_argument("--chunk-ms", type=float, default=2560, help="later streaming windows (audio)")
    p.add_argument("--match-db", type=float, default=55.0)
    p.add_argument("--timing-runs", type=int, default=50)
    p.add_argument("--no-judges", action="store_true", help="speed, streaming and frame statistics only")
    p.add_argument("--save-wavs", type=int, default=0, help="write the first N decoded utterances per system")
    p.add_argument("--out", default="outputs/resynthesis")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if "recording" not in args.systems:
        args.systems.insert(0, "recording")  # the speaker reference

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = MelDataset(args.data, args.split, min_frames=1, max_frames=10**9, with_audio=True)
    idx = select(ds, args.split, args.offset, args.num)
    wavs = [ds[i]["audio"] for i in idx]
    refs = [_plain(ds.items[i]["norm_text"]) for i in idx]
    audio: dict[str, tuple[list[np.ndarray], int]] = {"recording": ([w.numpy() for w in wavs], SAMPLE_RATE)}
    systems: dict[str, dict] = {"recording": {}}
    for name in args.systems:
        if name == "recording":
            continue
        backend = load_backend(name, args.device, **({"finetuned": args.vocoder} if name == "bigvgan" else {}))
        outs, res = resynthesize(backend, wavs, args)
        print(name, json.dumps({k: v for k, v in res.items() if k != "stats"}), flush=True)
        audio[name], systems[name] = (outs, backend.output_rate), res
        for k, w in enumerate(outs[: args.save_wavs]):
            (out / "wav").mkdir(exist_ok=True)
            sf.write(out / "wav" / f"{name}_{idx[k]:04d}.wav", w, backend.output_rate)
        del backend
        torch.cuda.empty_cache()
    for s, q in ({} if args.no_judges else score(audio, refs, args)).items():
        systems[s]["quality"] = q
    results = {"utterances": len(idx), "seconds": sum(len(w) for w in wavs) / SAMPLE_RATE, "split": args.split,
               "vocoder": args.vocoder or "released", "asr": args.asr, "band_hz": args.band,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               "torch": torch.__version__, "systems": systems}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.md").write_text(tables(results) + "\n")
    print(tables(results))
    print(f"-> {out / 'results.json'}")


if __name__ == "__main__":
    main()
