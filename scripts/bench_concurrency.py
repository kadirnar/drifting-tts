"""Time to first audio (TTFA) under concurrency: N requests arrive at the same instant (docs/LATENCY.md).

Strategies (``--strategy``):
- ``fifo``: one request after another, each streamed to its end with the single-request fast path
  (``Synthesizer(fast=True).stream``: CUDA graphs), as a naive server; request k waits for the k - 1 before it;
- ``batched``: all N requests together (:func:`drifting_tts.batched.stream_batched`): one padded pass of the acoustic
  model over the first sentence of every request and one batch of first vocoder windows, so every request gets its
  first piece after one round; then one batch per round with the next window of every request;
- ``microbatch``: groups of ``--micro-batch`` requests in arrival order, each group streamed to its end as a batch
  before the next group starts.

TTFA of a request runs from the common arrival instant to its first audio piece on the host (as
``scripts/bench_ttfa.py``). The requests are a fixed mix of Freya-TR-Eval sentences (``--texts``): every
``--paragraph-every``-th request is a paragraph of ``--paragraph-sentences`` consecutive sentences, the others one
sentence; request i has seed i under every strategy. N = 1 runs ``--single-runs`` single requests one at a time (the
first ones of the mix). ``--check`` compares batched outputs with the single-request path (same seeds).

    python scripts/bench_concurrency.py --release v3.2 --requests 1 64 128 256 \
        --strategy fifo batched microbatch --out runs/lat_concurrency.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from drifting_tts.audio import HOP_LENGTH, SAMPLE_RATE
from drifting_tts.batched import batchable, stream_batched, synth_mels, vocode_masked
from drifting_tts.benchmark import FREYA, load_texts
from drifting_tts.synthesize import Synthesizer, add_vocoder_args
from drifting_tts.text import normalize, split_sentences
from drifting_tts.voices import DEFAULT_VOICE, voice_id


def request_mix(spec: str, n: int, every: int = 4, sentences: int = 3) -> list[str]:
    """``n`` request texts: Freya items in order, every ``every``-th request a paragraph of ``sentences`` items."""
    items = [r["text"] for r in load_texts(spec)]
    out, k = [], 0
    for i in range(n):
        m = sentences if every and i % every == every - 1 else 1
        out.append(" ".join(items[(k + j) % len(items)] for j in range(m)))
        k += m
    return out


def environment(repo: Path) -> dict:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    quota = Path("/sys/fs/cgroup/cpu.max")
    q = quota.read_text().split() if quota.exists() else []
    return {"gpu": torch.cuda.get_device_name(), "gpu_memory_gb": round(torch.cuda.mem_get_info()[1] / 2**30, 1),
            "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "torch": torch.__version__,
            "python": platform.python_version(), "cpu": cpu,
            "cpu_quota": round(int(q[0]) / int(q[1]), 2) if len(q) == 2 and q[0].isdigit() else None,
            "torch_threads": torch.get_num_threads(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "other_gpu_processes": [x for x in sh("nvidia-smi", "--query-compute-apps=pid,used_memory",
                                                  "--format=csv,noheader").splitlines()
                                    if x and x.split(",")[0].strip() != str(os.getpid())],
            "commit": sh("git", "-C", str(repo), "rev-parse", "--short", "HEAD")}


def run_fifo(synth, texts: list[str], kw: dict) -> list[list[tuple[float, int]]]:
    """Each request streamed to its end, one after another: ``[(time on the host, samples), ...]`` per request."""
    events = [[] for _ in texts]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, text in enumerate(texts):
        for piece in synth.stream(text, seed=kw["seeds"][i], **kw["stream"]):
            events[i].append((time.perf_counter() - t0, piece.numel()))
    return events


def run_batched(synth, texts: list[str], kw: dict, group: int | None = None) -> list[list[tuple[float, int]]]:
    """All requests in one batch (``group=None``) or in groups of ``group`` in arrival order."""
    events = [[] for _ in texts]
    group = group or len(texts)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(texts), group):
        idx = list(range(s, min(s + group, len(texts))))
        for out in stream_batched(synth, [texts[i] for i in idx], seeds=[kw["seeds"][i] for i in idx],
                                  **kw["stream"]):
            t = time.perf_counter() - t0
            for j, piece in out:
                events[idx[j]].append((t, piece.numel()))
    return events


def stall(ev: list[tuple[float, int]]) -> float:
    """Seconds a client that plays each request from its first piece would wait for audio (buffer underruns)."""
    play, total = ev[0][0] + ev[0][1] / SAMPLE_RATE, 0.0
    for t, n in ev[1:]:
        if t > play:
            total, play = total + t - play, t
        play += n / SAMPLE_RATE
    return total


def summarize(runs: list[list[list[tuple[float, int]]]], peaks: list[float]) -> dict:
    """Per-request TTFA percentiles pooled over the runs; per-run aggregates as the median over the runs."""
    ttfa = np.array([ev[0][0] for run in runs for ev in run]) * 1000
    stalls = np.array([stall(ev) for run in runs for ev in run]) * 1000

    def per_run(f) -> float:
        return float(np.median([f(run) for run in runs]))

    def audio(run) -> float:
        return sum(n for ev in run for _, n in ev) / SAMPLE_RATE

    def done(run) -> float:
        return max(ev[-1][0] for ev in run)

    out = {"runs": len(runs), "requests": len(runs[0]),
           "ttfa_ms": {"p50": np.median(ttfa), "p90": np.percentile(ttfa, 90), "p99": np.percentile(ttfa, 99),
                       "max": ttfa.max(), "mean": ttfa.mean()},
           "all_first_ms": per_run(lambda r: max(ev[0][0] for ev in r) * 1000),
           "all_done_ms": per_run(lambda r: done(r) * 1000),
           "done_p50_ms": per_run(lambda r: np.median([ev[-1][0] for ev in r]) * 1000),
           "audio_s": per_run(audio),
           "throughput": per_run(lambda r: audio(r) / done(r)),
           "stalled_share": float((stalls > 0).mean()), "stall_ms_max": float(stalls.max()),
           "stall_ms_p90": float(np.percentile(stalls, 90)),
           "peak_gb": max(peaks)}
    out["ttfa_ms"] = {k: round(float(v), 2) for k, v in out["ttfa_ms"].items()}
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def measure(name: str, synth, texts: list[str], kw: dict, repeats: int, group: int | None = None) -> dict:
    fn = (lambda: run_fifo(synth, texts, kw)) if name == "fifo" else (lambda: run_batched(synth, texts, kw, group))
    fn()  # warm-up: allocator, kernels of these shapes
    runs, peaks = [], []
    for _ in range(repeats):
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        runs.append(fn())
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    res = summarize(runs, peaks)
    res["resident_gb"] = round(base / 2**30, 3)  # weights, CUDA graphs, cached tensors before the run
    res["ttfa_ms_first_run"] = [round(ev[0][0] * 1000, 2) for ev in runs[0]]  # in arrival order
    return res


def measure_single(name: str, synth, texts: list[str], kw: dict, runs: int) -> dict:
    """N = 1: ``runs`` single requests (the first ones of the mix), one at a time on an otherwise idle GPU."""
    one = []
    peaks = []
    for warm in (True, False):
        for j in range(min(runs, 10) if warm else runs):
            sub = {"seeds": [kw["seeds"][j]], "stream": kw["stream"]}
            torch.cuda.reset_peak_memory_stats()
            ev = run_fifo(synth, [texts[j]], sub) if name == "fifo" else run_batched(synth, [texts[j]], sub)
            if not warm:
                one.append(ev)
                peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    res = summarize(one, peaks)
    for k in ("all_first_ms", "all_done_ms", "done_p50_ms", "audio_s", "throughput"):
        res.pop(k)  # per-run aggregates of one request: see ttfa_ms and the RTF below
    audio = [sum(n for _, n in ev[0]) / SAMPLE_RATE for ev in one]
    total = [ev[0][-1][0] for ev in one]
    res["rtf_p50"] = round(float(np.median(np.array(total) / np.array(audio))), 4)
    res["total_ms_p50"] = round(float(np.median(total)) * 1000, 2)
    return res


@torch.no_grad()
def profile_first_round(synth, texts: list[str], kw: dict, repeats: int = 5) -> dict:
    """Stages of the batched first round (synchronised between stages): frontend (normalisation and sentence split
    on the host), acoustic model (with the per-request noise draws), first vocoder windows, copy to the host."""
    from drifting_tts.batched import _take, stream_windows

    st = kw["stream"]
    rows = []
    for r in range(repeats + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first = [split_sentences(normalize(t))[0] for t in texts]
        t1 = time.perf_counter()
        gens = [torch.Generator(device=synth.device).manual_seed(s) for s in kw["seeds"][: len(texts)]]
        mel, lens = synth_mels(synth, first, st["speaker"], st["cfg_scale"], st["temperature"], gens)
        mel = synth.stats.denormalize(mel)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ws = [stream_windows(t, context=synth.vocoder.context)[0] for t in lens.tolist()]
        width = max(b for _, b, _, _ in ws)
        x = _take(mel, torch.zeros(len(ws), dtype=torch.long, device=mel.device), width)
        wav = vocode_masked(synth.vocoder, x, torch.tensor([b for _, b, _, _ in ws], device=mel.device))
        wav = wav[:, : max(n for *_, n in ws) * HOP_LENGTH]
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        wav.cpu()
        t4 = time.perf_counter()
        if r:  # the first pass is a warm-up
            rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))
    med = np.median(np.array(rows), 0) * 1000
    return {k: round(float(v), 2) for k, v in zip(("frontend_ms", "acoustic_ms", "vocoder_ms", "to_host_ms",
                                                    "total_ms"), med)}


def snr(ref: torch.Tensor, x: torch.Tensor) -> float:
    ref, x = ref.float().cpu(), x.float().cpu()
    err = (x - ref).pow(2).sum().item()
    return float("inf") if err == 0 else 10 * np.log10(ref.pow(2).sum().item() / err)


_LOGMEL = None


def mel_distance(ref: torch.Tensor, x: torch.Tensor, window: int = 10) -> tuple[float, float]:
    """Mean absolute difference of the two waveforms' log-mels (the model's BigVGAN-style front end, floor 1e-5) in
    dB, over the whole waveform and over its worst ``window`` frames (0.1 s). The waveform SNR is a harsh measure for
    this vocoder: mel differences at the float-rounding level (80 dB SNR) already give it 33-52 dB."""
    global _LOGMEL
    if _LOGMEL is None:
        from drifting_tts.audio import make_logmel

        _LOGMEL = make_logmel("bigvgan")
    a, b = (_LOGMEL(y.float().cpu()[None]) for y in (ref, x))
    d = (a - b).abs().mean(1)[0] * 20 / np.log(10)  # per frame
    n = max(1, d.numel() // window) * window
    worst = d[:n].view(-1, min(window, n)).mean(1).max()
    return float(d.mean()), float(worst)


def _stats(v: list[float]) -> dict:
    return {"min": round(min(v), 2), "median": round(float(np.median(v)), 2), "max": round(max(v), 2)}


@torch.no_grad()
def check(synth, texts: list[str], kw: dict) -> dict:
    """Batched against single-request outputs with the same seeds: the first sentence's mel (frames, SNR against the
    fast path) and each request's whole streamed audio (sample count, SNR, log-mel distance). ``reference``: the
    same measures between the fast path and the eager single-request path (no CUDA graphs), which the docs call the
    same output: the float-rounding floor that the vocoder turns into waveform differences."""
    st, seeds = kw["stream"], kw["seeds"][: len(texts)]
    eager = synth.variant()
    eager.acoustic, eager.vocoder_graphs = None, None
    first = [split_sentences(normalize(t))[0] for t in texts]
    gens = [torch.Generator(device=synth.device).manual_seed(s) for s in seeds]
    mel, lens = synth_mels(synth, first, st["speaker"], st["cfg_scale"], st["temperature"], gens)
    pieces = {i: [] for i in range(len(texts))}
    for out in stream_batched(synth, texts, seeds=seeds, **st):
        for i, p in out:
            pieces[i].append(p)
    res = {"requests": len(texts)}
    for name in ("batched", "reference"):
        mel_snr, wav_snr, wav_mel, wav_worst, frames, samples = [], [], [], [], 0, 0
        for b, (text, s) in enumerate(zip(texts, seeds)):
            ref = synth.mels(text, st["speaker"], st["cfg_scale"], st["temperature"], seed=s)[0]
            if name == "batched":
                got = synth.stats.denormalize(mel[b: b + 1, :, : int(lens[b])])
                wav = torch.cat(pieces[b])
            else:
                got = eager.mels(text, st["speaker"], st["cfg_scale"], st["temperature"], seed=s)[0]
                wav = torch.cat(list(eager.stream(text, seed=s, **st)))
            if got.shape == ref.shape:
                frames += 1
                mel_snr.append(snr(synth.stats.normalize(ref), synth.stats.normalize(got)))
            ref_wav = torch.cat(list(synth.stream(text, seed=s, **st)))
            if wav.numel() == ref_wav.numel():
                samples += 1
                wav_snr.append(snr(ref_wav, wav))
                mean, worst = mel_distance(ref_wav, wav)
                wav_mel.append(mean)
                wav_worst.append(worst)
        res[name] = {"frames_equal": frames, "samples_equal": samples, "mel_snr_db": _stats(mel_snr),
                     "audio_snr_db": _stats(wav_snr), "audio_logmel_db": _stats(wav_mel),
                     "audio_logmel_worst_0.1s_db": _stats(wav_worst)}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", default="v3.2", help="a release of Synthesizer.from_pretrained")
    add_vocoder_args(p)  # default: the release's vocoder
    p.add_argument("--requests", type=int, nargs="+", default=[1, 64, 128, 256])
    p.add_argument("--strategy", nargs="+", choices=["fifo", "batched", "microbatch"],
                   default=["fifo", "batched", "microbatch"])
    p.add_argument("--micro-batch", type=int, nargs="+", default=[16, 32], help="group sizes of microbatch")
    p.add_argument("--texts", default=FREYA, help="HF dataset id, .jsonl or .txt (default: Freya-TR-Eval)")
    p.add_argument("--paragraph-every", type=int, default=4, help="every k-th request is a paragraph (0: none)")
    p.add_argument("--paragraph-sentences", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5, help="timed runs per (strategy, N > 1), after one warm-up")
    p.add_argument("--single-runs", type=int, default=100, help="N = 1: single requests timed, after 10 warm-ups")
    p.add_argument("--check", type=int, default=8, help="requests compared with the single-request path (0: skip)")
    p.add_argument("--speaker", default=DEFAULT_VOICE)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--pause", type=float, default=0.0, help="seconds between a paragraph's sentences")
    p.add_argument("--tf32", action="store_true",
                   help="TF32 matmuls everywhere (the fast path's CUDA graphs and the batched passes); changes the "
                        "output slightly (default: fp32 as shipped)")
    p.add_argument("--out", default=None, help="write the results as JSON")
    args = p.parse_args()

    env = environment(Path(__file__).resolve().parents[1])  # before loading: other processes on the GPU
    over = {"vocoder": args.vocoder} if args.vocoder else {}
    t = time.perf_counter()
    synth = Synthesizer.from_pretrained(args.release, "cuda", fast=True, cuda_kernel=args.cuda_kernel, tf32=args.tf32,
                                        **over)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # the eager (batched) passes; the graphs captured theirs
    load_s = time.perf_counter() - t
    n_max = max(max(args.requests), args.single_runs)
    texts = request_mix(args.texts, n_max, args.paragraph_every, args.paragraph_sentences)
    kw = {"seeds": list(range(n_max)),
          "stream": dict(speaker=voice_id(args.speaker), cfg_scale=args.cfg, temperature=args.temperature,
                         pause=args.pause)}
    res = {"env": env,
           "config": {**{k: v for k, v in vars(args).items() if k != "out"}, "vocoder_name": synth.vocoder.name,
                      "prosody": synth.prosody is not None, "prosody_durations": synth.prosody_durations,
                      "graphed_acoustic": synth.acoustic is not None, "load_s": round(load_s, 1)},
           "mix": {"paragraphs": sum(len(split_sentences(normalize(x))) > 1 for x in texts[: max(args.requests)]),
                   "sentences": sum(len(split_sentences(normalize(x))) for x in texts[: max(args.requests)]),
                   "requests": max(args.requests)},
           "results": {}, "profile": {}}
    print(json.dumps(res["env"]), json.dumps(res["config"]), json.dumps(res["mix"]), sep="\n")
    can_batch = batchable(synth)
    if args.check and can_batch:
        res["check"] = check(synth, texts[: args.check], kw)
        print("check", json.dumps(res["check"]))
    head = f"{'strategy':16s} {'N':>4s} {'TTFA p50':>9s} {'p90':>8s} {'p99':>8s} {'max':>8s} {'all first':>9s} " \
           f"{'all done':>9s} {'audio/s':>8s} {'stalled':>7s} {'peak GB':>7s}"
    print(head)
    for n in args.requests:
        for name in args.strategy:
            groups = args.micro_batch if name == "microbatch" else [None]
            for m in groups:
                if name != "fifo" and not can_batch:
                    continue
                if name == "microbatch" and m >= n:
                    continue  # one group: the batched strategy
                label = f"microbatch-{m}" if m else name
                if n == 1:
                    r = measure_single(name, synth, texts, kw, args.single_runs)
                else:
                    r = measure(name, synth, texts[:n], kw, args.repeats, m)
                res["results"].setdefault(label, {})[str(n)] = r
                tt = r["ttfa_ms"]
                first, done = r.get("all_first_ms", tt["max"]), r.get("all_done_ms", r.get("total_ms_p50", 0))
                print(f"{label:16s} {n:4d} {tt['p50']:7.1f}ms {tt['p90']:6.1f}ms {tt['p99']:6.1f}ms {tt['max']:6.1f}ms "
                      f"{first:7.1f}ms {done:7.1f}ms {r.get('throughput', 1 / r.get('rtf_p50', 1)):8.1f} "
                      f"{r['stalled_share']:7.2f} {r['peak_gb']:7.2f}", flush=True)
        if can_batch and n > 1 and "batched" in args.strategy:
            res["profile"][str(n)] = profile_first_round(synth, texts[:n], kw)
            print("batched first round", n, json.dumps(res["profile"][str(n)]), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
