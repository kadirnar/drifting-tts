"""Measure MLX latency until real float32 PCM is available on the host.

    python scripts/bench_mlx_ttfa.py --model /path/to/mlx --out mlx_latency.json
    python scripts/bench_mlx_ttfa.py --mode buffered --no-compile --runs 10
    python scripts/bench_mlx_ttfa.py --vocoder vocos-ft --out vocos.json      # a small vocoder

Run each configuration in a fresh process. Model resolution/download and materialized weight loading are reported
separately. The first inference has no synthesis warm-up; it is process-cold, not a claim about OS disk/Metal caches.
Each input then has its own warm-up before p50/p95 measurement. Buffered mode times the public full-waveform API;
its first audio is the whole utterance, whereas stream mode measures the first nonempty, nonsilence chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

TEXTS = {
    "short sentence": "Merhaba, nasılsınız?",
    "long sentence": "İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı.",
    "4-sentence paragraph": (
        "Toplantı yarın saat 14:30'da, 2. katta başlayacak. "
        "Lütfen raporlarınızı yanınızda getirin. "
        "Prof. Dr. Ayşe Yılmaz, 250 TL'lik bağışın tamamının öğrencilere ayrılacağını söyledi. "
        "Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."
    ),
}


def run(synth, text: str, *, mode: str, chunk_frames: int, first_chunk_frames: int, seed: int,
        synchronize: Callable[[], None], options: dict, reset_peak: Callable[[], None] | None = None,
        memory_metrics: Callable[[], dict] | None = None, stream_options: dict | None = None) -> dict:
    """Time the consumer boundary; NumPy conversion forces MLX's lazy work to complete before first audio."""
    synchronize()
    if reset_peak is not None:
        reset_peak()
    start = time.perf_counter()
    if mode == "stream":
        chunks = synth.stream(text, chunk_frames=chunk_frames, first_chunk_frames=first_chunk_frames,
                              seed=seed, **options, **(stream_options or {}))
    else:
        chunks = (synth(text, seed=seed, **options),)
    ttfa = None
    first_samples = samples = count = 0
    playback_deficit = 0.0
    for audio, info in chunks:
        pcm = np.asarray(audio, dtype=np.float32)
        if pcm.size == 0:
            continue
        arrived = time.perf_counter() - start
        if ttfa is None and not info.get("is_silence", False):
            ttfa = arrived
            first_samples = pcm.size
        if ttfa is not None:
            playback_deficit = max(playback_deficit, arrived - ttfa - samples / synth.sample_rate)
        samples += pcm.size
        count += 1
    synchronize()
    total = time.perf_counter() - start
    if ttfa is None:
        raise ValueError("The input produced no speech audio; choose text that remains nonempty after normalisation.")
    seconds = samples / synth.sample_rate
    result = {"seed": seed, "ttfa_ms": 1000 * ttfa, "total_ms": 1000 * total,
              "first_audio_s": first_samples / synth.sample_rate, "audio_s": seconds,
              "samples": samples, "chunks": count, "rtf": total / seconds,
              "playback_deficit_ms": 1000 * playback_deficit}
    if memory_metrics is not None:
        result["memory_bytes"] = memory_metrics()
    return result


def summarize(runs: list[dict]) -> dict:
    result = {}
    for field in ("ttfa_ms", "total_ms", "rtf", "playback_deficit_ms"):
        values = [r[field] for r in runs]
        result[f"{field}_p50"] = float(np.percentile(values, 50))
        result[f"{field}_p95"] = float(np.percentile(values, 95))
    for field in ("first_audio_s", "audio_s", "chunks"):
        result[f"{field}_mean"] = float(np.mean([r[field] for r in runs]))
    if all("memory_bytes" in r for r in runs):
        result["peak_memory_bytes_max"] = max(r["memory_bytes"]["peak"] for r in runs)
    return result


def command_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def provenance(model_path: Path, mx) -> dict:
    root = Path(__file__).resolve().parents[1]
    config = model_path / "config.json"
    git_status = command_output(["git", "status", "--porcelain"], cwd=root)
    machine = {"system": platform.system(), "release": platform.release(), "macos": platform.mac_ver()[0],
               "architecture": platform.machine(), "processor": platform.processor()}
    if platform.system() == "Darwin":
        machine["chip"] = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        machine["memory_bytes"] = command_output(["sysctl", "-n", "hw.memsize"])
    sources = {}
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if name.startswith("drifting_tts") and source and Path(source).is_file():
            path = Path(source).resolve()
            sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "python": sys.version,
        "packages": {name: package_version(name) for name in ("mlx", "mlx-metal", "numpy", "huggingface-hub")},
        "device": str(mx.default_device()),
        "benchmark_git_commit": command_output(["git", "rev-parse", "HEAD"], cwd=root),
        "benchmark_git_dirty": bool(git_status) if git_status is not None else None,
        "imported_sources": sources,
        "model_path": str(model_path.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "model_files": {p.name: {"bytes": p.stat().st_size} for p in sorted(model_path.glob("*.safetensors"))},
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, help="local converted MLX weights directory; otherwise download from --repo")
    p.add_argument("--repo", default="Vyvo/drifting-tts-tr")
    p.add_argument("--revision", default=None, help="Hugging Face revision (commit SHA recommended)")
    p.add_argument("--mode", choices=("stream", "buffered"), default="stream")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="opt in to MLX compilation; initial graph compilation can increase cold latency")
    p.add_argument("--chunk-frames", type=int, default=512,
                   help="maximum speech frames per stream chunk (default: 512)")
    p.add_argument("--first-chunk-frames", type=int, default=24, help="speech frames in the first chunk (default: 24)")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--cache-limit-mb", type=int, default=256,
                   help="MLX device cache limit in MiB for this dedicated benchmark process (default: 256)")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32",
                   help="DiT compute dtype; the text encoder and the vocoder remain float32")
    p.add_argument("--vocoder", default=None, help="bigvgan-v2-ft (default), bigvgan-base-ft, vocos-ft or a file")
    p.add_argument("--quantize", type=int, choices=(4, 8), default=None, help="quantised DiT weights")
    p.add_argument("--fused-activations", action="store_true", help="BigVGAN activations as Metal kernels")
    p.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                   help="queue the next stream chunk before copying the current one to the host")
    p.add_argument("--runs", type=int, default=10, help="measured warm runs per input")
    p.add_argument("--warmup", type=int, default=2, help="untimed warm-up runs per input after its first run")
    p.add_argument("--text", action="append", help="custom input, repeatable; otherwise use three built-in inputs")
    p.add_argument("--speaker", default=None, help="checkpoint default unless specified")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--length-scale", type=float, default=1.0)
    p.add_argument("--pause", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0, help="first seed; measured runs increment it")
    p.add_argument("--out", type=Path, help="write provenance, summaries and every raw measurement as JSON")
    return p


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    if args.runs < 1 or args.warmup < 0 or args.chunk_frames < 1 or args.first_chunk_frames < 1:
        p.error("--runs, --chunk-frames and --first-chunk-frames must be positive; --warmup must be nonnegative")
    if args.cache_limit_mb < 0:
        p.error("--cache-limit-mb must be nonnegative")

    import mlx.core as mx

    from drifting_tts.mlx import Synthesizer
    from drifting_tts.text import normalize

    if args.mode == "stream" and not hasattr(Synthesizer, "stream"):
        p.error("this checkout has no stream API; use --mode buffered to measure the original implementation")
    texts = {f"custom {i + 1}": value for i, value in enumerate(args.text)} if args.text else TEXTS
    if any(not normalize(value) for value in texts.values()):
        p.error("every input must contain text that remains nonempty after normalisation")
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    previous_cache_limit = mx.set_cache_limit(args.cache_limit_mb * 1024 * 1024)

    def memory_metrics() -> dict:
        return {"active": mx.get_active_memory(), "cache": mx.get_cache_memory(), "peak": mx.get_peak_memory()}

    start = time.perf_counter()
    model_path = args.model
    if model_path is None:
        from huggingface_hub import snapshot_download

        try:  # only the chosen vocoder's file
            from drifting_tts.mlx.vocoder import DEFAULT_VOCODER, VOCODERS

            weights = ["model.safetensors", VOCODERS.get(args.vocoder or DEFAULT_VOCODER, "")]
        except ImportError:  # a checkout before the vocoder registry
            weights = ["*.safetensors"]
        model_path = Path(snapshot_download(args.repo, revision=args.revision, allow_patterns=[
            f"mlx/{f}" for f in ["config.json", *weights] if f])) / "mlx"
    resolve_seconds = time.perf_counter() - start
    constructor = {"dtype": getattr(mx, args.dtype)}
    supported = inspect.signature(Synthesizer.__init__).parameters
    supports_compile = "compile" in supported
    for name, value in (("compile", args.compile), ("vocoder", args.vocoder), ("quantize", args.quantize),
                        ("fused_activations", args.fused_activations)):
        if name in supported:
            constructor[name] = value
        elif value:
            p.error(f"this checkout's Synthesizer has no {name!r} option")
    prefetch_supported = "prefetch" in inspect.signature(Synthesizer.stream).parameters
    stream_options = {"prefetch": args.prefetch} if prefetch_supported else {}
    start = time.perf_counter()
    mx.reset_peak_memory()
    synth = Synthesizer(model_path, **constructor)
    mx.eval(synth.model.parameters(), synth.vocoder.parameters())
    mx.synchronize()
    load_seconds = time.perf_counter() - start
    load_memory = memory_metrics()
    options = {"speaker": args.speaker, "cfg_scale": args.cfg, "temperature": args.temperature,
               "length_scale": args.length_scale, "pause": args.pause}
    call_options = {"mode": args.mode, "chunk_frames": args.chunk_frames, "first_chunk_frames": args.first_chunk_frames,
                    "synchronize": mx.synchronize, "options": options,
                    "reset_peak": mx.reset_peak_memory, "memory_metrics": memory_metrics,
                    "stream_options": stream_options}
    result = {"schema_version": 1, "provenance": provenance(model_path, mx),
              "settings": {"mode": args.mode, "compile": args.compile if supports_compile else False,
                           "compile_supported": supports_compile, "dtype": args.dtype,
                           "vocoder": args.vocoder, "quantize": args.quantize,
                           "fused_activations": args.fused_activations, **stream_options,
                           "cache_limit_bytes": args.cache_limit_mb * 1024 * 1024,
                           "previous_cache_limit_bytes": previous_cache_limit,
                           "chunk_frames": args.chunk_frames if args.mode == "stream" else None,
                           "first_chunk_frames": args.first_chunk_frames if args.mode == "stream" else None,
                           "runs": args.runs, "warmup_per_input": args.warmup, "seed": args.seed,
                           "sample_rate": synth.sample_rate, "repo": args.repo if args.model is None else None,
                           "revision_requested": args.revision, **options},
              "methodology": {"ttfa": "request to first nonempty nonsilence float32 PCM on host",
                              "buffered_ttfa": "request to entire waveform returned by public __call__",
                              "cold": "first inference in this process after materialized weight loading",
                              "load": "constructor plus evaluation of all model and vocoder weights",
                              "total": "consume all chunks including pauses, then synchronize; excludes playback",
                              "stream_chunks": "small first chunk, then 128/256/... frames up to chunk_frames",
                              "playback_deficit": "maximum chunk arrival delay beyond PCM delivered since first audio",
                              "memory": "MLX allocator bytes; peak reset before loading and each inference",
                              "warmup": "each input separately, after its first-input run",
                              "percentiles": "NumPy linear interpolation of raw measured runs"},
              "resolve_or_download_seconds": resolve_seconds, "load_seconds": load_seconds,
              "load_memory_bytes": load_memory, "rows": {}}
    print(f"{args.device}, {args.dtype}, {args.mode}, compile={result['settings']['compile']}; "
          f"resolve/download {resolve_seconds:.3f}s, load {load_seconds:.3f}s", flush=True)
    for index, (name, text) in enumerate(texts.items()):
        first = run(synth, text, seed=args.seed, **call_options)
        if index == 0:
            result["cold_input"] = name
            result["cold"] = first
            print(f"First inference: TTFA {first['ttfa_ms']:.1f}ms, total {first['total_ms']:.1f}ms", flush=True)
        for warmup in range(args.warmup):
            run(synth, text, seed=args.seed + warmup, **call_options)
        runs = [run(synth, text, seed=args.seed + i, **call_options) for i in range(args.runs)]
        summary = summarize(runs)
        result["rows"][name] = {"text": text, "first_input_run": first, "summary": summary, "runs": runs}
        print(f"{name}: TTFA p50 {summary['ttfa_ms_p50']:.1f}ms / p95 {summary['ttfa_ms_p95']:.1f}ms, "
              f"total p50 {summary['total_ms_p50']:.1f}ms, RTF p50 {summary['rtf_p50']:.4f}, "
              f"first chunk {summary['first_audio_s_mean']:.3f}s, "
              f"playback deficit p95 {summary['playback_deficit_ms_p95']:.1f}ms", flush=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                encoding="utf-8")
    if args.out:
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
