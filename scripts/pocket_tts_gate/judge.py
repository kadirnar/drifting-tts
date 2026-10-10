"""Phased judges for the pocket-tts gate (one judge on the GPU at a time, each phase waits until the shared GPU has
room, results cached per set/system/phase so a phase can be re-run alone).

* ``freya`` sets replicate ``drifting_tts.benchmark``: ``WhisperASR`` (faster-whisper large-v3, Turkish, beam 5) per
  utterance on audio band-matched to 8 kHz (and the full band), the repo's normaliser on both sides, corpus WER / CER;
  UTMOSv2 on the full band; DNSMOS P.835; WavLM-ECAPA similarity to the studio-voice centroid.
* ``gate`` sets replicate ``scripts/resynthesis_benchmark.score``: ``AsrScorer`` (batched large-v3, beam 5) on 8 kHz
  and full band, WavLM-ECAPA similarity to the recording, UTMOSv2, DNSMOS P.835, bandwidth.

``--sets`` is a JSON ``{set: {kind: "freya" | "gate", refs: [plain text], systems: {name: [24 kHz wav paths]},
centroid: [studio recordings] (freya)}}``; a ``gate`` set needs a ``recording`` system (the similarity reference).
GPU memory is checked with ``nvidia-smi`` against ``LIMIT_GB`` (the machine is shared)."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

NEED_GB = {"asr": 4.5, "mos": 2.0, "dns": 1.5, "sv": 2.2}
LIMIT_GB = 28.0


def gpu_used_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0]) / 1024


def wait_gpu(need: float) -> None:
    while (u := gpu_used_gb()) + need > LIMIT_GB:
        print(f"[wait] GPU {u:.1f} GB used, need {need} GB", flush=True)
        time.sleep(120)


def load(path: str) -> np.ndarray:
    w, sr = sf.read(path, dtype="float32")
    assert sr == 24000, (path, sr)
    return w


def to16k(w: np.ndarray, band: int = 0) -> np.ndarray:
    from drifting_tts.benchmark import band_match

    return band_match(torch.from_numpy(w), band)


def run_phase(phase: str, sets: dict, cache: Path, device: str, asr_batch: int = 16) -> None:
    todo = [(s, n) for s, d in sets.items() for n in d["systems"] if not (cache / f"{s}.{n}.{phase}.json").exists()]
    if not todo:
        return
    wait_gpu(NEED_GB[phase])
    print(f"[{phase}] start ({len(todo)} system sets)", flush=True)
    if phase == "asr":
        from drifting_tts.judges import WhisperASR
        from drifting_tts.score import AsrScorer

        whisper = WhisperASR("large-v3", device)  # per-utterance (benchmark path)
        batched = None
        for s, n in todo:
            paths = sets[s]["systems"][n]
            if sets[s]["kind"] == "freya":
                res = {"band": [whisper(to16k(load(p), 8000)) for p in paths],
                       "full": [whisper(to16k(load(p), 0)) for p in paths]}
            else:
                if batched is None:
                    batched = AsrScorer("large-v3", device, beam_size=5, batch_size=asr_batch)
                res = {"band": batched.transcribe([to16k(load(p), 8000) for p in paths]),
                       "full": batched.transcribe([to16k(load(p), 0) for p in paths])}
            (cache / f"{s}.{n}.asr.json").write_text(json.dumps(res, ensure_ascii=False))
            print(f"[asr] {s}/{n}", flush=True)
        del whisper, batched
    elif phase == "mos":
        from drifting_tts.judges import UTMOSv2

        mos = UTMOSv2(device)
        for s, n in todo:
            v = [mos(to16k(load(p), 0)) for p in sets[s]["systems"][n]]
            (cache / f"{s}.{n}.mos.json").write_text(json.dumps(v))
            print(f"[mos] {s}/{n} {np.mean(v):.3f}", flush=True)
        del mos
    elif phase == "dns":
        from drifting_tts.score import DnsMos, bandwidth_hz

        dns = DnsMos(device)
        for s, n in todo:
            wavs = [load(p) for p in sets[s]["systems"][n]]
            m = dns.score([to16k(w, 0).clip(-1, 1) for w in wavs])
            bw = bandwidth_hz([torch.from_numpy(w).to(device) for w in wavs], 50.0, 24000, 1024)
            (cache / f"{s}.{n}.dns.json").write_text(json.dumps({"sig": m[:, 0].tolist(), "bak": m[:, 1].tolist(),
                                                                 "ovrl": m[:, 2].tolist(),
                                                                 "bandwidth_hz": [float(b) for b in bw]}))
            print(f"[dns] {s}/{n} {m[:, 2].mean():.3f}", flush=True)
        del dns
    elif phase == "sv":
        from drifting_tts.judges import SpeakerEmbedder

        sv = SpeakerEmbedder("wavlm-large-ecapa", device)
        emb = lambda p: sv(to16k(load(p), 0))  # noqa: E731
        for s, n in todo:
            d = sets[s]
            if d["kind"] == "freya":
                ref = torch.stack([emb(p) for p in d["centroid"]])
                c = torch.nn.functional.normalize(ref.mean(0), dim=-1)
                v = [float(emb(p) @ c) for p in d["systems"][n]]
                loo = [float(torch.nn.functional.normalize(ref.sum(0) - e, dim=-1) @ e) for e in ref]
                res = {"sim": v, "centroid_loo": loo}
            else:
                refs = [emb(p) for p in d["systems"]["recording"]]
                res = {"sim": [float(emb(p) @ r) for p, r in zip(d["systems"][n], refs)]}
            (cache / f"{s}.{n}.sv.json").write_text(json.dumps(res))
            print(f"[sv] {s}/{n} {np.mean(res['sim']):.3f}", flush=True)
        del sv
    torch.cuda.empty_cache()


def summarize_all(sets: dict, cache: Path) -> dict:
    from drifting_tts.evaluate import _plain
    from drifting_tts.metrics import bootstrap_ci, error_counts

    def corpus(refs, hyps):
        rows = [error_counts(r, _plain(h)) for r, h in zip(refs, hyps)]
        out = {}
        for name, unit in (("cer", "char"), ("wer", "word")):
            err, n = [r[f"{unit}_errors"] for r in rows], [r[f"{unit}s"] for r in rows]
            out[name], out[f"{name}_ci"] = sum(err) / max(sum(n), 1), bootstrap_ci(err, n)
        return out

    mean_ci = lambda v: {"mean": float(np.mean(v)), "ci": bootstrap_ci(v)}  # noqa: E731
    res = {}
    for s, d in sets.items():
        res[s] = {}
        for n in d["systems"]:
            r = {}
            f = cache / f"{s}.{n}.asr.json"
            if f.exists():
                a = json.loads(f.read_text())
                r["band"], r["full"] = corpus(d["refs"], a["band"]), corpus(d["refs"], a["full"])
            if (f := cache / f"{s}.{n}.mos.json").exists():
                r["utmosv2"] = mean_ci(json.loads(f.read_text()))
            if (f := cache / f"{s}.{n}.dns.json").exists():
                m = json.loads(f.read_text())
                r.update({f"dnsmos_{k}": mean_ci(m[k]) for k in ("sig", "bak", "ovrl")})
                r["bandwidth_hz"] = float(np.median(m["bandwidth_hz"]))
            if (f := cache / f"{s}.{n}.sv.json").exists():
                m = json.loads(f.read_text())
                if not (d["kind"] == "gate" and n == "recording"):
                    r["sim"] = mean_ci(m["sim"])
                if "centroid_loo" in m:
                    r["centroid_loo"] = mean_ci(m["centroid_loo"])
            res[s][n] = r
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sets", required=True, help="json: {set: {kind, refs, systems: {name: [paths]}, centroid?}}")
    p.add_argument("--out", required=True)
    p.add_argument("--phases", nargs="+", default=["dns", "mos", "sv", "asr"])
    p.add_argument("--asr-batch", type=int, default=16, help="gate sets: batched Whisper (resynthesis_benchmark: 16)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    sets = json.loads(Path(args.sets).read_text())
    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for phase in args.phases:
        for _ in range(20):
            try:
                run_phase(phase, sets, cache, args.device, args.asr_batch)
                break
            except torch.cuda.OutOfMemoryError as e:  # the GPU is shared: wait and retry
                print(f"[{phase}] OOM ({e}); retry in 3 min", flush=True)
                torch.cuda.empty_cache()
                time.sleep(180)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                    raise
                print(f"[{phase}] CUDA error ({e}); retry in 3 min", flush=True)
                torch.cuda.empty_cache()
                time.sleep(180)
    res = summarize_all(sets, cache)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
