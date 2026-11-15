"""A panel of independent learned judges on renditions of the same sentences (#85).

One predicted MOS decides little: UTMOSv2 prefers the narrower samples of every ARDrift head and correlates with
listeners at 0.39 in TTSDS2's survey (docs/HEAD_REALISM.md). This script scores every system's wav files (the same
file names in every folder) with several published predictors that were trained on different data for different
questions, and compares each system with a baseline **sentence by sentence**:

* ``utmosv2`` (VoiceMOS 2024) and ``utmos22`` (VoiceMOS 2022): naturalness MOS of synthetic speech;
* ``scoreq``: SCOREQ in its synthetic-speech domain, no reference (arXiv 2410.06675);
* ``nisqa``: NISQA v2: overall MOS and its noisiness, discontinuity, coloration and loudness dimensions;
* ``dnsmos``: DNSMOS P.835 (speech signal, background, overall) and P.808: recording quality, not naturalness;
* ``squim``: torchaudio's SQUIM subjective MOS, against a recording of another sentence as its reference;
* ``audiobox``: Audiobox Aesthetics: production quality, content enjoyment, usefulness, complexity (arXiv 2502.05139).

    python scripts/judge_panel.py --systems recording=wav/recording released=wav/released new=wav/new \
        --baseline released --out outputs/judge/panel

A judge whose package is missing is skipped with a message. Scores are cached per judge and system in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

JUDGES = ("utmosv2", "utmos22", "scoreq", "nisqa", "dnsmos", "squim", "audiobox")
# the metric of each judge that is read as "how natural / how good", for the summary across judges
PRIMARY = {"utmosv2": "utmosv2", "utmos22": "utmos22", "scoreq": "scoreq", "nisqa": "nisqa_mos",
           "dnsmos": "dnsmos_ovrl", "squim": "squim_mos", "audiobox": "audiobox_pq"}


def load16(path: Path) -> np.ndarray:
    """A wav file as 16 kHz mono float32 (the repository's sharp resampler)."""
    import soundfile as sf
    import torch

    from drifting_tts.audio import resample

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    return (wav if sr == 16000 else resample(wav, sr, 16000)).numpy()


def _forget_speechmos(hub: bool) -> None:
    """Two packages are called ``speechmos``: PyPI's (the DNSMOS models) and the torch.hub repository of UTMOS22,
    which puts itself on ``sys.path``. Forget whichever is imported; ``hub=False`` also takes the hub's off the path."""
    for key in [k for k in sys.modules if k == "speechmos" or k.startswith("speechmos.")]:
        del sys.modules[key]
    if not hub:
        sys.path[:] = [p for p in sys.path if "SpeechMOS" not in p]


def _soundfile_load(path, *args, **kwargs):
    """``torchaudio.load`` without TorchCodec (torchaudio >= 2.9 needs it, SCOREQ calls it): ``[channels, T]``, rate."""
    import soundfile as sf
    import torch

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(wav.T.copy()), sr


def build(name: str, device: str):
    """``name`` -> a function ``(paths) -> {metric: [score per path]}``."""
    import torch

    if name in ("utmosv2", "utmos22"):
        from drifting_tts.judges import UTMOS22, UTMOSv2

        _forget_speechmos(hub=True)
        judge = UTMOSv2(device) if name == "utmosv2" else UTMOS22(device)
        return lambda paths: {name: [float(judge(load16(p))) for p in paths]}
    if name == "scoreq":
        import torchaudio
        from scoreq import Scoreq

        torchaudio.load = _soundfile_load
        judge = Scoreq(data_domain="synthetic", mode="nr")
        return lambda paths: {"scoreq": [float(judge.predict(test_path=str(p), ref_path=None)) for p in paths]}
    if name == "nisqa":
        from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

        judge = NonIntrusiveSpeechQualityAssessment(16000).to(device)
        keys = ("nisqa_mos", "nisqa_noisiness", "nisqa_discontinuity", "nisqa_coloration", "nisqa_loudness")

        def nisqa(paths):
            rows = [judge(torch.from_numpy(load16(p)).to(device)).float().cpu().tolist() for p in paths]
            return {k: [r[i] for r in rows] for i, k in enumerate(keys)}

        return nisqa
    if name == "dnsmos":
        from drifting_tts.score import DnsMos

        _forget_speechmos(hub=False)
        judge = DnsMos(device)

        def dnsmos(paths):
            s = judge.score([load16(p) for p in paths])
            keys = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808")
            return {k: s[:, i].tolist() for i, k in enumerate(keys)}

        return dnsmos
    if name == "squim":
        import torchaudio

        judge = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model().to(device).eval()

        def squim(paths, reference):
            out = []
            with torch.no_grad():
                for p, r in zip(paths, reference):
                    x, ref = (torch.from_numpy(load16(q))[None].to(device) for q in (p, r))
                    out.append(float(judge(x, ref)[0]))
            return {"squim_mos": out}

        return squim
    if name == "audiobox":
        from audiobox_aesthetics.infer import initialize_predictor

        judge = initialize_predictor()

        def audiobox(paths):
            rows = []
            for s in range(0, len(paths), 16):
                rows += judge.forward([{"path": str(p)} for p in paths[s: s + 16]])
            return {f"audiobox_{k.lower()}": [float(r[k]) for r in rows] for k in ("PQ", "CE", "CU", "PC")}

        return audiobox
    raise ValueError(f"unknown judge {name!r}: one of {', '.join(JUDGES)}")


def paired(new: np.ndarray, base: np.ndarray, seed: int = 0, n_boot: int = 5000) -> dict:
    """Mean difference ``new - base`` over the same sentences, its 95% bootstrap interval and the share of sentences
    on which ``new`` is higher."""
    d = np.asarray(new, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "higher": float((d > 0).mean()),
            "significant": bool(lo > 0 or hi < 0)}


def summarize(scores: dict[str, dict[str, list[float]]], baseline: str) -> dict:
    """``{metric: {system: [score per sentence]}}`` -> per metric the mean of every system and, for the systems other
    than ``baseline``, :func:`paired` against it."""
    out = {}
    for metric, by_system in scores.items():
        out[metric] = {"mean": {s: float(np.mean(v)) for s, v in by_system.items()},
                       "vs_baseline": {s: paired(v, by_system[baseline]) for s, v in by_system.items()
                                       if s != baseline}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", required=True, help="name=folder, the same wav file names in every folder")
    ap.add_argument("--baseline", required=True, help="the system the others are compared with")
    ap.add_argument("--judges", nargs="+", default=list(JUDGES), choices=JUDGES)
    ap.add_argument("--squim-reference", default=None,
                    help="system whose wavs are SQUIM's clean reference (another sentence's; default: the first)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    systems = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in args.systems}
    if args.baseline not in systems:
        ap.error(f"--baseline {args.baseline} is not one of the systems")
    names = sorted(set.intersection(*({p.name for p in d.glob("*.wav")} for d in systems.values())))
    if not names:
        raise SystemExit("the folders share no wav file names")
    ref_dir = systems[args.squim_reference or next(iter(systems))]
    reference = [ref_dir / names[(i + 1) % len(names)] for i in range(len(names))]  # another sentence: not matching
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(names)} sentences, {len(systems)} systems, baseline {args.baseline}", flush=True)

    scores: dict[str, dict[str, list[float]]] = {}
    for jname in args.judges:
        try:
            judge = None
            for system, folder in systems.items():
                cache = args.out / f"{jname}__{system}.json"
                if cache.exists() and json.loads(cache.read_text())["names"] == names:
                    res = json.loads(cache.read_text())["scores"]
                else:
                    judge = judge or build(jname, args.device)
                    paths = [folder / n for n in names]
                    res = judge(paths, reference) if jname == "squim" else judge(paths)
                    cache.write_text(json.dumps({"names": names, "scores": res}))
                for metric, v in res.items():
                    scores.setdefault(metric, {})[system] = v
            print(f"{jname}: done", flush=True)
        except ImportError as e:
            print(f"{jname}: skipped ({e})", flush=True)
        finally:
            judge = None
    summary = summarize(scores, args.baseline)
    (args.out / "summary.json").write_text(json.dumps({"baseline": args.baseline, "sentences": len(names),
                                                       "metrics": summary}, indent=1))

    order = list(systems)
    lines = ["| metric | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for metric, r in summary.items():
        cells = []
        for s in order:
            cell = f"{r['mean'][s]:.3f}"
            if s in r["vs_baseline"]:
                v = r["vs_baseline"][s]
                cell += f" ({v['diff']:+.3f}{'*' if v['significant'] else ''})"
            cells.append(cell)
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines += ["",
              f"(difference to {args.baseline} in brackets, over the same sentences; * its 95% interval excludes 0)",
              "",
              "| system | primary metrics higher than the baseline's | significantly higher | significantly lower |",
              "|---|---|---|---|"]
    primary = [PRIMARY[j] for j in args.judges if PRIMARY[j] in summary]
    for s in order:
        if s == args.baseline:
            continue
        v = [summary[m]["vs_baseline"][s] for m in primary]
        lines.append(f"| {s} | {sum(x['diff'] > 0 for x in v)} of {len(v)} | "
                     f"{sum(x['significant'] and x['diff'] > 0 for x in v)} | "
                     f"{sum(x['significant'] and x['diff'] < 0 for x in v)} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
