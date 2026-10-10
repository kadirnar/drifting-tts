"""Tables of ``scripts/eval_release.sh``: the Freya runs side by side (with DNSMOS OVRL of their saved audio) and the
prosody table, written to ``<out>/summary.md``.

    python scripts/release_summary.py runs/rel_eval_final [--no-dnsmos]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDER = {"v3.1-bigvgan": 0, "v3.1-vocos": 1, "v3.2": 2}


def dnsmos_ovrl(run: Path, device: str) -> float | None:
    """Mean DNSMOS P.835 OVRL of ``run/wav/*.wav`` (cached in ``run/dnsmos.json``)."""
    cache = run / "dnsmos.json"
    if cache.exists():
        return json.loads(cache.read_text())["ovrl"]
    wavs = sorted((run / "wav").glob("*.wav"))
    if not wavs:
        return None
    import soundfile as sf
    import torch
    import torchaudio

    from drifting_tts.score import DnsMos

    dns = DnsMos(device)
    w16 = [torchaudio.functional.resample(torch.from_numpy(sf.read(w, dtype="float32")[0]), 24_000, 16_000).numpy()
           for w in wavs]
    scores = np.concatenate([dns.score(w16[i: i + 64]) for i in range(0, len(w16), 64)])
    out = {"n": len(wavs), "sig": float(scores[:, 0].mean()), "bak": float(scores[:, 1].mean()),
           "ovrl": float(scores[:, 2].mean()), "p808": float(scores[:, 3].mean())}
    cache.write_text(json.dumps(out))
    return out["ovrl"]


def ci(r: dict, k: str, pct: bool) -> str:
    v, lo_hi = r.get(k), r.get(f"{k}_ci")
    if v is None:
        return "–"
    f = (lambda x: f"{100 * x:.2f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v)}{'%' if pct else ''}" + (f" [{f(lo_hi[0])}, {f(lo_hi[1])}]" if lo_hi else "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("--no-dnsmos", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out = Path(args.out)
    runs = []
    for res in sorted(out.glob("freya*/results.json")):
        proto, voice, system = res.parent.name.split("_", 2)
        r = json.loads(res.read_text())
        row = r["rows"][0]
        dns = None if args.no_dnsmos else dnsmos_ovrl(res.parent, args.device)
        runs.append((proto, voice, system, row, dns, r))
    runs.sort(key=lambda x: (x[0] != "freya100", ["studio", "male", "female"].index(x[1]) if x[1] in
                             ("studio", "male", "female") else 9, ORDER.get(x[2], 9)))
    lines = ["| protocol | voice | system | WER [95% CI] | CER | UTMOSv2 [95% CI] | DNSMOS OVRL | RTF |",
             "|---|---|---|---|---|---|---:|---:|"]
    for proto, voice, system, row, dns, _ in runs:
        lines.append(f"| {proto} | {voice} | {system} | {ci(row, 'wer', True)} | {ci(row, 'cer', True).split(' ')[0]} "
                     f"| {ci(row, 'mos', False)} | {'–' if dns is None else f'{dns:.3f}'} | {row.get('rtf', 0):.4f} |")
    md = "## Freya-TR-Eval\n\n" + "\n".join(lines) + "\n"
    pros = out / "prosody_val722" / "results.md"
    if pros.exists():
        md += "\n## Prosody: studio val, 100 utterances\n\n" + pros.read_text()
    (out / "summary.md").write_text(md)
    print(md)
    print(f"-> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
