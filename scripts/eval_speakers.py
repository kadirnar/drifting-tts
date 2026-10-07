"""Score every speaker ID of a checkpoint on a fixed public text set (default: the first 50 Freya-TR-Eval sentences).

Per speaker: WER / CER (Whisper large-v3, beam 5, 8 kHz band-matched as in the Freya protocol), DNSMOS P.835
(SIG / BAK / OVRL) and P.808, UTMOSv2 (on the first ``--mos-num`` sentences), median F0 of the generated speech and
speaking rate. Results are appended to a CSV (resumable); ``--table`` turns the CSV into a markdown table.

    python scripts/eval_speakers.py --model drifting_tts_v3.1.pt --vocoder bigvgan_v2_ft.pt --out speakers.csv
    python scripts/eval_speakers.py --table speakers.csv > SPEAKERS.md
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from drifting_tts.benchmark import band_match, load_texts
from drifting_tts.evaluate import _plain
from drifting_tts.metrics import error_counts
from drifting_tts.voices import VOICES

FIELDS = ["speaker", "voice", "wer", "cer", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_p808", "utmosv2",
          "f0_hz", "chars_per_s"]


def median_f0(wavs16: list[np.ndarray]) -> float:
    import pyworld

    f0 = []
    for w in wavs16:
        x = w.astype(np.float64)
        f, t = pyworld.dio(x, 16_000, f0_floor=60.0, f0_ceil=600.0, frame_period=10.0)
        f = pyworld.stonemask(x, f, t, 16_000)
        f0.append(f[f > 0])
    f0 = np.concatenate(f0)
    return float(np.median(f0)) if len(f0) else float("nan")


def score_speakers(args) -> None:
    from drifting_tts.judges import UTMOSv2
    from drifting_tts.score import AsrScorer, DnsMos
    from drifting_tts.synthesize import Synthesizer

    texts = [t["text"] for t in load_texts(args.texts)[: args.num]]
    refs = [_plain(t) for t in texts]
    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder)
    n_spk = synth.model.encoder.spk.num_embeddings
    asr = AsrScorer("large-v3", "cuda", beam_size=5, batch_size=25)
    dnsmos, utmos = DnsMos("cuda"), UTMOSv2("cuda", 1)
    names = {v["id"]: k for k, v in VOICES.items()}
    out = Path(args.out)
    done = {int(r["speaker"]) for r in csv.DictReader(open(out))} if out.exists() else set()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        if not done:
            writer.writeheader()
        for spk in range(n_spk):
            if spk in done:
                continue
            t0 = time.time()
            wavs = [synth(t, speaker=spk, cfg_scale=args.cfg, temperature=args.temperature, seed=i)[0]
                    for i, t in enumerate(texts)]
            full16 = [band_match(w, 0) for w in wavs]
            hyps = [_plain(h) for h in asr.transcribe([band_match(w, args.band) for w in wavs])]
            counts = [error_counts(r, h) for r, h in zip(refs, hyps)]
            mos = dnsmos.score(full16)
            seconds = sum(len(w) for w in full16) / 16_000
            row = {"speaker": spk, "voice": names.get(spk, ""),
                   "wer": sum(c["word_errors"] for c in counts) / sum(c["words"] for c in counts),
                   "cer": sum(c["char_errors"] for c in counts) / sum(c["chars"] for c in counts),
                   "dnsmos_sig": float(mos[:, 0].mean()), "dnsmos_bak": float(mos[:, 1].mean()),
                   "dnsmos_ovrl": float(mos[:, 2].mean()), "dnsmos_p808": float(mos[:, 3].mean()),
                   "utmosv2": float(np.mean([utmos(w) for w in full16[: args.mos_num]])),
                   "f0_hz": median_f0(full16[: args.mos_num]),
                   "chars_per_s": sum(len(r.replace(" ", "")) for r in refs) / seconds}
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
            f.flush()
            print(f"speaker {spk}/{n_spk}: WER {row['wer']:.2%} DNSMOS {row['dnsmos_ovrl']:.2f} "
                  f"UTMOSv2 {row['utmosv2']:.2f} ({time.time() - t0:.0f} s)", flush=True)


def summary(path: str) -> str:
    """Distribution of every metric over the speakers (markdown)."""
    rows = list(csv.DictReader(open(path)))
    cols = [("WER", "wer", 100, "%"), ("CER", "cer", 100, "%"), ("DNSMOS OVRL", "dnsmos_ovrl", 1, ""),
            ("DNSMOS P.808", "dnsmos_p808", 1, ""), ("UTMOSv2", "utmosv2", 1, "")]
    out = "| | " + " | ".join(c[0] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for name, q in [("best", 0), ("25th percentile", 25), ("median", 50), ("75th percentile", 75), ("worst", 100)]:
        cells = []
        for _, key, mult, unit in cols:
            v = np.array([float(r[key]) for r in rows]) * mult
            worse_high = key in ("wer", "cer")  # "best" = lowest error, highest score
            v = np.percentile(v, q if worse_high else 100 - q)
            cells.append(f"{v:.2f}{unit}")
        out += f"| {name} | " + " | ".join(cells) + " |\n"
    wer = np.array([float(r["wer"]) for r in rows])
    out += (f"\n{len(rows)} speakers: {int((wer < 0.02).sum())} below 2% WER, {int((wer < 0.05).sum())} below 5%, "
            f"{int((wer >= 0.10).sum())} at 10% or more.\n")
    return out


def table(path: str) -> str:
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: (float(r["wer"]), -float(r["utmosv2"])))
    head = ("| rank | speaker | voice | WER | CER | DNSMOS OVRL | SIG | BAK | P.808 | UTMOSv2 | median F0 | chars/s |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = [f"| {k} | {r['speaker']} | {r['voice'] or ''} | {100 * float(r['wer']):.2f}% | "
             f"{100 * float(r['cer']):.2f}% | "
             f"{float(r['dnsmos_ovrl']):.2f} | {float(r['dnsmos_sig']):.2f} | {float(r['dnsmos_bak']):.2f} | "
             f"{float(r['dnsmos_p808']):.2f} | {float(r['utmosv2']):.2f} | {float(r['f0_hz']):.0f} Hz | "
             f"{float(r['chars_per_s']):.1f} |" for k, r in enumerate(rows, 1)]
    return head + "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model")
    p.add_argument("--vocoder", default=None)
    p.add_argument("--texts", default="freyavoice/freya-tr-eval")
    p.add_argument("--num", type=int, default=50, help="sentences per speaker")
    p.add_argument("--mos-num", type=int, default=10, help="sentences per speaker for UTMOSv2 and F0")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--band", type=int, default=8000)
    p.add_argument("--out", default="speakers.csv")
    p.add_argument("--table", default=None, help="print the markdown table of this CSV and exit")
    p.add_argument("--summary", default=None, help="print the distribution over speakers of this CSV and exit")
    args = p.parse_args()
    if args.table:
        print(table(args.table), end="")
        return
    if args.summary:
        print(summary(args.summary), end="")
        return
    score_speakers(args)


if __name__ == "__main__":
    main()
