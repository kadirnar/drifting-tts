"""Compare vocoders on the same mels: Freya-TR-Eval quality, speed, streaming context and size (docs/VOCODERS.md).

The v3.1 mels of the 495 Freya-TR-Eval sentences are generated once, as ``drifting-tts benchmark`` does (seed =
sentence index, ``studio`` voice, T = 0.3, alpha = 2), and cached; every vocoder then turns the same
mels into audio, joined with 0.15 s pauses as :class:`Synthesizer` does.

* ``quality``: WER / CER (Whisper large-v3 on 8 kHz band-matched audio, ``benchmark``'s normalisation, 95% bootstrap
  intervals), UTMOSv2 (full band), DNSMOS P.835 OVRL and P.808. Vocoders run without the BigVGAN CUDA kernel here.
* ``latency``: ``Synthesizer(fast=True, cuda_kernel=True)``: vocoder RTF over the cached mels (whole sentences) and
  time to first audio of ``stream()`` (median of ``--ttfa-runs`` runs of the short and long sentence of
  ``scripts/bench_ttfa.py``).
* ``contexts``: SNR of streamed against whole-sentence audio for each window context (frames on each side), in full
  fp32 (TF32 off), so that it measures the truncated context alone.

Results accumulate in ``<out>/results.json`` (phases and vocoders can be rerun separately); ``--table`` writes the
tables into a markdown file, between ``<!-- vocoders:begin -->`` and ``<!-- vocoders:end -->`` if it has them.

    python scripts/compare_vocoders.py --model drifting_tts_v3.1.pt --table docs/VOCODERS.md \\
        --extra vocos-ft=runs/vocos_bigvgan/vocos_ft.pt
"""

import argparse
import contextlib
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from bench_ttfa import TEXTS, run_stream

from drifting_tts.audio import HOP_LENGTH, SAMPLE_RATE
from drifting_tts.benchmark import FREYA, band_match, load_texts
from drifting_tts.evaluate import _plain, summarize
from drifting_tts.fast import stream_vocoder
from drifting_tts.metrics import error_counts
from drifting_tts.vocoder import VOCODERS, load_vocoder
from drifting_tts.voices import voice_id

DEFAULT = ["bigvgan-v2-ft", "bigvgan-v2", "bigvgan-v1", "bigvgan-base", "bigvgan-base-ft", "vocos-ft", "griffin-lim"]
CONTEXTS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64]
MATCH_DB, EXACT_DB = 55.0, 90.0  # pieces match / equal up to the fp32 noise floor
BEGIN, END = "<!-- vocoders:begin -->", "<!-- vocoders:end -->"


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def cached_mels(args, texts: list[str], path: Path) -> list[list[torch.Tensor]]:
    """Per sentence of the text set: the unnormalised log-mel ``[100, T]`` of each of its sentences."""
    if path.exists():
        return torch.load(path)
    from drifting_tts.synthesize import Synthesizer

    synth = Synthesizer(args.model, args.device, vocoder="griffin-lim")  # weight-free: only the mels are needed
    mels = [[m[0].cpu() for m in synth.mels(t, speaker=args.speaker, cfg_scale=args.cfg,
                                            temperature=args.temperature, seed=i)] for i, t in enumerate(texts)]
    torch.save(mels, path)
    del synth
    free()
    return mels


@contextlib.contextmanager
def full_fp32():
    """TF32 off for convolutions and matmuls (cuDNN uses TF32 convolutions by default)."""
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


@full_fp32()
def context_sweep(voc, mels: list[torch.Tensor], contexts: list[int], chunk: int) -> dict:
    """SNR (dB) of streamed (``first`` 32, then ``chunk``-frame windows) against whole-sentence vocoding."""
    full = [voc(m[None])[0] for m in mels]
    out = {}
    for c in contexts:
        err = sig = 0.0
        for m, ref in zip(mels, full):
            x = torch.cat(list(stream_vocoder(voc, m[None].to(voc.device), chunk=chunk, context=c)))
            err, sig = err + float((ref - x).square().sum()), sig + float(ref.square().sum())
        out[str(c)] = 10 * np.log10(sig / max(err, 1e-20))
    return out


def vocode(voc, mels: list[list[torch.Tensor]], pause: float = 0.15) -> list[torch.Tensor]:
    silence = torch.zeros(int(pause * SAMPLE_RATE))
    wavs = []
    for item in mels:
        pieces = [voc(m[None])[0].cpu() for m in item]
        wavs.append(torch.cat([p for w in pieces for p in (w, silence)][:-1]))
    return wavs


def score(wavs: list[torch.Tensor], refs: list[str], judges, dnsmos, band: int, out: Path) -> dict:
    rows, full16 = [], []
    for i, (wav, ref) in enumerate(zip(wavs, refs)):
        row = {"id": i, "ref": ref, "hyp": _plain(judges.asr(band_match(wav, band)))}
        row.update(error_counts(ref, row["hyp"]))
        full16.append(band_match(wav, 0))
        row["mos"] = judges.mos(full16[-1])
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(wavs)}", flush=True)
    dns = dnsmos.score(full16)
    for row, d in zip(rows, dns):
        row.update(dnsmos_sig=float(d[0]), dnsmos_bak=float(d[1]), dnsmos_ovrl=float(d[2]), dnsmos_p808=float(d[3]))
    with open(out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    res = summarize(rows)
    for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        res[k] = float(np.mean([r[k] for r in rows]))
    res["audio_s"] = sum(w.numel() for w in wavs) / SAMPLE_RATE
    return res


def latency(args, spec: str, mels: list[list[torch.Tensor]]) -> dict:
    from drifting_tts.synthesize import Synthesizer

    synth = Synthesizer(args.model, args.device, vocoder=spec, cuda_kernel=True, fast=True)
    voc, sync = synth.vocoder, torch.cuda.synchronize
    flat = [m[None].to(args.device) for item in mels for m in item]
    for m in flat[:3]:
        voc(m)
    sync()
    t0 = time.perf_counter()
    for m in flat:
        voc(m)
    sync()
    res = {"cuda_kernel": voc.kind == "bigvgan", "graphs": synth.vocoder_graphs is not None, "context": voc.context,
           "rtf": (time.perf_counter() - t0) / (sum(m.shape[-1] for m in flat) * HOP_LENGTH / SAMPLE_RATE)}
    kw = dict(speaker=voice_id(args.speaker), temperature=args.temperature, cfg=args.cfg)
    run_stream(synth, TEXTS["long sentence"], 0, **kw)  # first call after loading
    for _ in range(5):
        run_stream(synth, TEXTS["4-sentence paragraph"], 1, **kw)
    for key in ("short sentence", "long sentence"):
        ttfa = np.array([run_stream(synth, TEXTS[key], seed, **kw)["ttfa"] for seed in range(args.ttfa_runs)])
        res[f"ttfa_ms_{key.split()[0]}"] = 1000 * float(np.median(ttfa))
        res[f"ttfa_ms_{key.split()[0]}_p90"] = 1000 * float(np.percentile(ttfa, 90))
    del synth, voc, flat
    free()
    return res


def tables(res: dict) -> str:
    vocs = res["vocoders"]
    names = [n for n in res.get("order", vocs) if n in vocs]

    def ci(q: dict, k: str) -> str:
        lo, hi = q[f"{k}_ci"]
        return f"{100 * q[k]:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]"

    lines = ["| vocoder | parameters | WER | CER | UTMOSv2 | DNSMOS OVRL | DNSMOS P.808 | vocoder RTF | "
             "TTFA short | TTFA long |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        v, q, lat = vocs[n], vocs[n].get("quality"), vocs[n].get("latency")
        cells = [f"`{n}`", f"{v['params'] / 1e6:.1f} M" if v["params"] else "0"]
        cells += ([ci(q, "wer"), ci(q, "cer"), f"{q['mos']:.3f}", f"{q['dnsmos_ovrl']:.3f}", f"{q['dnsmos_p808']:.3f}"]
                  if q else ["–"] * 5)
        cells += ([f"{lat['rtf']:.4f}", f"{lat['ttfa_ms_short']:.1f} ms", f"{lat['ttfa_ms_long']:.1f} ms"]
                  if lat else ["–"] * 3)
        lines.append("| " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"

    swept = [n for n in names if vocs[n].get("context_snr")]
    if swept:
        ctx = list(vocs[swept[0]]["context_snr"])
        out += ("\nStreamed against whole-sentence audio, SNR in dB (full fp32), by context in frames on each side of "
                "a window:\n\n| vocoder | " + " | ".join(ctx) + f" | > {MATCH_DB:g} dB from | > {EXACT_DB:g} dB from | "
                "used |\n|---|" + "---|" * (len(ctx) + 3) + "\n")
        for n in swept:
            snr = vocs[n]["context_snr"]
            first = [next((c for c in ctx if snr[c] > db), "–") for db in (MATCH_DB, EXACT_DB)]
            out += (f"| `{n}` | " + " | ".join(f"{snr[c]:.1f}" for c in ctx)
                    + f" | {first[0]} | {first[1]} | **{vocs[n]['registry_context']}** |\n")
    out += "\n| vocoder | kind | streaming | BigVGAN CUDA kernel | CUDA graphs (`fast=True`) |\n" + "|---" * 5 + "|\n"
    for n in names:
        v, lat = vocs[n], vocs[n].get("latency") or {}
        stream = f"{v['registry_context']} frames of context" if v["registry_context"] else "one piece per sentence"
        out += (f"| `{n}` | {v['kind']} | {stream} | {'yes' if v['kind'] == 'bigvgan' else '–'} | "
                f"{'yes' if lat.get('graphs') else 'no (eager)' if lat else '–'} |\n")
    return out


def write_table(path: Path, text: str) -> None:
    doc = path.read_text() if path.exists() else f"# Vocoders\n\n{BEGIN}\n{END}\n"
    if BEGIN not in doc:
        doc += f"\n{BEGIN}\n{END}\n"
    head, rest = doc.split(BEGIN, 1)
    path.write_text(f"{head}{BEGIN}\n{text}{END}{rest.split(END, 1)[1]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="TTS checkpoint (default: v3.1 from Vyvo/drifting-tts-tr)")
    p.add_argument("--vocoders", nargs="*", default=DEFAULT,
                   help="registry names; unavailable fine-tunes are skipped")
    p.add_argument("--extra", nargs="+", default=[], metavar="NAME=PATH", help="more vocoders: checkpoint paths")
    p.add_argument("--phases", nargs="+", default=["contexts", "quality", "latency"],
                   choices=["contexts", "quality", "latency"])
    p.add_argument("--texts", default=FREYA)
    p.add_argument("--num", type=int, default=0, help="first N sentences only (0: all)")
    p.add_argument("--speaker", default="studio")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--band", type=int, default=8000, help="ASR band-match rate (0: full band)")
    p.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    p.add_argument("--context-num", type=int, default=24, help="longest sentences used for the context sweep")
    p.add_argument("--context-chunk", type=int, default=64, help="window size of the sweep (4x the default's seams)")
    p.add_argument("--ttfa-runs", type=int, default=50)
    p.add_argument("--out", default="outputs/vocoders")
    p.add_argument("--table", default=None, help="write the markdown tables into this file (e.g. docs/VOCODERS.md)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.model is None:
        from huggingface_hub import hf_hub_download

        args.model = hf_hub_download("Vyvo/drifting-tts-tr", "drifting_tts_v3.1.pt")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"vocoders": {}}
    texts = [t["text"] for t in load_texts(args.texts)][: args.num or None]
    res["setup"] = {"model": str(args.model), "texts": args.texts, "sentences": len(texts), "speaker": args.speaker,
                    "temperature": args.temperature, "cfg": args.cfg, "band_hz": args.band,
                    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    specs = {n: n for n in args.vocoders} | dict(e.split("=", 1) for e in args.extra)
    res["order"] = list(dict.fromkeys(res.get("order", []) + list(specs)))
    mels = cached_mels(args, texts, out / f"mels_{len(texts)}_{args.speaker}_T{args.temperature:g}_cfg{args.cfg:g}.pt")
    refs = [_plain(t) for t in texts]

    def save() -> None:
        res_path.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    judges = dnsmos = None
    for name, spec in specs.items():
        if not ({"contexts", "quality"} & set(args.phases)):
            break
        try:
            voc = load_vocoder(spec, args.device, backend="bigvgan")
        except FileNotFoundError as e:
            print(f"skipping {name}: {e}", flush=True)
            continue
        entry = res["vocoders"].setdefault(name, {})
        entry.update(spec=spec, kind=voc.kind, params=voc.num_params, registry_context=voc.context,
                     about=VOCODERS[spec].about if spec in VOCODERS else spec)
        if "contexts" in args.phases and voc.context is not None:
            longest = sorted((m for item in mels for m in item), key=lambda m: -m.shape[-1])[: args.context_num]
            entry["context_snr"] = context_sweep(voc, [m.to(args.device) for m in longest], args.contexts,
                                                 args.context_chunk)
            print(name, "context SNR", {k: round(v, 1) for k, v in entry["context_snr"].items()}, flush=True)
        if "quality" in args.phases:
            t0 = time.perf_counter()
            wavs = vocode(voc, mels)
            del voc
            free()
            if judges is None:
                from drifting_tts.judges import load_judges
                from drifting_tts.score import DnsMos

                judges, dnsmos = load_judges("large-v3", None, "utmosv2", args.device), DnsMos(args.device)
            entry["quality"] = score(wavs, refs, judges, dnsmos, args.band, out / f"utterances_{name}.jsonl")
            q = entry["quality"]
            print(f"{name}: WER {100 * q['wer']:.2f}% CER {100 * q['cer']:.2f}% UTMOSv2 {q['mos']:.3f} "
                  f"DNSMOS {q['dnsmos_ovrl']:.3f} / {q['dnsmos_p808']:.3f} ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
        save()
        free()
    del judges, dnsmos
    free()
    if "latency" in args.phases:
        for name, spec in specs.items():
            if name not in res["vocoders"]:
                continue
            try:
                res["vocoders"][name]["latency"] = lat = latency(args, spec, mels)
            except FileNotFoundError as e:
                print(f"skipping {name}: {e}", flush=True)
                continue
            print(f"{name}: RTF {lat['rtf']:.4f}, TTFA {lat['ttfa_ms_short']:.1f} / {lat['ttfa_ms_long']:.1f} ms, "
                  f"graphs {lat['graphs']}", flush=True)
            save()
    save()
    print(tables(res))
    if args.table:
        write_table(Path(args.table), tables(res))


if __name__ == "__main__":
    main()
