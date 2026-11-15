"""Where does an ARDrift model differ from the recordings it was trained to imitate? Audio-level prosody of held-out
utterances in four renditions of the same text and speaker:

* ``recording``: the recorded audio;
* ``copy``: the recording's mel through the vocoder (what the vocoder alone changes);
* ``aligned``: the model free-running under the recording's own durations (MAS): the acoustic model alone;
* ``synth``: the model with its predicted durations: the pipeline (one pass per utterance).

For each: the statistics of :func:`drifting_tts.prosody.prosody_features` (F0 spread, movement, micro-variation,
pauses, rate) and, against the recording, :func:`drifting_tts.prosody.paired_metrics` (F0 correlation along a DTW
path). ``--seeds`` > 1 adds the spread between seeds of the same sentence.

    python scripts/prosody_gap.py --model checkpoints/en/ardrift_en.pt --vocoder checkpoints/en/bigvgan_en.pt \
        --data data/en_val --split train --num 60 --out outputs/diag/prosody_gap
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

KEYS = ("f0_median_hz", "f0_std", "f0_range", "f0_move", "f0_micro", "f0_reversals", "voiced_pct", "energy_std",
        "pauses", "pause_s", "speech_s", "duration_s", "artic_sps")
PAIRED = ("f0_corr", "f0_rmse", "vuv_agree", "mfcc_dist", "dur_ratio")


def _features(job: tuple) -> dict:
    from drifting_tts.prosody import f0_contour, paired_metrics, prosody_features

    name, wav, ref, text, language = job
    f0 = f0_contour(wav)
    out = {k: v for k, v in prosody_features(wav, f0=f0, text=text, language=language).items() if k != "pause_lengths"}
    if ref is not None:
        out.update(paired_metrics(ref, wav, gen_f0=f0))
    out["name"] = name
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocoder", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.5])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--save-wavs", type=int, default=0, help="write the first N utterances' renditions")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from drifting_tts.alignment import sequence_mask
    from drifting_tts.data import MelDataset
    from drifting_tts.models.text_encoder import align
    from drifting_tts.synthesize import Synthesizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tts = Synthesizer(args.model, dev, vocoder=args.vocoder)
    model, stats, lang = tts.model, tts.stats, tts.language
    model.enable_fast()  # CUDA graphs: the same draws as the eager loop, without its per-operation overhead
    ds = MelDataset(args.data, args.split, min_frames=1, max_frames=1500, with_audio=True)
    ds.stats.mean, ds.stats.std = stats.mean, stats.std
    voc = lambda mel: tts.vocoder(stats.denormalize(mel))[0].float().cpu().numpy()  # noqa: E731

    jobs = []
    for i in range(min(args.num, len(ds))):
        it, e = ds[i], ds.items[i]
        text, y = it["text"][None].to(dev), it["mel"][None].to(dev)
        tl, spk = torch.tensor([text.shape[1]], device=dev), torch.tensor([it["spk"]], device=dev)
        rec = it["audio"].numpy()
        norm = e["norm_text"]
        jobs.append((f"{i}/recording", rec, None, norm, lang))
        jobs.append((f"{i}/copy", voc(y), rec, norm, lang))
        h, mu, _, x_mask = model.encoder(text, tl, spk)
        y_mask = sequence_mask(torch.tensor([y.shape[-1]], device=dev), y.shape[-1])[:, None].float()
        attn, _ = align(mu, x_mask, y, y_mask)
        cond = model.frame_condition(h, mu, attn)
        for T in args.temperatures:
            for seed in range(args.seeds):
                g = torch.Generator(device=dev).manual_seed(1000 * seed + i)
                aligned = voc(model.fast_generator(cond, spk, T, g))
                mel, _ = model.synthesize(text, tl, spk, temperature=T, generator=g)
                synth = voc(mel)
                jobs.append((f"{i}/aligned/T{T:g}/s{seed}", aligned, rec, norm, lang))
                jobs.append((f"{i}/synth/T{T:g}/s{seed}", synth, rec, norm, lang))
                if i < args.save_wavs and seed == 0:
                    import soundfile as sf

                    for name, w in (("recording", rec), ("aligned", aligned), ("synth", synth)):
                        sf.write(out_dir / f"utt{i}_{name}_T{T:g}.wav", w, 24000)
        if (i + 1) % 20 == 0:
            print(f"generated {i + 1}", flush=True)

    with Pool(args.workers) as pool:
        rows = pool.map(_features, jobs, chunksize=2)
    (out_dir / "rows.json").write_text(json.dumps(rows))

    groups: dict[str, list[dict]] = {}
    for r in rows:
        parts = r["name"].split("/")
        groups.setdefault("/".join(parts[1:3]) if len(parts) > 2 else parts[1], []).append(r)
    mean = lambda rs, k: float(np.nanmean([r.get(k, np.nan) for r in rs]))  # noqa: E731
    summary = {g: {k: mean(rs, k) for k in KEYS + PAIRED} for g, rs in groups.items()}
    if args.seeds > 1:  # spread between the seeds of one sentence, against the spread between sentences
        for g, rs in groups.items():
            if g in ("recording", "copy"):
                continue
            by: dict[str, list[dict]] = {}
            for r in rs:
                by.setdefault(r["name"].split("/")[0], []).append(r)
            for k in ("f0_std", "duration_s", "pause_s"):
                summary[g][f"{k}_seed_sd"] = float(np.nanmean([np.nanstd([r.get(k, np.nan) for r in v])
                                                               for v in by.values()]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    rec = summary["recording"]
    cols = ("f0_std", "f0_range", "f0_move", "f0_micro", "f0_reversals", "voiced_pct", "energy_std", "pauses",
            "pause_s", "speech_s", "artic_sps", "f0_corr", "f0_rmse", "vuv_agree")
    print(f"\n{'':22s}" + "".join(f"{c:>13s}" for c in cols))
    for g, s in summary.items():
        print(f"{g:22s}" + "".join(f"{s[c]:13.3f}" for c in cols))
    print(f"{'':22s}" + "".join(f"{c:>13s}" for c in cols[:11]))
    for g, s in summary.items():
        if g != "recording":
            print(f"{g + ' / rec':22s}" + "".join(f"{s[c] / rec[c]:13.3f}" for c in cols[:11]))


if __name__ == "__main__":
    main()
