"""Measurements of a text-prefix model (ARDrift v2, ``model.kind: ar_prefix``, #103): the ones behind the tables of
docs/ENGLISH.md ("The pilot: what makes a prefix model start to read" and its continuation).

* ``render``: the texts of held-out utterances through the model and a vocoder, in each utterance's own voice.
  Utterance ``i`` uses seed ``i`` and is written as ``<out>/<i>.wav``. A clock model (``model.ar.clock``) speaks at
  the corpus's mean tempo, with ``--own-tempo`` at every recording's;
* ``score``: wav folders of ``render``. WER and CER (Whisper large-v3) with a bootstrap interval, how many words
  were said and how many of those are the text's, the quiet share of the audio, with ``--mos`` UTMOSv2, and with
  ``--pair a:b`` the paired difference of two folders' WER;
* ``text-use``: teacher-forced on the recordings, how much larger the error of the head's mean patch is with the
  text of another utterance (the same noise for both texts; 0%: the states do not depend on the text), and where
  the end head fires.

Temperature: 1.0, the one the head is trained with. At the aligned model's 0.5 a prefix model falls silent after the
first words, which reads as a WER that does not move (docs/ENGLISH.md).

    python scripts/prefix_eval.py render --model runs/ar_prefix_en/model_ema.pt \
        --vocoder checkpoints/en/bigvgan_en.pt --data data/train --out outputs/prefix/t1.0
    python scripts/prefix_eval.py score --data data/train --mos t1.0=outputs/prefix/t1.0
    python scripts/prefix_eval.py text-use --model runs/ar_prefix_en/model_ema.pt --data data/train --num 200
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

QUIET_DB = -35.0  # a 20 ms frame this far under the utterance's loudest one counts as quiet


def quiet_share(wav: np.ndarray, sr: int, frame: float = 0.02, floor_db: float = QUIET_DB) -> float:
    """The share of ``frame``-second frames whose level is more than ``-floor_db`` dB under the loudest frame's
    (recordings of the English corpus: 36%; an output that fell silent after its first words: 64%)."""
    n = int(frame * sr)
    m = len(wav) // n
    if m == 0:
        return 1.0
    rms = np.sqrt((np.asarray(wav[: m * n], dtype=np.float64).reshape(m, n) ** 2).mean(1) + 1e-12)
    return float((20 * np.log10(rms / (rms.max() + 1e-12)) < floor_db).mean())


def sentence_stats(rows: list[dict]) -> dict:
    """What the transcripts say about a system, from rows of ``{ref, hyp, words, word_errors, chars, char_errors}``:

    * ``wer``, ``cer``;
    * ``said``: the words transcribed, as a share of the texts' words;
    * ``right_of_said``: the share of the transcribed words that are the text's, in the text's order;
    * ``first``: how many words from the start are right, on average;
    * ``entire`` / ``under20`` / ``under50``: sentences without an error, under 20% and under 50% WER;
    * ``beginning_only``: sentences whose transcript is the beginning of the text and nothing else."""
    words = np.array([r["words"] for r in rows], dtype=np.float64)
    errors = np.array([r["word_errors"] for r in rows], dtype=np.float64)
    first, said, right, beginning = [], 0, 0, 0
    for r in rows:
        a, b = r["ref"].split(), r["hyp"].split()
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        first.append(k)
        beginning += len(b) <= k < len(a)
        said += len(b)
        right += sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    wer = errors / np.maximum(words, 1)
    return {"wer": float(errors.sum() / words.sum()),
            "cer": float(sum(r["char_errors"] for r in rows) / sum(r["chars"] for r in rows)),
            "said": float(said / words.sum()), "right_of_said": float(right / max(said, 1)),
            "first": float(np.mean(first)), "entire": int((errors == 0).sum()), "under20": int((wer < 0.2).sum()),
            "under50": int((wer < 0.5).sum()), "beginning_only": int(beginning)}


def paired_difference(a: list[dict], b: list[dict], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """WER of ``a`` minus WER of ``b`` on the sentences both have (matched by ``name``), with a 95% percentile
    bootstrap interval over sentences."""
    by_name = {r["name"]: r for r in b}
    pairs = [(r, by_name[r["name"]]) for r in a if r["name"] in by_name]
    if not pairs:
        raise ValueError("the two folders have no file name in common")
    ea = np.array([p[0]["word_errors"] for p in pairs], dtype=np.float64)
    eb = np.array([p[1]["word_errors"] for p in pairs], dtype=np.float64)
    w = np.array([p[0]["words"] for p in pairs], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(w), (n, len(w)))
    d = (ea[idx].sum(1) - eb[idx].sum(1)) / w[idx].sum(1)
    return float((ea.sum() - eb.sum()) / w.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load16(path: Path) -> np.ndarray:
    """A wav file as 16 kHz mono float32 (the repository's sharp resampler, as ``scripts/judge_panel.py``)."""
    from drifting_tts.audio import resample

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    return (wav if sr == 16000 else resample(wav, sr, 16000)).numpy()


def _held_out(data: str, split: str, offset: int, num: int):
    from drifting_tts.data import MelDataset
    from drifting_tts.evaluate import select

    ds = MelDataset(data, split, min_frames=1, max_frames=10**9)
    return ds, select(ds, split, offset, num)


def render(args) -> None:
    from drifting_tts.audio import HOP_LENGTH, SAMPLE_RATE
    from drifting_tts.data import MelStats
    from drifting_tts.train import load_tts
    from drifting_tts.vocoder import load_vocoder

    model, _, stats = load_tts(args.model, args.device)
    model.eval()
    st = MelStats.from_dict(stats)
    voc = load_vocoder(args.vocoder, args.device)
    ds, idx = _held_out(args.data, args.split, args.offset, args.num)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = bool(getattr(model, "clock", False))
    seconds, ended = [], 0
    with torch.no_grad():
        for i in idx:
            it = ds[i]
            text = it["text"][None].to(args.device)
            chars = text.shape[1] // 2  # the text is interleaved with blanks
            rate = math.ceil(it["mel"].shape[1] / model.patch) / chars if clock and args.own_tempo else None
            g = torch.Generator(device=args.device).manual_seed(i)
            mel, _ = model.synthesize(text, torch.tensor([text.shape[1]], device=args.device),
                                      torch.tensor([it["spk"]], device=args.device), temperature=args.temperature,
                                      generator=g, rate=rate)
            # the longest output: the text's end on the clock, or `max_seconds_per_char` without one
            bound = (math.ceil(chars * (rate or model.rate)) if clock
                     else int(chars * model.max_rate * SAMPLE_RATE / HOP_LENGTH / model.patch))
            ended += int(mel.shape[-1] < bound * model.patch)
            wav = (voc(st.denormalize(mel))[0].float().cpu().numpy().squeeze() if mel.shape[-1]
                   else np.zeros(SAMPLE_RATE // 10, dtype=np.float32))
            sf.write(out / f"{i:04d}.wav", wav, SAMPLE_RATE)
            seconds.append(len(wav) / SAMPLE_RATE)
    rec = np.array([ds.items[i]["audio_seconds"] for i in idx])
    seconds = np.array(seconds)
    print(f"{out}: {len(idx)} utterances at temperature {args.temperature}, mean length {seconds.mean():.2f} s "
          f"(the recordings: {rec.mean():.2f} s), median output / recording {np.median(seconds / rec):.2f}; the end "
          f"head fired in {ended}, the others ran to the length bound", flush=True)


def score(args) -> None:
    from drifting_tts.evaluate import _plain
    from drifting_tts.judges import UTMOSv2, WhisperASR
    from drifting_tts.metrics import bootstrap_ci, error_counts

    ds, _ = _held_out(args.data, args.split, 0, 1)
    asr = WhisperASR(args.asr, args.device, language=ds.language)
    mos = UTMOSv2(args.device) if args.mos else None
    systems: dict[str, list[dict]] = {}
    print("| system | WER (95%) | CER | quiet share | words said | of them the text's | first words right "
          "| sentences right / under 20% / under 50% WER | beginning only |" + (" UTMOSv2 |" if mos else ""))
    print("|---|---|---|---|---|---|---|---|---|" + ("---|" if mos else ""))
    for spec in args.folders:
        name, _, folder = spec.rpartition("=")
        folder, name = Path(folder), name or Path(folder).name
        cache = folder / "rows.json"
        rows = json.loads(cache.read_text()) if cache.exists() and not args.fresh else []
        if not rows:
            for p in sorted(folder.glob("*.wav")):  # <index in the split>.wav, as `render` writes them
                ref = _plain(ds.items[int(p.stem)]["norm_text"], ds.language)
                hyp = _plain(asr(load16(p)), ds.language)
                wav, sr = sf.read(p)
                row = {"name": p.name, "ref": ref, "hyp": hyp, **error_counts(ref, hyp),
                       "quiet": quiet_share(wav, sr)}
                if mos:
                    row["mos"] = mos(load16(p))
                rows.append(row)
            cache.write_text(json.dumps(rows, indent=1))
        systems[name] = rows
        s = sentence_stats(rows)
        lo, hi = bootstrap_ci([r["word_errors"] for r in rows], [r["words"] for r in rows])
        line = (f"| {name} | {100 * s['wer']:.1f}% ({100 * lo:.1f} to {100 * hi:.1f}) | {100 * s['cer']:.1f}% "
                f"| {100 * np.mean([r['quiet'] for r in rows]):.0f}% | {100 * s['said']:.0f}% "
                f"| {100 * s['right_of_said']:.0f}% | {s['first']:.1f} | {s['entire']} / {s['under20']} / "
                f"{s['under50']} of {len(rows)} | {s['beginning_only']} |")
        if mos:
            line += f" {np.mean([r['mos'] for r in rows if 'mos' in r]):.2f} |" if "mos" in rows[0] else " |"
        print(line, flush=True)
    for pair in args.pair or []:
        a, b = pair.split(":")
        d, lo, hi = paired_difference(systems[a], systems[b])
        print(f"WER of {a} minus {b}: {100 * d:+.1f} points (95% {100 * lo:+.1f} to {100 * hi:+.1f})")


def text_use(args) -> None:
    from drifting_tts.models.ar import patchify
    from drifting_tts.models.ar_prefix import characters
    from drifting_tts.train import load_tts

    device = args.device
    model, _, _ = load_tts(args.model, device)
    model.eval()
    ds, idx = _held_out(args.data, args.split, args.offset, args.num)
    items = [ds[i] for i in idx]
    k = model.patch
    g = torch.Generator(device=device).manual_seed(0)
    own, other, first, end_own, end_other = [], [], [], [], []
    with torch.no_grad():
        for n, it in enumerate(items):
            y = it["mel"][None].to(device)
            y = F.pad(y, (0, (-y.shape[-1]) % k))
            steps = y.shape[-1] // k
            spk = torch.tensor([it["spk"]], device=device)
            target = patchify(y, k)[0]  # [steps, k * n_mels]
            z = torch.randn(16, steps, model.head.noise_dim, device=device, generator=g)  # the same for both texts
            res = []
            for src in (it, items[(n + 1) % len(items)]):
                text = src["text"][None].to(device)
                chars, m = characters(text, torch.tensor([text.shape[1]], device=device))
                torch.manual_seed(n)  # a pre-net's dropout: the same masks on the history for both texts
                # a clock: either text is laid over the whole utterance, as in training
                kw = {"rate": torch.tensor([steps / int(m[0])], device=device)} if getattr(model, "clock", False) \
                    else {}
                s = model.teacher_forced(chars[:, : int(m[0])], m, y, spk, **kw)[0]  # [steps + 1, dim]
                x = model.head(s[None, :steps].expand(16, -1, -1), z).float().mean(0)  # the mean patch of a step
                p = torch.sigmoid(model.eos(s).float())[:, 0]
                fired = (p > 0.5).nonzero()
                res.append((((x - target) ** 2).mean(1), int(fired[0]) if len(fired) else -1, float(p[steps])))
            own.append(res[0][0].mean())
            other.append(res[1][0].mean())
            first.append((res[0][0][:8].mean(), res[1][0][:8].mean()))
            end_own.append((res[0][1], steps, res[0][2]))
            end_other.append((res[1][1], steps, res[1][2]))
    own, other = torch.stack(own).cpu(), torch.stack(other).cpu()
    rel = other / own - 1
    boot = torch.stack([rel[torch.randint(0, len(rel), (len(rel),))].mean() for _ in range(4000)])
    f = torch.tensor(first)
    print(f"{args.model}: {len(items)} utterances, teacher-forced")
    print(f"  error of the mean patch: own text {own.mean():.4f}, the text of another utterance {other.mean():.4f}")
    print(f"  text use: the other text's error is {100 * rel.mean():+.1f}% higher per utterance "
          f"(95% {100 * boot.quantile(0.025):+.1f} to {100 * boot.quantile(0.975):+.1f})")
    print(f"  the first 8 patches: own {f[:, 0].mean():.4f}, other {f[:, 1].mean():.4f} "
          f"({100 * (f[:, 1].mean() / f[:, 0].mean() - 1):+.0f}%)")
    print(f"  end head, own text: at the right step in {sum(a == s for a, s, _ in end_own)}, earlier in "
          f"{sum(0 <= a < s for a, s, _ in end_own)}, never in {sum(a < 0 for a, _, _ in end_own)}")
    print(f"  end head, the other text: at the audio's last step in {sum(a == s for a, s, _ in end_other)} "
          f"(it reads the end from the audio if this is as often as with the own text)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def common(q, model: bool) -> None:
        if model:
            q.add_argument("--model", required=True, help="an ar_prefix checkpoint (model_ema.pt)")
        q.add_argument("--data", required=True, help="the prepared data root of the model (its held-out split)")
        q.add_argument("--split", default="val")
        q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    r = sub.add_parser("render", help="held-out texts through the model and a vocoder")
    common(r, True)
    r.add_argument("--vocoder", required=True, help="a vocoder checkpoint or registry name")
    r.add_argument("--out", required=True, help="the folder of the wav files")
    r.add_argument("--num", type=int, default=100)
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--own-tempo", action="store_true", help="a clock model at every recording's tempo, not the mean")
    r.set_defaults(run=render)

    s = sub.add_parser("score", help="WER and what the transcripts say, of wav folders of `render`")
    common(s, False)
    s.add_argument("folders", nargs="+", help="[name=]folder")
    s.add_argument("--asr", default="large-v3")
    s.add_argument("--mos", action="store_true", help="UTMOSv2 as well")
    s.add_argument("--pair", action="append", help="a:b, the paired difference of two names' WER (repeatable)")
    s.add_argument("--fresh", action="store_true", help="ignore a folder's cached rows.json")
    s.set_defaults(run=score)

    t = sub.add_parser("text-use", help="teacher-forced: does the model read its text?")
    common(t, True)
    t.add_argument("--num", type=int, default=200)
    t.add_argument("--offset", type=int, default=0)
    t.set_defaults(run=text_use)

    args = p.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
