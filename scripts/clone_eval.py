"""Zero-shot voice cloning test (#95, #85): speakers with a prompt recording and the texts of other utterances of
theirs. Every system speaks the target texts in the prompt's voice. ``SIM-o`` is the cosine between the WavLM-ECAPA
embeddings of the generated audio and of the prompt recording; WER is Whisper's.

The manifest: ``{speaker: {"prompt": {"file", "text"}, "targets": [{"file", "text"}, ...]}}``, files next to it;
``"speaker_id"`` (a row of the model's speaker table) for voices the model knows, and a target's ``"file"`` only
where a recording of it exists.

Systems (``--systems``):

* ``recording``: the speaker's own recording of the target: what a perfect clone would score;
* ``other``: another speaker's recording: the floor of the similarity;
* ``id``: the speaker's own row of the model (a voice it was trained on): what the model does without a prompt;
* ``nearest``: no cloning. The training speaker whose voice, as the model renders it, is nearest to the prompt
  (``--speaker-embeddings``: one WavLM-ECAPA embedding per speaker row);
* ``prefix``: the nearest speaker, and the prompt as a prefix with its transcript: a model trained without prompts
  continues the recording as it continues a longer utterance (the zero-training baseline of #95);
* ``prompt``: a model trained with prompts (``model.ar.prompt``): the recording alone, no transcript, no speaker row;
* ``reference``: a model with a reference encoder (``model.ar.reference``, #96): the recording through the encoder,
  no prompt in front of the output;
* ``both``: the prompt and the reference encoder together.

``--f0``: the spread of the generated F0 (semitones) beside the similarity and the WER, the guard of #96.

Two similarities. ``SIM-o`` is with the prompt recording, the usual one; a prompt is continued by the model, so it
also rewards a copy of that recording's conditions. ``SIM (unheard)`` is with a recording of the speaker that the
model was not given (neither the prompt, nor the reference of ``--reference-from other``, nor the target's own): the
voice alone. Reported where a speaker has at least three target recordings.

    python scripts/clone_eval.py --model runs/ar_en_prompt/model_ema.pt \
        --vocoder runs/en_bigvgan_gta/bigvgan_ft_40000.pt --manifest data/clone_eval/libritts_r/manifest.json \
        --systems recording other prompt --out outputs/clone/prompt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SYSTEMS = ("recording", "other", "id", "nearest", "prefix", "prompt", "reference", "both")
MODES = {"prompt": "prefix", "reference": "reference", "both": "both"}  # Synthesizer.reference


def load16(path: Path) -> np.ndarray:
    from drifting_tts.audio import resample

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()


@torch.no_grad()
def continue_prompt(synth, prompt_mel: torch.Tensor, prompt_text: str, text: str, spk: int, temperature: float,
                    seed: int) -> torch.Tensor:
    """``prefix``: the prompt (normalised mel ``[1, n_mels, P]``) under its transcript's alignment as the acoustic
    context of ``text``, spoken by speaker row ``spk``. Returns the waveform of ``text`` alone."""
    from drifting_tts.alignment import sequence_mask
    from drifting_tts.models.text_encoder import align
    from drifting_tts.text import frontend

    m, dev = synth.model, synth.device
    fe = frontend(synth.language)
    row = torch.tensor([spk], device=dev)
    ids = torch.tensor([fe.text_to_ids(prompt_text)], device=dev)  # normalised by the frontend
    h, mu, _, x_mask = m.text(ids, torch.tensor([ids.shape[1]], device=dev), row)
    y_mask = sequence_mask(torch.tensor([prompt_mel.shape[-1]], device=dev), prompt_mel.shape[-1])[:, None].float()
    attn, _ = align(mu, x_mask, prompt_mel, y_mask)
    ids = torch.tensor([fe.text_to_ids(text)], device=dev)
    cond, _ = m.aligned_condition(ids, torch.tensor([ids.shape[1]], device=dev), row, m.duration_scale)
    g = torch.Generator(device=dev).manual_seed(seed)
    mel = m.generate(cond, row, temperature, g, prefix=prompt_mel, prefix_cond=m.frame_condition(h, mu, attn))
    return synth.vocoder(synth.stats.denormalize(mel))[0].float().cpu()


def summarize(rows: list[dict]) -> dict:
    from drifting_tts.metrics import bootstrap_ci

    sim = [r["sim"] for r in rows]
    err, n = [r["word_errors"] for r in rows], [r["words"] for r in rows]
    out = {"items": len(rows), "sim": float(np.mean(sim)), "sim_ci": bootstrap_ci(sim),
           "wer": sum(err) / max(sum(n), 1), "wer_ci": bootstrap_ci(err, n)}
    f0 = [r["f0_std"] for r in rows if r.get("f0_std") is not None and np.isfinite(r["f0_std"])]
    if f0:
        out["f0_std"] = float(np.mean(f0))
    unheard = [r["sim_spk"] for r in rows if r.get("sim_spk") is not None]
    if unheard:
        out["sim_spk"], out["sim_spk_ci"] = float(np.mean(unheard)), bootstrap_ci(unheard)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocoder", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--systems", nargs="+", default=["recording", "other", "prompt"], choices=SYSTEMS)
    ap.add_argument("--speaker-embeddings", default=None, help="[speakers, dim] embeddings of the model's speaker "
                                                                 "rows (nearest, prefix)")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=6.0, help="the prompt: this much of the recording's beginning")
    ap.add_argument("--reference-seconds", type=float, default=10.0, help="what the reference encoder reads")
    ap.add_argument("--reference-from", default="prompt", choices=("prompt", "other"),
                    help="what the reference encoder reads: the prompt recording, or another recording of the "
                         "speaker (the next target's: a reference is another utterance in training, #96)")
    ap.add_argument("--f0", action="store_true", help="F0 spread of every utterance (WORLD harvest, on the CPU)")
    ap.add_argument("--cfg", type=float, default=1.0, help="guidance scale of the generated systems (#99)")
    ap.add_argument("--guide", default="text", choices=("text", "voice", "both"),
                    help="what the null branch of --cfg is without (a model trained with drift.voice_dropout)")
    ap.add_argument("--num", type=int, default=0, help="first N speakers (0: all)")
    ap.add_argument("--save-wavs", type=int, default=0, help="keep the audio of the first N speakers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from drifting_tts.audio import resample
    from drifting_tts.evaluate import _plain
    from drifting_tts.judges import SpeakerEmbedder, WhisperASR
    from drifting_tts.metrics import error_counts
    from drifting_tts.prosody import f0_contour, f0_statistics
    from drifting_tts.synthesize import Synthesizer

    root = Path(args.manifest).parent
    manifest = json.loads(Path(args.manifest).read_text())
    speakers = list(manifest)[: args.num or None]
    synth = Synthesizer(args.model, args.device, vocoder=args.vocoder)
    synth.guidance = args.guide
    sv = SpeakerEmbedder("wavlm-large-ecapa", args.device)
    asr = WhisperASR("large-v3", args.device, language=synth.language or "tr")
    table = None
    if {"nearest", "prefix"} & set(args.systems):
        if not args.speaker_embeddings:
            ap.error("nearest / prefix need --speaker-embeddings")
        table = torch.load(args.speaker_embeddings).float()
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / "rows.json"
    rows: list[dict] = json.loads(saved.read_text()) if saved.exists() else []  # an interrupted run goes on
    done = {(r["system"], r["speaker"], r["target"]) for r in rows}
    for n, spk in enumerate(speakers):
        entry = manifest[spk]
        if all((s, spk, t) in done for s in args.systems for t in range(len(entry["targets"]))):
            continue
        rows = [r for r in rows if r["speaker"] != spk or r["system"] not in args.systems]
        prompt_path = root / entry["prompt"]["file"]
        ref_emb = sv(load16(prompt_path)).float().cpu()
        files = [t.get("file") for t in entry["targets"]]  # the speaker's other recordings, for SIM (unheard)
        heard = {i: sv(load16(root / f)).float().cpu() for i, f in enumerate(files) if f} if len(files) >= 3 else {}
        near = None if table is None else int((table @ ref_emb).argmax())
        def voice(system: str, t: int, prompt_path=prompt_path, entry=entry):
            """The voice of target ``t``: the prompt recording, and for the reference encoder with
            ``--reference-from other`` the recording of the speaker's next target (never the one spoken)."""
            mode = None if synth.model.ref_enc is None else MODES[system]
            if args.reference_from == "prompt" or mode in (None, "prefix"):
                return synth.reference(prompt_path, seconds=args.seconds, mode=mode,
                                       reference_seconds=args.reference_seconds)
            other_file = root / entry["targets"][(t + 1) % len(entry["targets"])]["file"]
            ref = synth.reference(other_file, mode="reference", reference_seconds=args.reference_seconds)
            if mode == "both":  # the prompt recording in front of the output, the other one through the encoder
                ref.mel = synth.reference_mel(prompt_path, seconds=args.seconds)
            return ref

        fixed = args.reference_from == "prompt"
        refs = {s: voice(s, 0) for s in args.systems if s in MODES} if fixed else {}
        prompt_mel = synth.reference_mel(prompt_path, seconds=args.seconds) if "prefix" in args.systems else None
        other = manifest[speakers[(n + 1) % len(speakers)]]
        for t, target in enumerate(entry["targets"]):
            seed = 1000 * n + t
            for system in args.systems:
                if system == "recording":
                    w16 = load16(root / target["file"])
                elif system == "other":
                    w16 = load16(root / other["targets"][t % len(other["targets"])]["file"])
                else:
                    with torch.no_grad():
                        if system in ("nearest", "id"):
                            row = near if system == "nearest" else int(entry["speaker_id"])
                            wav = synth(target["text"], speaker=row, temperature=args.temperature, seed=seed,
                                        cfg_scale=args.cfg)[0]
                        elif system == "prefix":
                            wav = continue_prompt(synth, prompt_mel, entry["prompt"]["text"], target["text"], near,
                                                  args.temperature, seed)
                        else:
                            wav = synth(target["text"], ref_audio=refs[system] if fixed else voice(system, t),
                                        temperature=args.temperature, seed=seed, cfg_scale=args.cfg)[0]
                    wav = wav.float().cpu().flatten()
                    if n < args.save_wavs:
                        sf.write(args.out / f"{system}_{spk}_{t}.wav", wav.numpy(), synth.sample_rate)
                    w16 = resample(wav, synth.sample_rate, 16000).numpy()
                lang = synth.language or "tr"
                text = other["targets"][t % len(other["targets"])]["text"] if system == "other" else target["text"]
                ref_text, hyp = _plain(text, lang), _plain(asr(w16), lang)
                emb = sv(w16).float().cpu()
                row = {"system": system, "speaker": spk, "target": t, "sim": float(emb @ ref_emb), "nearest": near,
                       **error_counts(ref_text, hyp)}
                third = (t + 2) % len(files)  # not the target's recording, not the reference of the next target's
                if third in heard and system != "other":
                    row["sim_spk"] = float(emb @ heard[third])
                if args.f0:
                    row["f0_std"] = f0_statistics(f0_contour(w16.astype(np.float64), 16000)).get("f0_std")
                rows.append(row)
        saved.write_text(json.dumps(rows))
        print(f"{n + 1} / {len(speakers)} speakers", flush=True)
    rows = [r for r in rows if r["speaker"] in speakers]
    summary = {s: summarize([r for r in rows if r["system"] == s]) for s in dict.fromkeys(r["system"] for r in rows)}
    (args.out / "summary.json").write_text(json.dumps({"model": args.model, "manifest": args.manifest,
                                                       "seconds": args.seconds, "cfg": args.cfg,
                                                       "reference_from": args.reference_from,
                                                       "guide": args.guide, "systems": summary}, indent=1))
    lines = ["| system | SIM-o | 95% | SIM (unheard) | 95% | WER | 95% | F0 std (st) | items |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, r in summary.items():
        f0 = f"{r['f0_std']:.2f}" if "f0_std" in r else "–"
        un = f"{r['sim_spk']:.3f} | {r['sim_spk_ci'][0]:.3f}-{r['sim_spk_ci'][1]:.3f}" if "sim_spk" in r else "– | –"
        lines.append(f"| {s} | {r['sim']:.3f} | {r['sim_ci'][0]:.3f}-{r['sim_ci'][1]:.3f} | {un} | "
                     f"{100 * r['wer']:.2f}% | {100 * r['wer_ci'][0]:.2f}-{100 * r['wer_ci'][1]:.2f} | {f0} | "
                     f"{r['items']} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
