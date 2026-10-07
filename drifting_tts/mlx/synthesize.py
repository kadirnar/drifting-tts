"""Text-to-speech with MLX (Apple silicon): the same pipeline as :class:`drifting_tts.synthesize.Synthesizer`.

    from drifting_tts.mlx import Synthesizer
    tts = Synthesizer.from_pretrained()                     # MLX weights from Vyvo/drifting-tts-tr (mlx/)
    wav, info = tts("Merhaba, nasılsınız?", speaker="studio")
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..text import normalize, split_sentences, text_to_ids
from .bigvgan import load_bigvgan
from .model import DriftingTTS, durations

REPO = "Vyvo/drifting-tts-tr"


class Synthesizer:
    def __init__(self, path: str | Path, dtype: mx.Dtype = mx.float32):
        """``path``: a directory with ``config.json``, ``model.safetensors`` and ``vocoder.safetensors``
        (:mod:`drifting_tts.mlx.convert`). ``dtype``: compute type of the acoustic model; the vocoder always runs in
        float32, which its snake activations need."""
        path = Path(path)
        self.config = cfg = json.loads((path / "config.json").read_text())
        self.model = DriftingTTS(cfg["model"], cfg["num_speakers"], cfg["n_vocab"], cfg["n_mels"])
        weights = mx.load(str(path / "model.safetensors"))
        self.model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
        self.model.eval()
        self.vocoder = load_bigvgan(str(path / "vocoder.safetensors"), cfg["vocoder"])
        self.sample_rate = cfg["sample_rate"]
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, repo: str = REPO, revision: str | None = None, **kw) -> Synthesizer:
        from huggingface_hub import snapshot_download

        return cls(Path(snapshot_download(repo, allow_patterns=["mlx/*"], revision=revision)) / "mlx", **kw)

    def speaker_id(self, speaker: str | int) -> int:
        """A voice name (``studio``, ``male``, ``female``) or a speaker ID of the checkpoint."""
        voices = self.config["voices"]
        spk = voices[speaker] if speaker in voices else int(speaker)
        if not 0 <= spk < self.config["num_speakers"]:
            raise ValueError(f"speaker {speaker!r} is not in this model (voices: {', '.join(voices)})")
        return spk

    def __call__(self, text: str, speaker: str | int | None = None, cfg_scale: float = 2.0,
                 temperature: float | None = None, length_scale: float = 1.0, seed: int = 0, pause: float = 0.15,
                 pitch_shift: float = 0.0) -> tuple[np.ndarray, dict]:
        """Text -> 24 kHz float32 waveform, sentence by sentence. Returns the audio and timings.

        ``temperature`` defaults to the checkpoint's calibrated value (0.3); ``length_scale`` > 1 speaks slower;
        ``pitch_shift`` is in semitones."""
        cfg = self.config
        spk_id = self.speaker_id(cfg["default_voice"] if speaker is None else speaker)
        temperature = cfg["temperature"] if temperature is None else temperature
        scale = cfg["duration_scales"].get(str(spk_id), cfg["duration_scale"]) * length_scale
        spk, alpha = mx.array([spk_id]), mx.array([cfg_scale], dtype=self.dtype)
        key = mx.random.key(seed)
        silence = np.zeros(int(pause * self.sample_rate), np.float32)
        wavs, t_acoustic, t_vocoder = [], 0.0, 0.0
        for sentence in split_sentences(normalize(text)):
            ids = text_to_ids(sentence, normalized=True)
            t0 = time.perf_counter()
            tokens, logw = self.model.encode(mx.array([ids]), spk, pitch_shift)
            w = durations(np.array(logw[0].astype(mx.float32)), scale)
            frames = tokens[:, mx.array(np.repeat(np.arange(len(ids)), w))]
            key, kz, kl = mx.random.split(key, 3)
            z = mx.random.normal((1, frames.shape[1], self.model.n_mels), key=kz).astype(self.dtype) * temperature
            gen = self.model.generator
            labels = mx.random.randint(0, gen.noise_classes, (1, max(1, gen.noise_coords)), key=kl)
            mel = self.model.generate(z, frames, spk, alpha, labels)
            mel = mel.astype(mx.float32) * cfg["stats"]["std"] + cfg["stats"]["mean"]
            mx.eval(mel)
            t1 = time.perf_counter()
            wav = mx.clip(self.vocoder(mel.transpose(0, 2, 1))[0], -1.0, 1.0)
            mx.eval(wav)
            t_acoustic += t1 - t0
            t_vocoder += time.perf_counter() - t1
            wavs += [np.array(wav), silence]
        wav = np.concatenate(wavs[:-1]) if wavs else np.zeros(0, np.float32)
        dur = max(len(wav) / self.sample_rate, 1e-6)
        return wav, {"seconds": dur, "rtf_acoustic": t_acoustic / dur, "rtf_total": (t_acoustic + t_vocoder) / dur}


def write_wav(path: str | Path, wav: np.ndarray, sample_rate: int = 24000) -> None:
    """16-bit PCM WAV with the standard library (no soundfile needed)."""
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m drifting_tts.mlx", description="Turkish text-to-speech with MLX")
    p.add_argument("--text", required=True)
    p.add_argument("--speaker", default=None, help="studio (default), male, female, or a speaker ID")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--model", default=None, help="local directory of converted weights (default: download)")
    p.add_argument("--cfg", type=float, default=2.0, help="guidance scale")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--length-scale", type=float, default=1.0, help="> 1 speaks slower")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    tts = Synthesizer(args.model) if args.model else Synthesizer.from_pretrained()
    wav, info = tts(args.text, speaker=args.speaker, cfg_scale=args.cfg, temperature=args.temperature,
                    length_scale=args.length_scale, seed=args.seed)
    write_wav(args.out, wav, tts.sample_rate)
    print(f"wrote {args.out}: {info['seconds']:.2f} s, RTF {info['rtf_total']:.3f} "
          f"(acoustic {info['rtf_acoustic']:.3f})")
