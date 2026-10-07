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
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..text import normalize, split_sentences, text_to_ids
from .bigvgan import load_bigvgan
from .model import DriftingTTS, durations

REPO = "Vyvo/drifting-tts-tr"


class Synthesizer:
    def __init__(self, path: str | Path, dtype: mx.Dtype = mx.float32, compile: bool = False):
        """``path``: a directory with ``config.json``, ``model.safetensors`` and ``vocoder.safetensors``
        (:mod:`drifting_tts.mlx.convert`). ``dtype``: compute type of the acoustic model; the vocoder always runs in
        float32, which its snake activations need. ``compile=True`` specialises the full graphs to input shapes:
        this can improve warm throughput but adds compilation latency on new lengths. Weights are immutable after
        compiling; create a new synthesizer to change them."""
        path = Path(path)
        self.config = cfg = json.loads((path / "config.json").read_text())
        self.model = DriftingTTS(cfg["model"], cfg["num_speakers"], cfg["n_vocab"], cfg["n_mels"])
        weights = mx.load(str(path / "model.safetensors"))
        self.model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
        self.model.eval()
        self.vocoder = load_bigvgan(str(path / "vocoder.safetensors"), cfg["vocoder"])
        self.sample_rate = cfg["sample_rate"]
        self.dtype = dtype
        # Materialise weights before requests; otherwise the first request also pays for lazy loading/casts.
        mx.eval(self.model.parameters(), self.vocoder.parameters())
        self.compiled = compile
        self._encode = mx.compile(self.model.encode) if compile else self.model.encode
        self._generate = mx.compile(self.model.generate) if compile else self.model.generate
        self._vocode = mx.compile(self.vocoder) if compile else self.vocoder

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
        start = time.perf_counter()
        chunks = list(self.stream(text, speaker, cfg_scale, temperature, length_scale, seed, pause, pitch_shift,
                                  chunk_frames=None))
        wav = np.concatenate([audio for audio, _ in chunks]) if chunks else np.zeros(0, np.float32)
        elapsed = time.perf_counter() - start
        dur = len(wav) / self.sample_rate
        acoustic = sum(info["acoustic_seconds"] for _, info in chunks)
        return wav, {"seconds": dur, "rtf_acoustic": acoustic / max(dur, 1e-6),
                     "rtf_total": elapsed / dur if dur else 0.0, "total_seconds": elapsed,
                     "ttfa_seconds": elapsed if len(wav) else None}

    def stream(self, text: str, speaker: str | int | None = None, cfg_scale: float = 2.0,
               temperature: float | None = None, length_scale: float = 1.0, seed: int = 0, pause: float = 0.15,
               pitch_shift: float = 0.0, *, chunk_frames: int | None = 512, first_chunk_frames: int = 24,
               context_frames: int | None = None) -> Iterator[tuple[np.ndarray, dict]]:
        """Yield host-ready float32 audio and timing metadata, without overlapping or dropping samples.

        The complete sentence mel is generated once (attention is bidirectional). Decode a small first chunk,
        then grow from 128 frames to ``chunk_frames`` (doubling each time), with both left and right vocoder
        context cropped away. The ramp gives playback a buffer before decoding larger chunks with less repeated
        context. ``chunk_frames=None`` yields whole sentences. The default first chunk is 256 ms; loading is excluded
        from TTFA. Timings start on the first ``next()``; elapsed time includes any pauses by the consumer.
        Compiled graphs specialise to input shapes, so an unseen length may need additional warmup.
        """
        start = time.perf_counter()
        cfg = self.config
        for name, value in (("length_scale", length_scale), ("pause", pause), ("cfg_scale", cfg_scale),
                            ("pitch_shift", pitch_shift)):
            if not np.isfinite(value) or (name == "length_scale" and value <= 0) or (name == "pause" and value < 0):
                raise ValueError(f"invalid {name}: {value}")
        for name, value in (("chunk_frames", chunk_frames), ("first_chunk_frames", first_chunk_frames)):
            if value is None and name == "chunk_frames":
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        context = self.vocoder.context_frames if context_frames is None else context_frames
        if not isinstance(context, int) or context < self.vocoder.context_frames:
            raise ValueError(f"context_frames must be at least {self.vocoder.context_frames}")
        spk_id = self.speaker_id(cfg["default_voice"] if speaker is None else speaker)
        temperature = cfg["temperature"] if temperature is None else temperature
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        scale = cfg["duration_scales"].get(str(spk_id), cfg["duration_scale"]) * length_scale
        spk, alpha = mx.array([spk_id]), mx.array([cfg_scale], dtype=self.dtype)
        key = mx.random.key(seed)
        silence = np.zeros(int(pause * self.sample_rate), np.float32)
        first_audio = None
        hop = self.vocoder.hop_length
        for sentence_index, sentence in enumerate(split_sentences(normalize(text))):
            if sentence_index and len(silence):
                yield silence.copy(), {"sentence_index": sentence_index, "chunk_index": -1, "is_silence": True,
                                       "seconds": len(silence) / self.sample_rate, "acoustic_seconds": 0.0,
                                       "vocoder_seconds": 0.0, "ttfa_seconds": first_audio,
                                       "elapsed_seconds": time.perf_counter() - start}
            ids = text_to_ids(sentence, normalized=True)
            t0 = time.perf_counter()
            tokens, logw = self._encode(mx.array([ids]), spk, pitch_shift)
            w = durations(np.array(logw[0].astype(mx.float32)), scale)
            if not w.sum():
                raise ValueError("predicted duration is zero; increase length_scale")
            frames = tokens[:, mx.array(np.repeat(np.arange(len(ids)), w))]
            key, kz, kl = mx.random.split(key, 3)
            z = mx.random.normal((1, frames.shape[1], self.model.n_mels), key=kz).astype(self.dtype) * temperature
            gen = self.model.generator
            labels = mx.random.randint(0, gen.noise_classes, (1, max(1, gen.noise_coords)), key=kl)
            mel = self._generate(z, frames, spk, alpha, labels)
            mel = mel.astype(mx.float32) * cfg["stats"]["std"] + cfg["stats"]["mean"]
            mx.eval(mel)
            del tokens, logw, frames, z, labels
            t1 = time.perf_counter()
            total, offset, chunk_index = mel.shape[1], 0, 0
            next_size = min(128, chunk_frames) if chunk_frames is not None else total
            while offset < total:
                size = total if chunk_frames is None else (
                    min(first_chunk_frames, chunk_frames) if chunk_index == 0 else next_size)
                end = min(offset + size, total)
                left, right = max(0, offset - context), min(total, end + context)
                tv = time.perf_counter()
                decoded = self._vocode(mel[:, left:right].transpose(0, 2, 1))[0]
                wav = mx.clip(decoded[(offset - left) * hop:(end - left) * hop], -1.0, 1.0)
                audio = np.array(wav, dtype=np.float32)  # evaluation and host transfer are part of TTFA
                del decoded, wav
                now = time.perf_counter()
                if first_audio is None:
                    first_audio = now - start
                yield audio, {"sentence_index": sentence_index, "chunk_index": chunk_index, "is_silence": False,
                              "seconds": len(audio) / self.sample_rate,
                              "acoustic_seconds": t1 - t0 if chunk_index == 0 else 0.0,
                              "vocoder_seconds": now - tv, "ttfa_seconds": first_audio,
                              "elapsed_seconds": now - start}
                if chunk_frames is not None and chunk_index:
                    next_size = min(2 * next_size, chunk_frames)
                offset, chunk_index = end, chunk_index + 1


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
    p.add_argument("--stream", action="store_true", help="write audio as soon as each vocoder chunk is ready")
    p.add_argument("--chunk-frames", type=int, default=512, help="maximum mel frames per subsequent audio chunk")
    p.add_argument("--first-chunk-frames", type=int, default=24, help="mel frames in the first audio chunk")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="opt into shape-specialised acoustic/vocoder graphs (slower first calls)")
    p.add_argument("--cache-limit-mb", type=int, default=256, help="MLX allocator cache limit for this CLI process")
    args = p.parse_args(argv)
    if args.cache_limit_mb < 0 or args.chunk_frames < 1 or args.first_chunk_frames < 1:
        p.error("cache limit must be nonnegative and chunk sizes must be positive")
    mx.set_cache_limit(args.cache_limit_mb * 1024 * 1024)
    kw = {"compile": args.compile}
    tts = Synthesizer(args.model, **kw) if args.model else Synthesizer.from_pretrained(**kw)
    options = dict(speaker=args.speaker, cfg_scale=args.cfg, temperature=args.temperature,
                   length_scale=args.length_scale, seed=args.seed)
    if args.stream:
        start, samples, ttfa = time.perf_counter(), 0, None
        with wave.open(args.out, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(tts.sample_rate)
            for wav, _ in tts.stream(args.text, chunk_frames=args.chunk_frames,
                                     first_chunk_frames=args.first_chunk_frames, **options):
                if ttfa is None:
                    ttfa = time.perf_counter() - start
                    print(f"first PCM: {1000 * ttfa:.1f} ms (model load excluded)", flush=True)
                f.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
                samples += len(wav)
        elapsed = time.perf_counter() - start
        seconds = samples / tts.sample_rate
        print(f"wrote {args.out}: {seconds:.2f} s, RTF {elapsed / seconds if seconds else 0:.3f}")
        return
    wav, info = tts(args.text, **options)
    write_wav(args.out, wav, tts.sample_rate)
    print(f"wrote {args.out}: {info['seconds']:.2f} s, RTF {info['rtf_total']:.3f} "
          f"(acoustic {info['rtf_acoustic']:.3f})")
