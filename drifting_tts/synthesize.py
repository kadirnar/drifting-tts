"""Synthesise speech: text -> durations -> one DriftDiT evaluation (1-NFE) -> vocoder."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Iterator
from pathlib import Path

import soundfile as sf
import torch

from .audio import SAMPLE_RATE
from .data import MelStats
from .text import normalize, split_sentences, text_to_ids  # noqa: F401 (split_sentences re-exported)
from .voices import DEFAULT_VOICE, VOICES, voice_id

DEFAULT_TEMPERATURE = 0.5  # CLI default noise temperature (lowest CER in the README sweep)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="runs/tts/model_ema.pt")
    p.add_argument("--text", help="text to synthesise (or use --text-file)")
    p.add_argument("--text-file", help="one utterance per line; writes <out-dir>/<n>.wav")
    p.add_argument("--speaker", default=DEFAULT_VOICE, help=f"voice: {' or '.join(VOICES)} (default: {DEFAULT_VOICE})")
    p.add_argument("--list-speakers", action="store_true", help="list the voices")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--out-dir", default="outputs/synth")
    p.add_argument("--cfg", type=float, default=1.0, help="guidance scale alpha learned during training")
    p.add_argument("--temperature", type=float, default=None,
                   help=f"noise temperature (default: the checkpoint's preferred one, else {DEFAULT_TEMPERATURE})")
    p.add_argument("--length-scale", type=float, default=1.0,
                   help=">1 slower, <1 faster speech (on top of the calibrated duration_scale)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pause", type=float, default=0.15, help="seconds of silence between sentences")
    p.add_argument("--pitch-shift", type=float, default=0.0, help="semitones (models trained with pitch only)")
    p.add_argument("--steps", type=int, default=None, help="generator evaluations (default: as trained, 1-NFE)")
    p.add_argument("--attn-window", type=int, default=None,
                   help="sliding-window attention radius in tokens (2 frames each); default: full attention")
    add_vocoder_args(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def add_vocoder_args(p: argparse.ArgumentParser, default: str | None = None) -> None:
    from .vocoder import VOCODERS

    stock = "the stock vocoder of the model's mel front end (BigVGAN-v2 or Vocos)"
    p.add_argument("--vocoder", default=default,
                   help=f"{' | '.join(VOCODERS)} or a checkpoint (bigvgan_ft.pt / vocos_ft.pt from finetune-vocoder); "
                        "revox[:<F0 source>[:dio|harvest]] is Minori Live - Revox Vocoder 1.0 "
                        "(https://huggingface.co/minori-live/revox-vocoder-1), CC BY-NC-SA 4.0: non-commercial use "
                        f"only; default: {default or stock}")
    p.add_argument("--cuda-kernel", action="store_true",
                   help="BigVGAN: fused anti-aliased activation CUDA kernel (~3x faster vocoder, built with nvcc)")


def preferred_temperature(model) -> float:
    """The checkpoint's preferred noise temperature (``calibrate-durations --temperature``), else 0.5."""
    t = getattr(model, "temperature", None)
    return DEFAULT_TEMPERATURE if t is None else float(t)


class Synthesizer:
    def __init__(self, model_path: str | Path, device: str = "cuda", vocoder: str | None = None,
                 cuda_kernel: bool = False, fast: bool = False, compile: bool = False, tf32: bool = False):
        """``vocoder``: a name of :data:`drifting_tts.vocoder.VOCODERS` (e.g. ``bigvgan-v2-ft``, ``griffin-lim``), a
        checkpoint path, or ``None`` for the stock vocoder of the model's mel front end.
        ``fast`` (CUDA): the acoustic model and the streaming vocoder's windows run as CUDA graphs
        (:mod:`drifting_tts.fast`), captured here (about 2 s); the output is the same as without it. ``compile`` also
        fuses the DiT with ``torch.compile`` (about 20 s the first time) and ``tf32`` uses TF32 matmuls: both are
        faster but change the output slightly (0.4-0.7 dB log-spectral distance with TF32)."""
        from .train import load_tts
        from .vocoder import load_vocoder

        self.model, self.cfg, stats = load_tts(model_path, device)
        self.stats = MelStats(stats["mean"], stats["std"])
        backend = stats.get("backend", "vocos")
        self.vocoder = load_vocoder(vocoder, device, cuda_kernel=cuda_kernel, backend=backend)
        if self.vocoder.mel != backend:
            raise ValueError(f"vocoder {self.vocoder.name!r} expects {self.vocoder.mel} mels, but the model produces "
                             f"{backend} mels")
        self.device = device
        self.default_temperature = preferred_temperature(self.model)
        spk_file = Path(self.cfg.data.root) / "speakers.json"
        self.speakers = json.loads(spk_file.read_text()) if spk_file.exists() else {}
        self.acoustic, self.vocoder_graphs = None, None
        if fast:
            from .fast import GraphedAcoustic

            self.acoustic = GraphedAcoustic(self.model, compile=compile, tf32=tf32)
            self.acoustic.warmup()
            self.vocoder_graphs = self._capture_vocoder()

    def _capture_vocoder(self, first: int = 32, chunk: int = 256) -> dict | None:
        """CUDA graphs of the streaming vocoder's two window sizes, or ``None`` (eager windows) for a vocoder that
        does not stream or cannot be captured."""
        from .fast import stream_vocoder

        voc = self.vocoder
        if not voc.graphs or voc.context is None:
            return None
        graphs = {"pool": torch.cuda.graph_pool_handle()}
        mel = torch.full((1, 100, first + chunk + voc.context), self.stats.mean, device=self.device)
        try:
            with torch.no_grad():
                for _ in stream_vocoder(voc, mel, first=first, chunk=chunk, context=voc.context, graphs=graphs):
                    pass
        except RuntimeError as e:
            warnings.warn(f"vocoder {voc.name!r} runs eagerly: CUDA graph capture failed ({e})", stacklevel=3)
            return None
        return graphs

    def speaker_id(self, speaker: str | int) -> int:
        """A voice name (``male`` / ``female``), a training speaker ID, or a dataset speaker name."""
        if speaker in self.speakers and speaker not in VOICES:
            return self.speakers[speaker]
        return voice_id(speaker)

    def _speaker(self, speaker: str | int) -> tuple[torch.Tensor, float]:
        """Speaker id tensor and the voice's duration factor."""
        spk_id = self.speaker_id(speaker)
        if spk_id >= self.model.encoder.spk.num_embeddings:
            raise ValueError(f"speaker {speaker!r} ({spk_id}) is not in this checkpoint")
        tempo = getattr(self.model, "duration_scales", {}).get(spk_id, self.model.duration_scale)
        return torch.tensor([spk_id], device=self.device), tempo

    def _mel(self, sentence: str, spk: torch.Tensor, g: torch.Generator, cfg_scale: float, temperature: float,
             length_scale: float, attn_window: int | None = None, pitch_shift: float = 0.0,
             steps: int | None = None) -> torch.Tensor:
        """One normalised sentence -> normalised mel ``[1, n_mels, T]`` (CUDA graphs when ``fast`` allows it)."""
        ids = torch.tensor([text_to_ids(sentence, normalized=True)], device=self.device)
        if self.acoustic is not None and attn_window is None and not pitch_shift and steps in (None, 1):
            return self.acoustic(ids, spk, cfg_scale, temperature, length_scale, generator=g)
        mel, _ = self.model.synthesize(ids, torch.tensor([ids.shape[1]], device=self.device), spk,
                                       cfg_scale=cfg_scale, temperature=temperature, length_scale=length_scale,
                                       generator=g, attn_window=attn_window, pitch_shift=pitch_shift, steps=steps)
        return mel

    @torch.no_grad()
    def mels(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float = 1.0, temperature: float = 1.0,
             length_scale: float = 1.0, seed: int = 0) -> list[torch.Tensor]:
        """The unnormalised log-mel ``[1, n_mels, T]`` of each sentence, with the same draws as :meth:`__call__`
        (whose waveform joins their vocoded sentences with ``pause`` seconds of silence)."""
        spk, tempo = self._speaker(speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        return [self.stats.denormalize(self._mel(s, spk, g, cfg_scale, temperature, length_scale * tempo))
                for s in split_sentences(normalize(text))]

    @torch.no_grad()
    def __call__(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float = 1.0,
                 temperature: float = 1.0, length_scale: float = 1.0, seed: int = 0, pause: float = 0.15,
                 attn_window: int | None = None, pitch_shift: float = 0.0,
                 steps: int | None = None) -> tuple[torch.Tensor, dict]:
        spk, tempo = self._speaker(speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        wavs, t_acoustic, t_vocoder = [], 0.0, 0.0
        silence = torch.zeros(int(pause * SAMPLE_RATE))
        for sentence in split_sentences(normalize(text)):
            self._sync()
            t0 = time.perf_counter()
            mel = self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo, attn_window,
                            pitch_shift, steps)
            self._sync()
            t1 = time.perf_counter()
            wav = self.vocoder(self.stats.denormalize(mel))[0].cpu()
            self._sync()
            t_acoustic += t1 - t0
            t_vocoder += time.perf_counter() - t1
            wavs += [wav, silence]
        wav = torch.cat(wavs[:-1]) if wavs else torch.zeros(0)
        dur = max(wav.numel() / SAMPLE_RATE, 1e-6)
        return wav, {"seconds": dur, "rtf_acoustic": t_acoustic / dur, "rtf_total": (t_acoustic + t_vocoder) / dur}

    @torch.no_grad()
    def stream(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float = 1.0, temperature: float = 1.0,
               length_scale: float = 1.0, seed: int = 0, pause: float = 0.15, first: int = 32,
               chunk: int = 256) -> Iterator[torch.Tensor]:
        """Yield the waveform in pieces as soon as each is ready (CPU float tensors at 24 kHz).

        Each sentence is generated in one pass. The vocoder then streams: the first ``first`` mel frames (0.34 s), then
        ``chunk``-frame windows, each with the vocoder's ``context`` (32 frames for BigVGAN-v2; see
        :func:`drifting_tts.fast.stream_vocoder`). A vocoder without one (Griffin-Lim) yields each sentence whole."""
        from .fast import stream_vocoder

        spk, tempo = self._speaker(speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        silence = torch.zeros(int(pause * SAMPLE_RATE))
        for k, sentence in enumerate(split_sentences(normalize(text))):
            mel = self.stats.denormalize(self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo))
            if k:
                yield silence
            for piece in stream_vocoder(self.vocoder, mel, first=first, chunk=chunk, context=self.vocoder.context,
                                        graphs=self.vocoder_graphs):
                yield piece.cpu()

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


def run(args) -> None:
    synth = Synthesizer(args.model, args.device, vocoder=args.vocoder, cuda_kernel=args.cuda_kernel)
    if args.list_speakers:
        for name, v in VOICES.items():
            default = " (default)" if name == DEFAULT_VOICE else ""
            print(f"{name:7s} speaker {v['id']}, median pitch {v['pitch_hz']} Hz{default}")
        return
    temperature = synth.default_temperature if args.temperature is None else args.temperature
    kw = dict(speaker=args.speaker, cfg_scale=args.cfg, temperature=temperature,
              length_scale=args.length_scale, seed=args.seed, pause=args.pause, attn_window=args.attn_window,
              pitch_shift=args.pitch_shift, steps=args.steps)
    if args.text_file:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [line.strip() for line in Path(args.text_file).read_text().splitlines() if line.strip()]
        for i, line in enumerate(lines):
            wav, info = synth(line, **kw)
            sf.write(out_dir / f"{i:03d}.wav", wav.numpy(), SAMPLE_RATE)
            print(f"{out_dir / f'{i:03d}.wav'}  {info['seconds']:.2f}s  RTF {info['rtf_total']:.4f}")
    else:
        if not args.text:
            raise SystemExit("--text or --text-file is required")
        wav, info = synth(args.text, **kw)
        sf.write(args.out, wav.numpy(), SAMPLE_RATE)
        print(f"{args.out}  {info['seconds']:.2f}s  "
              f"RTF acoustic {info['rtf_acoustic']:.4f} total {info['rtf_total']:.4f}")
