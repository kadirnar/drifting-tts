"""Synthesise speech: text -> durations -> one DriftDiT evaluation (1-NFE) -> vocoder."""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import soundfile as sf
import torch

from .audio import SAMPLE_RATE
from .data import MelStats
from .latents import VAE_BACKENDS
from .text import normalize, split_sentences, text_to_ids  # noqa: F401 (split_sentences re-exported)
from .voices import DEFAULT_VOICE, VOICES, voice_id

DEFAULT_TEMPERATURE = 0.5  # CLI default noise temperature (lowest CER in the README sweep)


def add_args(p: argparse.ArgumentParser) -> None:
    from .hub import RELEASES

    p.add_argument("--release", choices=list(RELEASES), default=None,
                   help="a published release of Vyvo/drifting-tts-tr (downloaded): its model, vocoder, prosody and "
                        "pauses (v3.2: vocos-v2, the drift prosody predictor, --pause punct); --model / --vocoder / "
                        "--prosody / --pause override them")
    p.add_argument("--model", default=None, help="TTS checkpoint (default: the release's, else runs/tts/model_ema.pt)")
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
    p.add_argument("--pause", type=pause_arg, default=None,
                   help="silence between sentences: seconds, or 'punct' (by the sentence's final punctuation, "
                        "measured per voice: drifting_tts.prosody.PausePolicy; --pause-jitter > 0 varies it); "
                        "default: the release's, else 0.15")
    p.add_argument("--pause-policy", choices=["fixed", "punct"], default="fixed", help="punct: as --pause punct")
    p.add_argument("--pause-jitter", type=float, default=0.0, help="punct policy: fraction of the measured std")
    p.add_argument("--pitch-shift", type=float, default=0.0, help="semitones (models trained with pitch only)")
    p.add_argument("--steps", type=int, default=None, help="generator evaluations (default: as trained, 1-NFE)")
    p.add_argument("--attn-window", type=int, default=None,
                   help="sliding-window attention radius in tokens (2 frames each); default: full attention")
    p.add_argument("--prosody", default=None,
                   help="stochastic prosody predictor: 'drift' (the published one) or a checkpoint (train-prosody, "
                        "prosody_ema.pt); 'none': the model's deterministic duration / pitch regressors (the default "
                        "without --release)")
    p.add_argument("--prosody-temperature", type=float, default=None,
                   help="noise temperature of --prosody (default: its preferred one, else 1)")
    p.add_argument("--prosody-spread", type=float, default=1.0,
                   help="output-space temperature of --prosody (< 1: closer to its mean, flatter, less varied)")
    p.add_argument("--prosody-durations", choices=["sampled", "regressor"], default="sampled",
                   help="regressor: --prosody samples only the token pitch; durations stay the model's")
    add_vocoder_args(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def add_vocoder_args(p: argparse.ArgumentParser, default: str | None = None) -> None:
    from .vocoder import VOCODERS

    stock = "the stock vocoder of the model's mel front end (BigVGAN-v2 or Vocos)"
    p.add_argument("--vocoder", default=default,
                   help=f"{' | '.join(VOCODERS)} or a checkpoint (bigvgan_ft.pt / vocos_ft.pt from finetune-vocoder; "
                        "for a model trained on VAE latents: a fine-tuned decoder, decoder_ft.pt); "
                        "revox[:<F0 source>[:dio|harvest]] is Minori Live - Revox Vocoder 1.0 "
                        "(https://huggingface.co/minori-live/revox-vocoder-1), CC BY-NC-SA 4.0: non-commercial use "
                        f"only; default: {default or stock}")
    p.add_argument("--cuda-kernel", action="store_true",
                   help="BigVGAN: fused anti-aliased activation CUDA kernel (~3x faster vocoder, built with nvcc)")


Pause = float | str | Callable[[str, random.Random], float]
_KEEP = object()  # Synthesizer.variant: argument not given


def pause_arg(value: str) -> float | str:
    """``--pause``: seconds or ``punct[:<jitter>]``."""
    try:
        return float(value)
    except ValueError:
        if value.partition(":")[0] != "punct":
            raise argparse.ArgumentTypeError(f"expected seconds or 'punct', got {value!r}") from None
        return value


def resolve_pause(pause: Pause, speaker_id: int) -> float | Callable[[str, random.Random], float]:
    """``pause``: seconds, a callable (e.g. a :class:`drifting_tts.prosody.PausePolicy`), or ``"punct"`` /
    ``"punct:<jitter>"``: the voice's measured policy (``PausePolicy.for_voice``)."""
    if not isinstance(pause, str):
        return pause
    name, _, jitter = pause.partition(":")
    if name != "punct":
        raise ValueError(f"pause must be seconds, a callable or 'punct[:<jitter>]', got {pause!r}")
    from .prosody import PausePolicy

    return PausePolicy.for_voice(speaker_id, jitter=float(jitter or 0.0))


def silence(pause: float | Callable[[str, random.Random], float], sentence: str, rng: random.Random) -> torch.Tensor:
    """The silence after ``sentence``: ``pause`` seconds, or what a pause policy picks for it (e.g.
    :class:`drifting_tts.prosody.PausePolicy`, by its final punctuation)."""
    return torch.zeros(int((pause(sentence, rng) if callable(pause) else pause) * SAMPLE_RATE))


def preferred_temperature(model) -> float:
    """The checkpoint's preferred noise temperature (``calibrate-durations --temperature``), else 0.5."""
    t = getattr(model, "temperature", None)
    return DEFAULT_TEMPERATURE if t is None else float(t)


class Synthesizer:
    def __init__(self, model_path: str | Path, device: str = "cuda", vocoder: str | None = None,
                 cuda_kernel: bool = False, fast: bool = False, compile: bool = False, tf32: bool = False,
                 prosody: str | Path | None = None, prosody_temperature: float | None = None,
                 prosody_spread: float = 1.0, prosody_durations: str = "sampled", pause: Pause = 0.15):
        """``vocoder``: a name of :data:`drifting_tts.vocoder.VOCODERS` (e.g. ``bigvgan-v2-ft``, ``griffin-lim``), a
        checkpoint path, or ``None`` for the stock vocoder of the model's mel front end. A model trained on VAE
        latents decodes with the VAE decoder: ``vocoder`` is then ``None`` (the released decoder) or a fine-tuned
        decoder (``decoder_ft.pt`` of ``finetune-vocoder``).
        ``fast`` (CUDA): the acoustic model and the streaming vocoder's windows run as CUDA graphs
        (:mod:`drifting_tts.fast`), captured here (about 2 s); the output is the same as without it. ``compile`` also
        fuses the DiT with ``torch.compile`` (about 20 s the first time) and ``tf32`` uses TF32 matmuls: both are
        faster but change the output slightly (0.4-0.7 dB log-spectral distance with TF32).
        ``prosody``: a stochastic prosody predictor that samples the durations and token pitch instead of the
        model's deterministic regressors: a name of :data:`drifting_tts.hub.PROSODY_MODELS` (``"drift"``,
        downloaded) or a ``train-prosody`` checkpoint (see :meth:`set_prosody`).
        ``pause``: the default silence between sentences of :meth:`__call__` / :meth:`stream` (seconds, a
        :class:`~drifting_tts.prosody.PausePolicy`, or ``"punct"``: by punctuation, measured per voice).
        :meth:`from_pretrained` builds a published release (v3.2: ``vocos-v2``, ``prosody="drift"``,
        ``pause="punct"``)."""
        from .train import load_tts
        from .vocoder import load_vocoder

        self.model, self.cfg, stats = load_tts(model_path, device)
        self.stats = MelStats(stats["mean"], stats["std"])
        self.backend = stats.get("backend", "vocos")
        if self.backend in VAE_BACKENDS:  # a model trained on VAE latents: the VAE decoder is the vocoder
            from .latents.vocoder import LatentVocoder

            if vocoder is not None and not Path(vocoder).is_file():
                raise ValueError(f"a model trained on {self.backend} latents decodes them with its VAE decoder: "
                                 f"vocoder must be a fine-tuned decoder checkpoint (decoder_ft.pt), not {vocoder!r}")
            self.vocoder = LatentVocoder(self.backend, device, repeat=stats.get("latent_repeat", 1), decoder=vocoder)
        else:
            self.vocoder = load_vocoder(vocoder, device, cuda_kernel=cuda_kernel, backend=self.backend)
        if self.vocoder.mel != self.backend:
            raise ValueError(f"vocoder {self.vocoder.name!r} expects {self.vocoder.mel} mels, but the model produces "
                             f"{self.backend} mels")
        self.device = device
        self.default_temperature = preferred_temperature(self.model)
        spk_file = Path(self.cfg.data.root) / "speakers.json"
        self.speakers = json.loads(spk_file.read_text()) if spk_file.exists() else {}
        self.acoustic, self.vocoder_graphs, self.prosody = None, None, None
        self.fast, self._graph_opts = fast, {"compile": compile, "tf32": tf32}
        resolve_pause(pause, 0)  # validate
        self.pause = pause
        self.set_prosody(prosody, prosody_temperature, prosody_spread, prosody_durations)
        if fast:
            self.vocoder_graphs = self._capture_vocoder()

    @classmethod
    def from_pretrained(cls, release: str | None = None, device: str = "cuda", **kw) -> Synthesizer:
        """A published release of ``Vyvo/drifting-tts-tr`` (:data:`drifting_tts.hub.RELEASES`; default: the latest):
        its acoustic model, vocoder, prosody source and pause rule, downloaded once. Keyword arguments override
        them (``vocoder="bigvgan-v2-ft"``, ``prosody=None``, ``model=<path or Hub file>``) or pass other options
        (``fast=True``, ``cuda_kernel=True``, ...).

        ``v3.1``: BigVGAN-v2-ft, the deterministic regressors, 0.15 s pauses. ``v3.2``: the same acoustic model with
        ``vocos-v2``, the ``drift`` prosody predictor (prosody temperature 0.5) and ``pause="punct"``."""
        from .hub import LATEST, RELEASES, hub_file

        release = release or LATEST
        if release not in RELEASES:
            raise ValueError(f"unknown release {release!r}; choose one of {', '.join(RELEASES)}")
        opts = {**RELEASES[release], **kw}
        model = str(opts.pop("model"))
        return cls(model if Path(model).is_file() else hub_file(model, f"release {release}"), device, **opts)

    def variant(self, vocoder=_KEEP, prosody=_KEEP, pause: Pause = _KEEP, prosody_temperature: float | None = None,
                prosody_spread: float = 1.0, prosody_durations: str = "sampled") -> Synthesizer:
        """A Synthesizer that shares this one's acoustic model (no second copy in memory) with another vocoder (a
        registry name, a checkpoint or a loaded :class:`~drifting_tts.vocoder.Vocoder`), prosody source
        (:meth:`set_prosody`) or default pause; arguments left out are kept. E.g. v3.1 next to v3.2:
        ``Synthesizer.from_pretrained("v3.2").variant(vocoder="bigvgan-v2-ft", prosody=None, pause=0.15)``."""
        import copy

        from .vocoder import Vocoder, load_vocoder

        other = copy.copy(self)
        if vocoder is not _KEEP:
            if self.backend in VAE_BACKENDS:
                raise ValueError("a model trained on VAE latents keeps its decoder")
            other.vocoder = vocoder if isinstance(vocoder, Vocoder) else load_vocoder(vocoder, self.device,
                                                                                     backend=self.backend)
            if other.vocoder.mel != self.backend:
                raise ValueError(f"vocoder {other.vocoder.name!r} expects {other.vocoder.mel} mels")
            other.vocoder_graphs = other._capture_vocoder() if self.fast else None
        if prosody is not _KEEP:
            other.set_prosody(prosody, prosody_temperature, prosody_spread, prosody_durations)
        if pause is not _KEEP:
            resolve_pause(pause, 0)
            other.pause = pause
        return other

    def set_prosody(self, prosody=None, temperature: float | None = None, spread: float = 1.0,
                    durations: str = "sampled") -> None:
        """Where the durations and token pitch come from. ``prosody``: ``None`` (the model's deterministic
        regressors), a :class:`~drifting_tts.models.prosody_net.ProsodyPredictor`, a name of
        :data:`drifting_tts.hub.PROSODY_MODELS` (``"drift"``) or a ``train-prosody`` checkpoint.

        The predictor samples with noise temperature ``temperature`` (``None``: the checkpoint's preferred one, else
        1) and output-space temperature ``spread`` (:meth:`ProsodyPredictor.sample`); the seed of each call drives
        it, before the DiT's noise. ``durations="regressor"`` keeps the model's durations (and per-voice factors) and
        samples only the token pitch; otherwise its own per-voice duration factors replace the model's. With
        ``fast``, a one-pass predictor (``drift`` / ``mse``, spread 1, no word features) runs in the acoustic model's
        CUDA graphs (recaptured here); others run eagerly."""
        if durations not in ("sampled", "regressor"):
            raise ValueError(f"prosody_durations must be 'sampled' or 'regressor', got {durations!r}")
        self.prosody, self.prosody_temperature = None, temperature
        self.prosody_spread, self.prosody_durations = spread, durations
        if prosody is not None:
            from .models.prosody_net import ProsodyPredictor

            self.prosody = (prosody if isinstance(prosody, ProsodyPredictor)
                            else ProsodyPredictor.load(prosody, self.device, tts=self.model))
            if temperature is None:
                self.prosody_temperature = 1.0 if self.prosody.temperature is None else self.prosody.temperature
        if self.fast:
            self._capture_acoustic()

    def _capture_acoustic(self) -> None:
        from .fast import GraphedAcoustic, graphable

        self.acoustic = None
        if not graphable(self.prosody, self.prosody_spread):
            warnings.warn("this prosody predictor runs eagerly: only the vocoder windows use CUDA graphs", stacklevel=3)
            return
        self.acoustic = GraphedAcoustic(self.model, prosody=self.prosody, prosody_durations=self.prosody_durations,
                                        **self._graph_opts)
        self.acoustic.warmup()

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
        if self.prosody is not None and self.prosody_durations == "sampled":  # the sampler's own factors (1: none)
            tempo = self.prosody.duration_scales.get(spk_id, 1.0)
        else:
            tempo = getattr(self.model, "duration_scales", {}).get(spk_id, self.model.duration_scale)
        return torch.tensor([spk_id], device=self.device), tempo

    def _mel(self, sentence: str, spk: torch.Tensor, g: torch.Generator, cfg_scale: float, temperature: float,
             length_scale: float, attn_window: int | None = None, pitch_shift: float = 0.0,
             steps: int | None = None, prosody_temperature: float | None = None) -> torch.Tensor:
        """One normalised sentence -> normalised mel ``[1, n_mels, T]`` (CUDA graphs when ``fast`` allows it)."""
        ids = torch.tensor([text_to_ids(sentence, normalized=True)], device=self.device)
        pt = self.prosody_temperature if prosody_temperature is None else prosody_temperature
        if self.acoustic is not None and attn_window is None and not pitch_shift and steps in (None, 1):
            return self.acoustic(ids, spk, cfg_scale, temperature, length_scale, generator=g,
                                 prosody_temperature=1.0 if pt is None else pt)
        ids_len = torch.tensor([ids.shape[1]], device=self.device)
        durations = pitch = None
        if self.prosody is not None:  # sampled first, from the same generator as the DiT's noise
            durations, pitch = self.prosody.predict(self.model, ids, ids_len, spk, pt, length_scale, generator=g,
                                                    spread=self.prosody_spread)
            if self.prosody_durations == "regressor":
                durations = None
        mel, _ = self.model.synthesize(ids, ids_len, spk, cfg_scale=cfg_scale, temperature=temperature,
                                       length_scale=length_scale, generator=g, attn_window=attn_window,
                                       pitch_shift=pitch_shift, steps=steps, durations=durations, pitch=pitch)
        return mel

    @torch.no_grad()
    def mels(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float = 1.0, temperature: float = 1.0,
             length_scale: float = 1.0, seed: int = 0, prosody_temperature: float | None = None) -> list[torch.Tensor]:
        """The unnormalised log-mel ``[1, n_mels, T]`` of each sentence, with the same draws as :meth:`__call__`
        (whose waveform joins their vocoded sentences with ``pause`` seconds of silence)."""
        spk, tempo = self._speaker(speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        return [self.stats.denormalize(self._mel(s, spk, g, cfg_scale, temperature, length_scale * tempo,
                                                 prosody_temperature=prosody_temperature))
                for s in split_sentences(normalize(text))]

    @torch.no_grad()
    def __call__(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float = 1.0,
                 temperature: float = 1.0, length_scale: float = 1.0, seed: int = 0, pause: Pause | None = None,
                 attn_window: int | None = None, pitch_shift: float = 0.0, steps: int | None = None,
                 prosody_temperature: float | None = None) -> tuple[torch.Tensor, dict]:
        """``text`` -> waveform (CPU float tensor at 24 kHz) and timings. Sentence by sentence, each in one pass and
        vocoded whole, joined by ``pause`` (default: the Synthesizer's; seconds, a callable or ``"punct"``).
        ``prosody_temperature``: this call's noise temperature of the prosody predictor (default: the
        Synthesizer's)."""
        spk, tempo = self._speaker(speaker)
        pause = resolve_pause(self.pause if pause is None else pause, self.speaker_id(speaker))
        g = torch.Generator(device=self.device).manual_seed(seed)
        wavs, t_acoustic, t_vocoder = [], 0.0, 0.0
        rng = random.Random(seed)  # pause jitter only: the acoustic draws stay those of ``g``
        for sentence in split_sentences(normalize(text)):
            self._sync()
            t0 = time.perf_counter()
            mel = self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo, attn_window,
                            pitch_shift, steps, prosody_temperature)
            self._sync()
            t1 = time.perf_counter()
            wav = self.vocoder(self.stats.denormalize(mel))[0].cpu()
            self._sync()
            t_acoustic += t1 - t0
            t_vocoder += time.perf_counter() - t1
            wavs += [wav, silence(pause, sentence, rng)]
        wav = torch.cat(wavs[:-1]) if wavs else torch.zeros(0)
        dur = max(wav.numel() / SAMPLE_RATE, 1e-6)
        return wav, {"seconds": dur, "rtf_acoustic": t_acoustic / dur, "rtf_total": (t_acoustic + t_vocoder) / dur}

    @torch.no_grad()
    def stream(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float = 1.0, temperature: float = 1.0,
               length_scale: float = 1.0, seed: int = 0, pause: Pause | None = None,
               first: int = 32, chunk: int = 256, prosody_temperature: float | None = None) -> Iterator[torch.Tensor]:
        """Yield the waveform in pieces as soon as each is ready (CPU float tensors at 24 kHz).

        Each sentence is generated in one pass. The vocoder then streams: the first ``first`` mel frames (0.34 s), then
        ``chunk``-frame windows, each with the vocoder's ``context`` (32 frames for BigVGAN-v2; see
        :func:`drifting_tts.fast.stream_vocoder`). A vocoder without one (Griffin-Lim) yields each sentence whole."""
        from .fast import stream_vocoder

        spk, tempo = self._speaker(speaker)
        pause = resolve_pause(self.pause if pause is None else pause, self.speaker_id(speaker))
        g = torch.Generator(device=self.device).manual_seed(seed)
        rng, previous = random.Random(seed), None
        for sentence in split_sentences(normalize(text)):
            mel = self.stats.denormalize(self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo,
                                                   prosody_temperature=prosody_temperature))
            if previous is not None:
                yield silence(pause, previous, rng)
            previous = sentence
            for piece in stream_vocoder(self.vocoder, mel, first=first, chunk=chunk, context=self.vocoder.context,
                                        graphs=self.vocoder_graphs):
                yield piece.cpu()

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


def pipeline_args(args) -> dict:
    """Synthesizer arguments of ``--release`` / ``--model`` / ``--vocoder`` / ``--prosody`` / ``--pause`` (the
    explicit ones win over the release's)."""
    from .hub import RELEASES, hub_file

    rel = RELEASES[args.release] if getattr(args, "release", None) else {}
    model = args.model or (hub_file(rel["model"], f"release {args.release}") if rel else "runs/tts/model_ema.pt")
    prosody = args.prosody if args.prosody is not None else rel.get("prosody")
    pause = args.pause if args.pause is not None else rel.get("pause", 0.15)
    if getattr(args, "pause_policy", "fixed") == "punct":
        pause = "punct"
    if pause == "punct" and getattr(args, "pause_jitter", 0.0):
        pause = f"punct:{args.pause_jitter}"
    return {"model_path": model, "vocoder": args.vocoder or rel.get("vocoder"),
            "prosody": None if str(prosody).lower() == "none" else prosody, "pause": pause}


def run(args) -> None:
    synth = Synthesizer(device=args.device, cuda_kernel=args.cuda_kernel, prosody_temperature=args.prosody_temperature,
                        prosody_spread=args.prosody_spread, prosody_durations=args.prosody_durations,
                        **pipeline_args(args))
    if args.list_speakers:
        for name, v in VOICES.items():
            default = " (default)" if name == DEFAULT_VOICE else ""
            print(f"{name:7s} speaker {v['id']}, median pitch {v['pitch_hz']} Hz{default}")
        return
    temperature = synth.default_temperature if args.temperature is None else args.temperature
    kw = dict(speaker=args.speaker, cfg_scale=args.cfg, temperature=temperature,
              length_scale=args.length_scale, seed=args.seed, attn_window=args.attn_window,
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
