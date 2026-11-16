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

from .audio import HOP_LENGTH, SAMPLE_RATE
from .chunked import Chunking
from .data import MelStats
from .latents import VAE_BACKENDS
from .text import frontend, group_sentences, normalize, split_sentences, text_to_ids  # noqa: F401 (re-exported)
from .voices import DEFAULT_VOICE, VOICES, voice_id

DEFAULT_TEMPERATURE = 0.5  # noise temperature of a checkpoint without a stored one (lowest CER in the README sweep)
DEFAULT_CFG = {"dit": 2.0, "ar": 1.0}  # guidance of a checkpoint without a stored one: the DiT's operating point
#                                        (README), none for ARDrift (it does not help: docs/AR_DRIFT.md)
REFERENCE_MODE = "both"  # a recording's voice on a model trained with prompts and references (#95, #96)
AR_CONTEXT_FRAMES = 256  # ARDrift: frames of the audio before a unit that it continues from (2.7 s)
PARAGRAPH_CHARS = 180  # paragraph mode: characters per unit, within the CUDA graphs' buckets (fast.warmup)
CONTEXT_CHARS = 190  # left context + unit, within the 384-token bucket of the CUDA graphs
DURATION_SOURCES = ("sampled", "regressor", "pauses")


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
    p.add_argument("--ref-audio", default=None,
                   help="a recording whose voice is spoken instead of --speaker (an ARDrift model trained with "
                        "prompts or references: model.ar.prompt, model.ar.reference)")
    p.add_argument("--list-speakers", action="store_true", help="list the voices")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--out-dir", default="outputs/synth")
    p.add_argument("--cfg", type=float, default=None,
                   help="guidance scale alpha learned during training (default: the release's, the checkpoint's "
                        "preferred one, else 2 for the DiT and 1 for ARDrift)")
    p.add_argument("--temperature", type=float, default=None,
                   help=f"noise temperature (default: the release's, the checkpoint's preferred one, else "
                        f"{DEFAULT_TEMPERATURE})")
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
    p.add_argument("--prosody-duration-temperature", type=float, default=None,
                   help="noise temperature of the sampled durations apart from the pitch's (default: the "
                        "checkpoint's preferred one, else --prosody-temperature)")
    p.add_argument("--prosody-pitch-model", default=None,
                   help="a second prosody predictor that samples the token pitch (e.g. with word / sentence features) "
                        "at --prosody-temperature; the durations stay --prosody's")
    p.add_argument("--prosody-durations", choices=list(DURATION_SOURCES), default=None,
                   help="regressor: --prosody samples only the token pitch; durations stay the model's; pauses: it "
                        "also samples the durations between words (pauses, phrase breaks), the letters' stay the "
                        "model's (default: the release's, else sampled)")
    p.add_argument("--paragraph", type=int, nargs="?", const=PARAGRAPH_CHARS, default=None, metavar="CHARS",
                   help=f"paragraph mode: consecutive sentences in one pass, up to CHARS characters per unit "
                        f"(default {PARAGRAPH_CHARS}); the model renders the gaps between them. 0: sentence by "
                        f"sentence. Default: the release's, else on for ARDrift and off for the DiT")
    p.add_argument("--acoustic-context", type=int, default=None, metavar="FRAMES",
                   help=f"ARDrift: frames of the audio before a unit that it continues from (default "
                        f"{AR_CONTEXT_FRAMES}; 0: every unit starts from silence)")
    p.add_argument("--context", action="store_true",
                   help="condition each sentence (unit) on the text before it (text encoder and prosody predictor)")
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
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                   help="rate of the audio; 0: the codec's own (48 kHz) for a model trained on DAC-VAE / VoxCPM "
                        "latents (default: 24000)")
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


def resolve_pause(pause: Pause, speaker_id: int, edge: float | None = None,
                  context: bool = False) -> float | Callable[[str, random.Random], float]:
    """``pause``: seconds, a callable (e.g. a :class:`drifting_tts.prosody.PausePolicy`), or ``"punct"`` /
    ``"punct:<jitter>"``: the voice's measured policy (``PausePolicy.for_voice``, with the generated sentences' edge
    silence ``edge`` if known)."""
    if not isinstance(pause, str):
        return pause
    name, _, jitter = pause.partition(":")
    if name != "punct":
        raise ValueError(f"pause must be seconds, a callable or 'punct[:<jitter>]', got {pause!r}")
    from .prosody import PausePolicy

    return PausePolicy.for_voice(speaker_id, jitter=float(jitter or 0.0), edge=edge, context=context)


def silence(pause: float | Callable[[str, random.Random], float], sentence: str, rng: random.Random,
            sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """The silence after ``sentence``: ``pause`` seconds, or what a pause policy picks for it (e.g.
    :class:`drifting_tts.prosody.PausePolicy`, by its final punctuation)."""
    return torch.zeros(int((pause(sentence, rng) if callable(pause) else pause) * sample_rate))


def preferred_temperature(model) -> float:
    """The checkpoint's preferred noise temperature (``calibrate-durations --temperature``), else 0.5."""
    t = getattr(model, "temperature", None)
    return DEFAULT_TEMPERATURE if t is None else float(t)


def is_ar(model) -> bool:
    """Whether ``model`` is an ARDrift model (:class:`drifting_tts.models.ar.ARDriftTTS`), not the DiT."""
    return getattr(model, "generator", None) is None


def preferred_cfg(model) -> float:
    """The checkpoint's preferred guidance scale (``calibrate-durations --cfg``), else the architecture's
    (:data:`DEFAULT_CFG`)."""
    c = getattr(model, "cfg_scale", None)
    return DEFAULT_CFG["ar" if is_ar(model) else "dit"] if c is None else float(c)


class Reference:
    """A voice given by a recording (:meth:`Synthesizer.reference`): ``mel``, the normalised mel ``[1, n_mels, P]``
    of the recording's beginning (whole patches), which the model continues from (#95, ``model.ar.prompt``; ``None``:
    no prompt); ``tokens`` ``[1, K, dim]``, the reference encoder's reading of the recording, in front of the
    backbone's sequence (#96, ``model.ar.reference``; ``None``: the model's null tokens); and ``vector``, the speaker
    vector the text encoder gets in place of a table row."""

    def __init__(self, mel: torch.Tensor | None, vector: torch.Tensor, tokens: torch.Tensor | None = None):
        self.mel, self.vector, self.tokens = mel, vector, tokens


class AcousticContext:
    """ARDrift across the units of one call: the end of what was generated so far (the aligned condition and the
    normalised mel, the last ``frames`` frames, whole patches), given to the next unit as the prefix of
    :meth:`drifting_tts.models.ar.ARDriftTTS.generate`. The next unit then continues the voice and the intonation of
    the previous one, as the model continues a multi-sentence utterance in training, instead of starting from the
    beginning-of-sequence patch. ``limit``: frames of context and unit together (the longest training utterance).
    ``reference``: a voice prompt, kept in front of every unit's prefix with a zero condition of ``channels``
    channels (the model was given no text for it)."""

    def __init__(self, frames: int, patch: int, limit: int, reference: Reference | None = None, channels: int = 0):
        self.frames, self.patch, self.limit = frames // patch * patch, patch, limit
        self.cond: torch.Tensor | None = None
        self.mel: torch.Tensor | None = None
        self.reference, self.channels = reference, channels

    def prefix(self, unit_frames: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``(prefix_cond, prefix)`` for a unit of ``unit_frames`` frames, or ``(None, None)``."""
        ref = self.reference
        if ref is not None and ref.mel is None:  # a reference without a prompt: its tokens alone
            ref = None
        room = self.limit - unit_frames - (0 if ref is None else ref.mel.shape[-1])
        n = 0 if self.cond is None else max(min(self.cond.shape[-1], room), 0) // self.patch * self.patch
        cond, mel = (self.cond[..., -n:], self.mel[..., -n:]) if n > 0 else (None, None)
        if ref is None:
            return cond, mel
        zero = ref.mel.new_zeros(1, self.channels, ref.mel.shape[-1])
        return (zero, ref.mel) if cond is None else (torch.cat([zero, cond], -1), torch.cat([ref.mel, mel], -1))

    def update(self, cond: torch.Tensor, mel: torch.Tensor) -> None:
        """Append a generated unit (its condition ``[1, C, T]`` and normalised mel ``[1, n_mels, T]``)."""
        t = min(cond.shape[-1], mel.shape[-1]) // self.patch * self.patch  # whole patches: the tail frames are dropped
        if not t or not self.frames:
            return
        cond, mel = cond[..., :t], mel[..., :t]
        if self.cond is not None:
            cond, mel = torch.cat([self.cond, cond], -1), torch.cat([self.mel, mel], -1)
        self.cond, self.mel = cond[..., -self.frames:], mel[..., -self.frames:]


class Synthesizer:
    def __init__(self, model_path: str | Path, device: str = "cuda", vocoder: str | None = None,
                 cuda_kernel: bool = False, fast: bool = False, compile: bool = False, tf32: bool = False,
                 prosody: str | Path | None = None, prosody_temperature: float | None = None,
                 prosody_spread: float = 1.0, prosody_durations: str | None = None, pause: Pause = 0.15,
                 prosody_duration_temperature: float | None = None, prosody_pitch: str | Path | None = None,
                 paragraph: bool | int | None = None, context: bool = False, temperature: float | None = None,
                 cfg_scale: float | None = None, acoustic_context: bool | int | None = None,
                 sample_rate: int | None = SAMPLE_RATE):
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
        downloaded) or a ``train-prosody`` checkpoint (see :meth:`set_prosody`); ``prosody_pitch``: a second one
        that samples only the token pitch.
        ``pause``: the default silence between sentences of :meth:`__call__` / :meth:`stream` (seconds, a
        :class:`~drifting_tts.prosody.PausePolicy`, or ``"punct"``: by punctuation, measured per voice).
        ``paragraph``: synthesise consecutive sentences together, up to :data:`PARAGRAPH_CHARS` characters (or the
        given number) per pass (:func:`drifting_tts.text.group_sentences`); the pauses then only separate the units.
        ``None``: on for an ARDrift model (it is trained on multi-sentence utterances and reads them better in one
        pass, docs/AR_DRIFT.md), off for the DiT.
        ``temperature`` / ``cfg_scale``: the operating point of the calls that do not give one (``default_temperature``
        / ``default_cfg``). ``None``: the checkpoint's (``calibrate-durations --temperature / --cfg``), else 0.5 and
        the architecture's guidance (:data:`DEFAULT_CFG`: 2 for the DiT, 1 for ARDrift).
        ``acoustic_context`` (ARDrift): a unit after the first continues from the last frames generated before it
        (:class:`AcousticContext`), instead of starting from silence: ``None`` / ``True``
        :data:`AR_CONTEXT_FRAMES` frames, a number that many, ``False`` / 0 none. The DiT has none.
        ``sample_rate``: the rate of the audio (``self.sample_rate``). A model trained on VAE latents can return its
        decoder's own rate with ``None`` (48 kHz for DAC-VAE and VoxCPM2) instead of that output resampled to
        24 kHz. The decoders fine-tuned by ``finetune-vocoder`` were trained on 24 kHz losses and are meant for the
        default; the mel vocoders are 24 kHz.
        ``context``: each sentence (unit) is conditioned on the text before it: the end of the previous one, whole
        words, up to :data:`CONTEXT_CHARS` characters with the current one, runs through the text encoder and the
        prosody predictor with it, and only the current one is generated (:meth:`DriftingTTS.synthesize`).
        :meth:`from_pretrained` builds a published release (v3.2: ``vocos-v2``, ``prosody="drift"``,
        ``pause="punct"``, T 0.3, guidance 2)."""
        from .train import load_tts
        from .vocoder import load_vocoder

        self.model, self.cfg, stats = load_tts(model_path, device)
        self.language = self.cfg.model.text.get("language", "tr")
        self.text = frontend(self.language)  # normalisation and symbols of the model's language
        self.stats = MelStats(stats["mean"], stats["std"])
        self.backend = stats.get("backend", "vocos")
        if self.backend in VAE_BACKENDS:  # a model trained on VAE latents: the VAE decoder is the vocoder
            from .latents.vocoder import LatentVocoder

            if vocoder is not None and not Path(vocoder).is_file():
                raise ValueError(f"a model trained on {self.backend} latents decodes them with its VAE decoder: "
                                 f"vocoder must be a fine-tuned decoder checkpoint (decoder_ft.pt), not {vocoder!r}")
            self.vocoder = LatentVocoder(self.backend, device, repeat=stats.get("latent_repeat", 1), decoder=vocoder,
                                         sample_rate=sample_rate)
            self.sample_rate = self.vocoder.sample_rate
        else:
            if sample_rate not in (None, SAMPLE_RATE):
                raise ValueError(f"the {self.backend} vocoders make {SAMPLE_RATE} Hz audio, not {sample_rate} Hz")
            self.vocoder = load_vocoder(vocoder, device, cuda_kernel=cuda_kernel, backend=self.backend)
            self.sample_rate = SAMPLE_RATE
        if self.vocoder.mel != self.backend:
            raise ValueError(f"vocoder {self.vocoder.name!r} expects {self.vocoder.mel} mels, but the model produces "
                             f"{self.backend} mels")
        self.device = device
        self.ar = is_ar(self.model)
        self.default_temperature = preferred_temperature(self.model) if temperature is None else float(temperature)
        self.default_cfg = preferred_cfg(self.model) if cfg_scale is None else float(cfg_scale)
        if acoustic_context and not self.ar:
            raise ValueError("acoustic_context is ARDrift's: the DiT generates each unit from noise alone")
        if acoustic_context is None or acoustic_context is True:
            acoustic_context = AR_CONTEXT_FRAMES if self.ar else 0
        self.acoustic_context = int(acoustic_context)
        spk_file = Path(self.cfg.data.root) / "speakers.json"
        self.speakers = json.loads(spk_file.read_text()) if spk_file.exists() else {}
        self.acoustic, self.vocoder_graphs, self.prosody = None, None, None
        self.fast, self._graph_opts = fast, {"compile": compile, "tf32": tf32}
        resolve_pause(pause, 0)  # validate
        self.pause = pause
        self.paragraph, self.context = self.ar if paragraph is None else paragraph, context
        self.set_prosody(prosody, prosody_temperature, prosody_spread, prosody_durations, prosody_duration_temperature,
                         prosody_pitch)
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
                prosody_spread: float = 1.0, prosody_durations: str | None = None,
                prosody_duration_temperature: float | None = None,
                prosody_pitch: str | Path | None = None, paragraph: bool | int = _KEEP,
                context: bool = _KEEP, temperature: float = _KEEP, cfg_scale: float = _KEEP,
                acoustic_context: bool | int = _KEEP) -> Synthesizer:
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
            other.set_prosody(prosody, prosody_temperature, prosody_spread, prosody_durations,
                              prosody_duration_temperature, prosody_pitch)
        if pause is not _KEEP:
            resolve_pause(pause, 0)
            other.pause = pause
        if paragraph is not _KEEP:
            other.paragraph = paragraph
        if context is not _KEEP:
            other.context = context
        if temperature is not _KEEP:
            other.default_temperature = float(temperature)
        if cfg_scale is not _KEEP:
            other.default_cfg = float(cfg_scale)
        if acoustic_context is not _KEEP:
            if acoustic_context and not self.ar:
                raise ValueError("acoustic_context is ARDrift's")
            other.acoustic_context = AR_CONTEXT_FRAMES if acoustic_context is True else int(acoustic_context)
        return other

    def operating_point(self, cfg_scale: float | None = None, temperature: float | None = None) -> tuple[float, float]:
        """``(cfg_scale, temperature)`` of a call: the given ones, else the Synthesizer's defaults."""
        return (self.default_cfg if cfg_scale is None else float(cfg_scale),
                self.default_temperature if temperature is None else float(temperature))

    def _history(self, acoustic_context: bool | int | None = None,
                 speaker: str | int | Reference | None = None) -> AcousticContext | None:
        """The acoustic context of one call (``acoustic_context``: this call's, default the Synthesizer's), or
        ``None``: the DiT, or switched off. A :class:`Reference` as the speaker is carried by it: its mel is the
        prompt in front of every unit."""
        n = self.acoustic_context if acoustic_context is None else \
            AR_CONTEXT_FRAMES if acoustic_context is True else int(acoustic_context)
        ref = speaker if isinstance(speaker, Reference) else None
        if not self.ar or (n <= 0 and ref is None):
            return None
        return AcousticContext(max(n, 0), self.model.patch, int(self.cfg.data.get("max_frames", 1500)), ref,
                               self.model.n_mels + self.model.encoder.d)

    @torch.no_grad()
    def reference(self, audio: str | Path | torch.Tensor, sample_rate: int | None = None,
                  seconds: float = 6.0, mode: str | None = None, reference_seconds: float = 10.0) -> Reference:
        """A voice from a recording: a file, or a mono waveform at ``sample_rate`` (default 24 kHz), prepared as the
        training data was (silence trimmed, loudness normalised); no transcript is needed. Pass the result as
        ``speaker`` (or the recording as ``ref_audio``). ``mode``: how the model gets the voice:

        * ``"prefix"`` (#95, ``model.ar.prompt``): the first ``seconds`` are a prompt the model continues from;
        * ``"reference"`` (#96, ``model.ar.reference``): the first ``reference_seconds`` go through the reference
          encoder;
        * ``"both"``: the two together. The default: what the model was trained for (:data:`REFERENCE_MODE`)."""
        m = self.model
        ways = [w for w, on in (("prefix", self.ar and m.prompted), ("reference", self.ar and m.ref_enc is not None))
                if on]
        if not ways:
            raise ValueError("a reference voice needs an ARDrift model trained with prompts or references "
                             "(model.ar.prompt, model.ar.reference)")
        mode = mode or (REFERENCE_MODE if len(ways) == 2 else ways[0])
        if mode not in ("prefix", "reference", "both") or any(w not in ways for w in
                                                              (("prefix", "reference") if mode == "both" else (mode,))):
            raise ValueError(f"reference mode {mode!r}: this model was trained for {' and '.join(ways)}")
        mel = tokens = vector = None
        if mode != "reference":
            mel = self.reference_mel(audio, sample_rate, seconds)
            vector = m.prompt_vector(mel)
        if mode != "prefix":  # with both, the text encoder's voice is the reference's, as in training
            tokens, vector = m.reference_tokens(self.reference_mel(audio, sample_rate, reference_seconds))
        return Reference(mel, vector, tokens)

    def reference_mel(self, audio: str | Path | torch.Tensor, sample_rate: int | None = None,
                      seconds: float = 6.0) -> torch.Tensor:
        """The normalised mel ``[1, n_mels, P]`` of a recording's first ``seconds`` (whole patches of an ARDrift
        model), prepared as the training data was."""
        from .audio import make_logmel, normalize_loudness, resample, trim_silence

        if isinstance(audio, (str, Path)):
            wav, sr = sf.read(str(audio), dtype="float32", always_2d=True)
            wav = wav.mean(1)
        else:
            wav, sr = torch.as_tensor(audio).float().flatten().cpu().numpy(), sample_rate or SAMPLE_RATE
        wav = resample(torch.from_numpy(normalize_loudness(trim_silence(wav, sr))), sr, SAMPLE_RATE)
        mel = self.stats.normalize(make_logmel(self.backend)(wav))
        k = getattr(self.model, "patch", 1)
        frames = min(mel.shape[-1], int(seconds * SAMPLE_RATE / HOP_LENGTH)) // k * k
        if frames < k:
            raise ValueError("the reference recording is too short")
        return mel[None, :, :frames].to(self.device)

    def _ar_generate(self, cond: torch.Tensor, spk: torch.Tensor, temperature: float, g: torch.Generator,
                     cfg_scale: float, history: AcousticContext | None) -> torch.Tensor:
        """ARDrift: the normalised mel of one unit's aligned condition, continuing from ``history`` (then updated)."""
        m = self.model
        pc, pm = (None, None) if history is None else history.prefix(cond.shape[-1])
        tokens = getattr(getattr(history, "reference", None), "tokens", None)  # a reference's (#96)
        hn = float(getattr(self, "history_noise", 0.0))  # noise on the model's own history (#101; default none)
        guide = getattr(self, "guidance", "text")  # what the null branch of cfg_scale is without (#99)
        if m.fast_generator is not None:
            mel = m.fast_generator(cond, spk, temperature, g, cfg_scale, prefix=pm, prefix_cond=pc, tokens=tokens,
                                   history_noise=hn, guide=guide)
        else:
            mel = m.generate(cond, spk, temperature, g, prefix=pm, cfg_scale=cfg_scale, prefix_cond=pc,
                             tokens=tokens, history_noise=hn, guide=guide)
        if history is not None:
            history.update(cond, mel)
        return mel

    @staticmethod
    def with_context(sentence: str, previous: str | None, language: str = "tr") -> tuple[list[int], int]:
        """Token ids of the end of ``previous`` (whole words, so that both fit in :data:`CONTEXT_CHARS`), a space and
        ``sentence``, and the number of leading context tokens (``0``: no context)."""
        to_ids = frontend(language).text_to_ids
        room = CONTEXT_CHARS - len(sentence) - 1
        ctx = (previous or "")[-room:] if room > 0 else ""
        if previous and len(ctx) < len(previous):  # cut at a word boundary
            ctx = ctx.partition(" ")[2]
        if not ctx.strip():
            return to_ids(sentence, normalized=True), 0
        return to_ids(f"{ctx} {sentence}", normalized=True), 2 * len(ctx) + 2

    def units(self, text: str, paragraph: bool | int | None = None) -> list[str]:
        """The normalised pieces of ``text`` that are synthesised in one pass each: its sentences, or with
        ``paragraph`` (default: the Synthesizer's) groups of consecutive sentences (:func:`group_sentences`)."""
        sentences = split_sentences(self.text.normalize(text))
        paragraph = self.paragraph if paragraph is None else paragraph
        if not paragraph:
            return sentences
        return group_sentences(sentences, PARAGRAPH_CHARS if paragraph is True else int(paragraph))

    def _units(self, text: str, paragraph: bool | int | None, context: bool | None) -> Iterator[tuple[str, str | None]]:
        """:meth:`units` with the unit before each one when ``context`` (default: the Synthesizer's) is on."""
        use = self.context if context is None else context
        previous = None
        for unit in self.units(text, paragraph):
            yield unit, previous if use else None
            previous = unit

    def set_prosody(self, prosody=None, temperature: float | None = None, spread: float = 1.0,
                    durations: str | None = None, duration_temperature: float | None = None, pitch=None) -> None:
        """Where the durations and token pitch come from. ``prosody``: ``None`` (the model's deterministic
        regressors), a :class:`~drifting_tts.models.prosody_net.ProsodyPredictor`, a name of
        :data:`drifting_tts.hub.PROSODY_MODELS` (``"drift"``) or a ``train-prosody`` checkpoint.

        The predictor samples with noise temperature ``temperature`` (``None``: the checkpoint's preferred one, else
        1) and output-space temperature ``spread`` (:meth:`ProsodyPredictor.sample`); the seed of each call drives
        it, before the DiT's noise. ``durations="regressor"`` keeps the model's durations (and per-voice factors) and
        samples only the token pitch; ``"pauses"`` also samples the durations of the tokens between words (pauses and
        phrase breaks, at the predictor's own factors) and keeps the model's for the letters; ``"sampled"`` takes
        every duration from the predictor, whose own per-voice duration factors replace the model's. ``None``: the
        predictor's preferred source (``calibrate.durations`` of ``train-prosody``), else ``"sampled"``.
        ``duration_temperature``: the noise temperature of the sampled durations apart from the pitch's
        (``ProsodyPredictor.sample``; ``None``: the checkpoint's preferred one, else the prosody temperature of each
        call). A voice listed in the predictor's ``rhythm`` table samples its durations as that speaker (its
        rhythm, scaled to the voice's rate by ``duration_scales``) and its pitch as itself.
        ``pitch``: a second predictor (same kinds of argument) that samples the token pitch at ``temperature`` /
        ``spread`` (e.g. one with word / sentence features), drawn after ``prosody``'s noise; the durations stay
        ``prosody``'s (needs ``prosody``). With ``fast``, a one-pass predictor (``drift`` / ``mse``, spread 1, no
        word or sentence features, no second pitch predictor) runs in the acoustic model's CUDA graphs (recaptured
        here); others run eagerly."""
        if durations is not None and durations not in DURATION_SOURCES:
            raise ValueError(f"prosody_durations must be one of {DURATION_SOURCES}, got {durations!r}")
        if pitch is not None and prosody is None:
            raise ValueError("a second pitch predictor needs a prosody predictor for the durations")
        self.prosody, self.prosody_temperature = None, temperature
        self.prosody_spread, self.prosody_durations = spread, durations or "sampled"
        self.prosody_duration_temperature = duration_temperature
        self.prosody_pitch = None
        if prosody is not None:
            from .models.prosody_net import ProsodyPredictor

            def load(p):
                return p if isinstance(p, ProsodyPredictor) else ProsodyPredictor.load(p, self.device, tts=self.model)

            self.prosody = load(prosody)
            if durations is None:  # the predictor's preferred source (calibrate.durations), else all sampled
                self.prosody_durations = self.prosody.durations or "sampled"
            if temperature is None:
                self.prosody_temperature = 1.0 if self.prosody.temperature is None else self.prosody.temperature
            if duration_temperature is None:
                self.prosody_duration_temperature = self.prosody.duration_temperature
            if pitch is not None:
                self.prosody_pitch = load(pitch)
        if self.fast:
            self._capture_acoustic()

    def _capture_acoustic(self) -> None:
        from .fast import GraphedAcoustic, graphable

        self.acoustic = None
        if getattr(self.model, "generator", None) is None:  # an ARDrift model: one CUDA graph per generation step
            self.model.enable_fast()
            return
        if not graphable(self.prosody, self.prosody_spread) or self.prosody_pitch is not None:
            warnings.warn("this prosody predictor runs eagerly: only the vocoder windows use CUDA graphs", stacklevel=3)
            return
        self.acoustic = GraphedAcoustic(self.model, prosody=self.prosody, prosody_durations=self.prosody_durations,
                                        duration_row=self._duration_row(), **self._graph_opts)
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

    def _duration_row(self) -> bool:
        """Whether the sampled durations need their own row in the prosody predictor's batch: a duration
        temperature of their own or a voice that borrows another speaker's rhythm."""
        return (self.prosody is not None and self.prosody_durations == "sampled"
                and (self.prosody_duration_temperature is not None or bool(self.prosody.rhythm)))

    def _rhythm(self, speaker: str | int) -> tuple[torch.Tensor, float] | None:
        """The speaker whose rhythm this voice's sampled durations borrow (``ProsodyPredictor.rhythm``) and the ratio of
        the voice's regressor factor to its sampler factor (the sentence's edges keep its regressor durations), or
        ``None``."""
        if self.prosody is None or self.prosody_durations != "sampled":
            return None
        spk_id = self.speaker_id(speaker)
        src = self.prosody.rhythm.get(spk_id)
        if src is None or src == spk_id:
            return None
        own = getattr(self.model, "duration_scales", {}).get(spk_id, self.model.duration_scale)
        return torch.tensor([src], device=self.device), own / self.prosody.scale(spk_id)

    def _voice(self, speaker, ref_audio):
        """The speaker of a call: ``ref_audio`` as a :class:`Reference` when given, else ``speaker``."""
        if ref_audio is None:
            return speaker
        return ref_audio if isinstance(ref_audio, Reference) else self.reference(ref_audio)

    def speaker_id(self, speaker: str | int | Reference) -> int:
        """A voice name (``male`` / ``female``), a training speaker ID, or a dataset speaker name; the prompted row
        for a :class:`Reference` (its durations and pauses are then the pooled defaults)."""
        if isinstance(speaker, Reference):
            return self.model.prompt_row
        if speaker in self.speakers and speaker not in VOICES:
            return self.speakers[speaker]
        return voice_id(speaker)

    def _speaker(self, speaker: str | int) -> tuple[torch.Tensor, float]:
        """Speaker id tensor and the voice's duration factor."""
        spk_id = self.speaker_id(speaker)
        if spk_id >= self.model.encoder.spk.num_embeddings:
            raise ValueError(f"speaker {speaker!r} ({spk_id}) is not in this checkpoint")
        if self.prosody is not None and self.prosody_durations in ("sampled", "pauses"):  # the sampler's own factors
            tempo = self.prosody.scale(spk_id)
        else:
            tempo = getattr(self.model, "duration_scales", {}).get(spk_id, self.model.duration_scale)
        return torch.tensor([spk_id], device=self.device), tempo

    def _regressor_scale(self, spk: torch.Tensor, length_scale: float) -> float | None:
        """``prosody_durations="pauses"``: the letters' scale (the model's factor of the voice) for a call at
        ``length_scale`` (which carries the sampler's factor, :meth:`_speaker`); else ``None``."""
        if self.prosody is None or self.prosody_durations != "pauses":
            return None
        spk_id = int(spk[0])
        own = getattr(self.model, "duration_scales", {}).get(spk_id, self.model.duration_scale)
        return length_scale * own / self.prosody.scale(spk_id)

    def _sample_prosody(self, ids: torch.Tensor, spk: torch.Tensor, g: torch.Generator, length_scale: float,
                        pt: float | None, dt: float | None, rs: torch.Tensor | None, edge_scale: float | None
                        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The prosody predictor's frames per token and token pitch for one unit (``None`` for what the model's own
        regressors give), drawn from ``g`` before the acoustic model's noise. ``pt`` / ``dt``: the prosody and the
        letters' duration temperature; ``rs`` / ``edge_scale``: :meth:`_rhythm`."""
        durations = pitch = None
        prompted = self.ar and (self.model.prompted or self.model.ref_enc is not None) \
            and int(spk[0]) == self.model.prompt_row
        if self.prosody is None or prompted:  # a prosody predictor does not know a voice given by a prompt
            return durations, pitch
        ids_len = torch.tensor([ids.shape[1]], device=self.device)
        durations, pitch = self.prosody.predict(self.model, ids, ids_len, spk, pt, length_scale, generator=g,
                                                spread=self.prosody_spread, duration_temperature=dt,
                                                duration_speaker=rs, edge_scale=edge_scale,
                                                regressor_scale=self._regressor_scale(spk, length_scale))
        if self.prosody_pitch is not None:  # the token pitch from the second predictor, drawn next
            _, pitch = self.prosody_pitch.predict(self.model, ids, ids_len, spk, pt, length_scale, generator=g,
                                                  spread=self.prosody_spread)
        if self.prosody_durations == "regressor":
            durations = None
        if self.ar:  # a duration-only predictor (#100): the intonation is the model's own
            pitch = None
        return durations, pitch

    def _mel(self, sentence: str, spk: torch.Tensor, g: torch.Generator, cfg_scale: float, temperature: float,
             length_scale: float, attn_window: int | None = None, pitch_shift: float = 0.0,
             steps: int | None = None, prosody_temperature: float | None = None,
             rhythm: tuple[torch.Tensor, float] | None = None, previous: str | None = None,
             history: AcousticContext | None = None) -> torch.Tensor:
        """One normalised sentence -> normalised mel ``[1, n_mels, T]`` (CUDA graphs when ``fast`` allows it).
        ``rhythm``: the speaker whose durations are sampled and the edge factor (:meth:`_rhythm`); ``previous``: the
        text before it, as left context (:meth:`with_context`); ``history`` (ARDrift): the audio before it
        (:meth:`_history`)."""
        ids, ctx = self.with_context(sentence, previous, self.language)
        ids = torch.tensor([ids], device=self.device)
        pt = self.prosody_temperature if prosody_temperature is None else prosody_temperature
        dt = self.prosody_duration_temperature if self.prosody_durations == "sampled" else None
        if dt is None and self._duration_row():  # the same row layout as the CUDA graphs
            dt = pt
        rs, edge_scale = (None, None) if rhythm is None else (rhythm[0], length_scale * rhythm[1])
        if self.acoustic is not None and attn_window is None and not pitch_shift and steps in (None, 1):
            return self.acoustic(ids, spk, cfg_scale, temperature, length_scale, generator=g,
                                 prosody_temperature=1.0 if pt is None else pt, duration_temperature=dt,
                                 rhythm=rs, edge_scale=edge_scale,
                                 regressor_scale=self._regressor_scale(spk, length_scale), context=ctx)
        ids_len = torch.tensor([ids.shape[1]], device=self.device)
        # sampled first, from the same generator as the acoustic model's noise
        durations, pitch = self._sample_prosody(ids, spk, g, length_scale, pt, dt, rs, edge_scale)
        if self.ar:  # ARDriftTTS.synthesize, with the acoustic context of the units before
            ref = getattr(history, "reference", None)  # a voice prompt: the text encoder's speaker vector
            cond, _ = self.model.aligned_condition(ids, ids_len, spk, length_scale, pitch_shift, durations, pitch, ctx,
                                                   None if ref is None else ref.vector)
            return self._ar_generate(cond, spk, temperature, g, cfg_scale, history)
        mel, _ = self.model.synthesize(ids, ids_len, spk, cfg_scale=cfg_scale, temperature=temperature,
                                       length_scale=length_scale, generator=g, attn_window=attn_window,
                                       pitch_shift=pitch_shift, steps=steps, durations=durations, pitch=pitch,
                                       context=ctx)
        return mel

    def _prepare(self, sentence: str, spk: torch.Tensor, g: torch.Generator, cfg_scale: float, temperature: float,
                 length_scale: float, prosody_temperature: float | None = None,
                 rhythm: tuple[torch.Tensor, float] | None = None, previous: str | None = None
                 ) -> tuple[torch.Tensor, ...]:
        """:meth:`_mel` without the DiT, for :mod:`drifting_tts.chunked`: the noise scaled by ``temperature``
        ``[1, n_mels, T]``, the aligned condition ``[1, C, T]``, the CFG scale ``[1]`` and the style codes, with the
        same draws from ``g`` (CUDA graphs for the encoder when ``fast`` allows it)."""
        ids, ctx = self.with_context(sentence, previous, self.language)
        ids = torch.tensor([ids], device=self.device)
        pt = self.prosody_temperature if prosody_temperature is None else prosody_temperature
        dt = self.prosody_duration_temperature if self.prosody_durations == "sampled" else None
        if dt is None and self._duration_row():
            dt = pt
        rs, edge_scale = (None, None) if rhythm is None else (rhythm[0], length_scale * rhythm[1])
        if self.acoustic is not None:
            return self.acoustic.prepare(ids, spk, cfg_scale, temperature, length_scale, generator=g,
                                         prosody_temperature=1.0 if pt is None else pt, duration_temperature=dt,
                                         rhythm=rs, edge_scale=edge_scale,
                                         regressor_scale=self._regressor_scale(spk, length_scale), context=ctx)
        ids_len = torch.tensor([ids.shape[1]], device=self.device)
        durations, pitch = self._sample_prosody(ids, spk, g, length_scale, pt, dt, rs, edge_scale)  # as _mel
        cond, _ = self.model.aligned_condition(ids, ids_len, spk, length_scale, durations=durations, pitch=pitch,
                                               context=ctx)
        z = torch.randn(1, self.model.n_mels, cond.shape[-1], device=self.device, generator=g)
        labels = self.model.style_codes(1, self.device, g)
        return z * temperature, cond, torch.full((1,), float(cfg_scale), device=self.device), labels

    def _prepare_philox(self, sentence: str, spk: torch.Tensor, seed: int, k: int, cfg_scale: float,
                        temperature: float, length_scale: float,
                        prosody_temperature: float | None = None) -> tuple:
        """:meth:`_prepare` with the counter-based noise of :mod:`drifting_tts.noise` (sentence ``k`` of the request
        with seed ``seed``), as :func:`drifting_tts.batched.stream_batched` draws it: the DiT's noise is a
        :class:`~drifting_tts.noise.DiTNoise` (drawn window by window)."""
        from .batched import batchable, prepare_batch
        from .chunked import to_device
        from .noise import DiTNoise, Philox, philox_normal, philox_randint

        if not batchable(self):
            raise NotImplementedError("counter-based noise supports the prosody sources of batched synthesis")
        ids = self.text.text_to_ids(sentence, normalized=True)
        keys = Philox([seed], [k])
        pt = self.prosody_temperature if prosody_temperature is None else prosody_temperature
        pt = 1.0 if pt is None else pt
        if self.acoustic is None:  # the batched path's own text pass, one row
            z, cond, _, alpha, labels, _ = prepare_batch(self.model, [ids], int(spk[0]), cfg_scale, temperature,
                                                         length_scale, keys, self.prosody, pt, self.prosody_durations)
            return z, cond, alpha, labels
        seeds, sentences = keys.tensors(self.device)
        noise = None
        if self.prosody is not None and self.prosody.kind == "drift":
            net = self.prosody.net
            noise = (philox_normal(seeds, sentences, "prosody_tok", net.noise_tok, len(ids)),
                     philox_normal(seeds, sentences, "prosody_glob", net.noise_glob, 1)[:, :, 0])
        else:  # no prosody noise to draw: prepare draws nothing when given an empty one
            noise = (torch.zeros(1, 0, len(ids), device=self.device), torch.zeros(1, 0, device=self.device))
        _, cond, alpha, _ = self.acoustic.prepare(torch.tensor([ids], device=self.device), spk, cfg_scale,
                                                  temperature, length_scale, prosody_temperature=pt,
                                                  prosody_noise=noise)
        gen = self.model.generator
        labels = philox_randint(seeds, sentences, "style", max(1, gen.noise_coords), gen.noise_classes)
        z = DiTNoise(keys, self.model.n_mels, to_device([cond.shape[-1]], self.device), temperature, cond.shape[-1])
        return z, cond, alpha, labels

    def _window_generator(self):
        """``generate(z, cond, spk, alpha, mask, labels)`` of :class:`~drifting_tts.chunked.WindowedMel` for one
        request: the acoustic model's CUDA graphs when ``fast``, else the eager rollout."""
        if self.acoustic is not None:
            return self.acoustic.generate_window
        model = self.model

        def generate(z, cond, spk, alpha, mask, labels):
            return model.rollout(z, cond, spk, alpha, model.generator.num_steps, mask=mask, noise_labels=labels)

        return generate

    @torch.no_grad()
    def mels(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float | None = None,
             temperature: float | None = None, length_scale: float = 1.0, seed: int = 0,
             prosody_temperature: float | None = None, paragraph: bool | int | None = None,
             context: bool | None = None, acoustic_context: bool | int | None = None,
             ref_audio: str | Path | torch.Tensor | Reference | None = None) -> list[torch.Tensor]:
        """The unnormalised log-mel ``[1, n_mels, T]`` of each sentence (each unit, :meth:`units`), with the same
        draws as :meth:`__call__` (whose waveform joins their vocoded sentences with ``pause`` seconds of silence)."""
        speaker = self._voice(speaker, ref_audio)
        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        cfg_scale, temperature = self.operating_point(cfg_scale, temperature)
        g = torch.Generator(device=self.device).manual_seed(seed)
        history = self._history(acoustic_context, speaker)
        return [self.stats.denormalize(self._mel(s, spk, g, cfg_scale, temperature, length_scale * tempo,
                                                 prosody_temperature=prosody_temperature, rhythm=rhythm,
                                                 previous=prev, history=history))
                for s, prev in self._units(text, paragraph, context)]

    @torch.no_grad()
    def __call__(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float | None = None,
                 temperature: float | None = None, length_scale: float = 1.0, seed: int = 0,
                 pause: Pause | None = None, attn_window: int | None = None, pitch_shift: float = 0.0,
                 steps: int | None = None, prosody_temperature: float | None = None,
                 paragraph: bool | int | None = None, context: bool | None = None,
                 acoustic_context: bool | int | None = None,
                 ref_audio: str | Path | torch.Tensor | Reference | None = None) -> tuple[torch.Tensor, dict]:
        """``text`` -> waveform (CPU float tensor at ``self.sample_rate``) and timings. Sentence by sentence (or
        ``paragraph`` by paragraph, :meth:`units`), each in one pass and vocoded whole, joined by ``pause`` (default:
        the Synthesizer's; seconds, a callable or ``"punct"``). ``cfg_scale`` / ``temperature``: default the
        Synthesizer's operating point (:meth:`operating_point`: the checkpoint's or the release's, not 1 / 1).
        ``prosody_temperature``: this call's noise temperature of the prosody predictor (default: the
        Synthesizer's); ``context`` / ``acoustic_context``: see :class:`Synthesizer`. ``ref_audio`` (an ARDrift
        model trained with prompts, #95): a recording, or a :class:`Reference` made from one, whose voice is
        spoken instead of ``speaker``'s."""
        speaker = self._voice(speaker, ref_audio)
        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        pause = self._pause(pause, speaker, context)
        cfg_scale, temperature = self.operating_point(cfg_scale, temperature)
        g = torch.Generator(device=self.device).manual_seed(seed)
        history = self._history(acoustic_context, speaker)
        wavs, t_acoustic, t_vocoder = [], 0.0, 0.0
        rng = random.Random(seed)  # pause jitter only: the acoustic draws stay those of ``g``
        for sentence, prev in self._units(text, paragraph, context):
            self._sync()
            t0 = time.perf_counter()
            mel = self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo, attn_window,
                            pitch_shift, steps, prosody_temperature, rhythm, prev, history)
            self._sync()
            t1 = time.perf_counter()
            wav = self.vocoder(self.stats.denormalize(mel))[0].cpu()
            self._sync()
            t_acoustic += t1 - t0
            t_vocoder += time.perf_counter() - t1
            wavs += [wav, silence(pause, sentence, rng, self.sample_rate)]
        wav = torch.cat(wavs[:-1]) if wavs else torch.zeros(0)
        dur = max(wav.numel() / self.sample_rate, 1e-6)
        return wav, {"seconds": dur, "rtf_acoustic": t_acoustic / dur, "rtf_total": (t_acoustic + t_vocoder) / dur}

    @torch.no_grad()
    def stream(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float | None = None,
               temperature: float | None = None, length_scale: float = 1.0, seed: int = 0,
               pause: Pause | None = None, first: int = 32, chunk: int = 256,
               prosody_temperature: float | None = None, chunked: Chunking | bool | None = None,
               paragraph: bool | int | None = None, context: bool | None = None,
               acoustic_context: bool | int | None = None,
               ref_audio: str | Path | torch.Tensor | Reference | None = None) -> Iterator[torch.Tensor]:
        """Yield the waveform in pieces as soon as each is ready (CPU float tensors at ``self.sample_rate``); the
        operating point and the options are those of :meth:`__call__`.

        Each sentence (each unit, :meth:`units`) is generated in one pass. The vocoder then streams: the first
        ``first`` mel frames (0.34 s), then ``chunk``-frame windows, each with the vocoder's ``context`` (32 frames
        for BigVGAN-v2; see
        :func:`drifting_tts.fast.stream_vocoder`). A vocoder without one (Griffin-Lim) yields each sentence whole.

        ``chunked`` (a :class:`~drifting_tts.chunked.Chunking`, or ``True`` for its defaults): the DiT streams too.
        It runs on windows of the sentence (the first one ``first`` + the vocoder's context + ``right`` frames long),
        so the first piece no longer waits for the whole sentence's mel; ``chunk`` is then ``chunked.chunk``. The
        output is another sample of the same model, not the same audio (docs/LATENCY.md)."""
        cfg_scale, temperature = self.operating_point(cfg_scale, temperature)
        speaker = self._voice(speaker, ref_audio)
        if self.ar:  # ARDrift: generated step by step, streamed as it goes
            yield from self._stream_ar(text, speaker, cfg_scale, temperature, length_scale, seed, pause, first, chunk,
                                       paragraph, context, acoustic_context, prosody_temperature)
            return
        if chunked:
            yield from self._stream_chunked(text, speaker, cfg_scale, temperature, length_scale, seed, pause, first,
                                            prosody_temperature, Chunking() if chunked is True else chunked,
                                            paragraph, context)
            return
        from .fast import stream_vocoder

        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        pause = self._pause(pause, speaker, context)
        g = torch.Generator(device=self.device).manual_seed(seed)
        rng, previous = random.Random(seed), None
        for sentence, prev in self._units(text, paragraph, context):
            mel = self.stats.denormalize(self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo,
                                                   prosody_temperature=prosody_temperature, rhythm=rhythm,
                                                   previous=prev))
            if previous is not None:
                yield silence(pause, previous, rng, self.sample_rate)
            previous = sentence
            for piece in stream_vocoder(self.vocoder, mel, first=first, chunk=chunk, context=self.vocoder.context,
                                        graphs=self.vocoder_graphs):
                yield piece.cpu()

    def _stream_ar(self, text: str, speaker: str | int, cfg_scale: float, temperature: float, length_scale: float,
                   seed: int, pause: Pause | None, first: int, chunk: int, paragraph: bool | int | None,
                   context: bool | None, acoustic_context: bool | int | None = None,
                   prosody_temperature: float | None = None) -> Iterator[torch.Tensor]:
        """:meth:`stream` for an ARDrift model: each vocoder window runs as soon as the steps that cover it and its
        right context are generated, so the first piece waits for ``first`` + context frames, not the sentence (a
        unit after the first also for the steps of its acoustic context, which only fill the caches). The same draws
        as :meth:`__call__` (the audio matches it up to the vocoder's streaming windows)."""
        from .batched import stream_windows
        from .fast import stream_vocoder, vocode_window
        from .models.ar import ARGenerator

        m = self.model
        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        rs, edge = (None, None) if rhythm is None else (rhythm[0], length_scale * tempo * rhythm[1])
        pt = self.prosody_temperature if prosody_temperature is None else prosody_temperature
        dt = self.prosody_duration_temperature if self.prosody_durations == "sampled" else None
        if dt is None and self._duration_row():  # as _mel
            dt = pt
        pause = self._pause(pause, speaker, context)
        g = torch.Generator(device=self.device).manual_seed(seed)
        gen = m.fast_generator or ARGenerator(m, graphs=False)
        voc, ctx = self.vocoder, self.vocoder.context
        sizes = () if ctx is None else (first + ctx, chunk + 2 * ctx)
        rng, previous = random.Random(seed), None
        history = self._history(acoustic_context, speaker)
        for sentence, prev in self._units(text, paragraph, context):
            ids, c = self.with_context(sentence, prev, self.language)
            ids = torch.tensor([ids], device=self.device)
            durations, _ = self._sample_prosody(ids, spk, g, length_scale * tempo, pt, dt, rs, edge)  # as _mel
            ref = getattr(history, "reference", None)
            cond, _ = m.aligned_condition(ids, torch.tensor([ids.shape[1]], device=self.device), spk,
                                          length_scale * tempo, durations=durations, context=c,
                                          spk_emb=None if ref is None else ref.vector)
            t = cond.shape[-1]
            pc, pm = (None, None) if history is None else history.prefix(t)
            if previous is not None:
                yield silence(pause, previous, rng, self.sample_rate)
            previous = sentence
            tokens = None if ref is None else ref.tokens
            if not gen.fits(t, 0 if pc is None else pc.shape[-1]):  # longer than the caches: whole sentence
                mel = m.generate(cond, spk, temperature, g, prefix=pm, cfg_scale=cfg_scale, prefix_cond=pc,
                                 tokens=tokens, guide=getattr(self, "guidance", "text"))
                if history is not None:
                    history.update(cond, mel)
                for piece in stream_vocoder(voc, self.stats.denormalize(mel), first=first, chunk=chunk, context=ctx,
                                            graphs=self.vocoder_graphs):
                    yield piece.cpu()
                continue
            steps = gen.steps(cond, spk, temperature, g, cfg_scale, prefix=pm, prefix_cond=pc, tokens=tokens,
                              guide=getattr(self, "guidance", "text"))
            mel = torch.zeros(1, m.n_mels, m.pad_frames(cond).shape[-1], device=self.device)
            done = 0
            for a, b, start, n in stream_windows(t, first, chunk, ctx):
                while done < b:
                    mel[..., done: done + m.patch] = next(steps).view(1, m.patch, m.n_mels).transpose(1, 2)
                    done += m.patch
                x = self.stats.denormalize(mel[..., a:b])
                yield vocode_window(voc, x, self.vocoder_graphs, sizes)[start * HOP_LENGTH: (start + n) * HOP_LENGTH
                                                                        ].cpu()
            for patch in steps:  # the remaining draws, so that the next sentence's match __call__'s
                mel[..., done: done + m.patch] = patch.view(1, m.patch, m.n_mels).transpose(1, 2)
                done += m.patch
            if history is not None:
                history.update(cond, mel[..., :t])

    def _stream_chunked(self, text: str, speaker: str | int, cfg_scale: float, temperature: float,
                        length_scale: float, seed: int, pause: Pause | None, first: int,
                        prosody_temperature: float | None, chunking: Chunking,
                        paragraph: bool | int | None = None, context: bool | None = None) -> Iterator[torch.Tensor]:
        """:meth:`stream` with the DiT on windows (:mod:`drifting_tts.chunked`): each vocoder window runs as soon as
        the DiT has committed its frames."""
        from .batched import stream_windows
        from .chunked import WindowedMel
        from .fast import vocode_window

        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        pause = self._pause(pause, speaker, context)
        g = torch.Generator(device=self.device).manual_seed(seed)
        rng, previous = random.Random(seed), None
        voc, ctx = self.vocoder, self.vocoder.context
        sizes = () if ctx is None else (first + ctx, chunking.chunk + 2 * ctx)
        generate = self._window_generator()
        for k, (sentence, prev) in enumerate(self._units(text, paragraph, context)):
            if chunking.noise == "philox":
                if prev is not None:
                    raise NotImplementedError("counter-based noise does not support context")
                z, cond, alpha, labels = self._prepare_philox(sentence, spk, seed, k, cfg_scale, temperature,
                                                              length_scale * tempo, prosody_temperature)
            else:
                z, cond, alpha, labels = self._prepare(sentence, spk, g, cfg_scale, temperature,
                                                       length_scale * tempo, prosody_temperature, rhythm, prev)
            t = cond.shape[-1]
            wm = WindowedMel(generate, z, cond, spk, alpha, labels, [t], chunking, first + (ctx or 0))
            if previous is not None:
                yield silence(pause, previous, rng, self.sample_rate)
            previous = sentence
            for a, b, start, n in stream_windows(t, first, chunking.chunk, ctx):
                while wm.committed[0] < b:
                    wm.step([0])
                x = self.stats.denormalize(wm.mel[..., a:b])
                yield vocode_window(voc, x, self.vocoder_graphs, sizes)[start * HOP_LENGTH: (start + n) * HOP_LENGTH
                                                                        ].cpu()

    def _pause(self, pause: Pause | None, speaker: str | int, context: bool | None = None):
        """The pause of a call: ``"punct"`` uses the prosody predictor's measured edge silence of this voice when its
        durations are sampled (``ProsodyPredictor.pause_edges``), else the table's (v3.1's regressors), less the
        leading silence that context removes (``context``, default: the Synthesizer's)."""
        spk_id = self.speaker_id(speaker)
        edge = None
        if self.prosody is not None and self.prosody_durations == "sampled":
            edge = self.prosody.pause_edges.get(spk_id)
        ctx = self.context if context is None else context
        return resolve_pause(self.pause if pause is None else pause, spk_id, edge, bool(ctx))

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
    durations = getattr(args, "prosody_durations", None) or rel.get("prosody_durations")  # None: the predictor's
    paragraph = getattr(args, "paragraph", None)  # None: the release's, else the model's default; 0: off
    return {"model_path": model, "vocoder": args.vocoder or rel.get("vocoder"),
            "prosody": None if str(prosody).lower() == "none" else prosody, "prosody_durations": durations,
            "pause": pause, "paragraph": rel.get("paragraph") if paragraph is None else paragraph or False,
            "context": bool(getattr(args, "context", False)),
            "temperature": rel.get("temperature"), "cfg_scale": rel.get("cfg_scale"),
            "acoustic_context": getattr(args, "acoustic_context", None)}


def run(args) -> None:
    synth = Synthesizer(device=args.device, cuda_kernel=args.cuda_kernel, prosody_temperature=args.prosody_temperature,
                        prosody_spread=args.prosody_spread,
                        prosody_duration_temperature=args.prosody_duration_temperature,
                        prosody_pitch=args.prosody_pitch_model, sample_rate=args.sample_rate or None,
                        **pipeline_args(args))
    if args.list_speakers:
        for name, v in VOICES.items():
            default = " (default)" if name == DEFAULT_VOICE else ""
            print(f"{name:7s} speaker {v['id']}, median pitch {v['pitch_hz']} Hz{default}")
        return
    kw = dict(speaker=args.speaker, cfg_scale=args.cfg, temperature=args.temperature,  # None: the Synthesizer's
              length_scale=args.length_scale, seed=args.seed, attn_window=args.attn_window,
              pitch_shift=args.pitch_shift, steps=args.steps)
    if getattr(args, "ref_audio", None):
        kw["ref_audio"] = synth.reference(args.ref_audio)  # prepared once for every line
    if args.text_file:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [line.strip() for line in Path(args.text_file).read_text().splitlines() if line.strip()]
        for i, line in enumerate(lines):
            wav, info = synth(line, **kw)
            sf.write(out_dir / f"{i:03d}.wav", wav.numpy(), synth.sample_rate)
            print(f"{out_dir / f'{i:03d}.wav'}  {info['seconds']:.2f}s  RTF {info['rtf_total']:.4f}")
    else:
        if not args.text:
            raise SystemExit("--text or --text-file is required")
        wav, info = synth(args.text, **kw)
        sf.write(args.out, wav.numpy(), synth.sample_rate)
        print(f"{args.out}  {info['seconds']:.2f}s  "
              f"RTF acoustic {info['rtf_acoustic']:.4f} total {info['rtf_total']:.4f}")
