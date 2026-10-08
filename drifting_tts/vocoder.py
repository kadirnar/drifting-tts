"""Vocoders for the 24 kHz, 100-bin log-mel backends, and a registry of drop-in choices.

:func:`load_vocoder` takes a registry name (:data:`VOCODERS`), a checkpoint path or ``None`` (the stock vocoder of
the model's backend) and returns a :class:`Vocoder`: unnormalised log-mel ``[B, 100, T]`` -> waveform
``[B, T * 256]`` at 24 kHz in [-1, 1] (``[B, (T - 1) * 256]`` for Vocos on its own centred mels).
"""

from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .audio import HOP_LENGTH, N_FFT

VOCOS_REPO = "charactr/vocos-mel-24khz"
BIGVGAN_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"
HUB_REPO = "Vyvo/drifting-tts-tr"
REVOX_REPO, REVOX_REVISION = "minori-live/revox-vocoder-1", "025862deee2003230d9d0fd8f3ad56a5bc09d31c"
REVOX_CREDIT = (f"Minori Live — Revox Vocoder 1.0 (https://huggingface.co/{REVOX_REPO}), CC BY-NC-SA 4.0: "
                "non-commercial use only")


@dataclass(frozen=True)
class VocoderEntry:
    """``kind``: ``bigvgan``, ``vocos``, ``griffin-lim`` or ``revox``; ``repo``: the NVIDIA repo (code, config,
    weights), the Vocos init or Revox's repo; ``hub_file``: a fine-tuned checkpoint in :data:`HUB_REPO`, read from
    ``local`` instead when that file exists (a ``finetune-vocoder`` output not yet on the Hub); ``context``: frames of
    context on each side of a streaming window, the smallest multiple of 8 at which streamed audio matches
    whole-utterance vocoding to the fp32 noise floor (> 90 dB SNR, docs/VOCODERS.md; ``None``: one piece per
    sentence); ``mel``: the mel front end it expects (``None``: the model's)."""

    kind: str
    about: str
    repo: str | None = None
    hub_file: str | None = None
    local: str | None = None
    context: int | None = 32
    mel: str | None = "bigvgan"


BASE_REPO = "nvidia/bigvgan_base_24khz_100band"
VOCODERS: dict[str, VocoderEntry] = {
    "bigvgan-v2-ft": VocoderEntry("bigvgan", "BigVGAN-v2 fine-tuned on this model's mels (release default)",
                                  BIGVGAN_REPO, "bigvgan_v2_ft.pt", context=32),
    "bigvgan-v2": VocoderEntry("bigvgan", "NVIDIA BigVGAN-v2, 24 kHz 100-band 256x", BIGVGAN_REPO, context=32),
    "bigvgan-v1": VocoderEntry("bigvgan", "NVIDIA BigVGAN (v1), 24 kHz 100-band", "nvidia/bigvgan_24khz_100band",
                               context=24),
    "bigvgan-base": VocoderEntry("bigvgan", "NVIDIA BigVGAN-base (v1), 24 kHz 100-band", BASE_REPO, context=16),
    "bigvgan-base-ft": VocoderEntry("bigvgan", "BigVGAN-base fine-tuned on this model's mels", BASE_REPO,
                                    "bigvgan_base_ft.pt", "runs/bigvgan_base_ft/bigvgan_ft.pt", context=16),
    "vocos-ft": VocoderEntry("vocos", "Vocos fine-tuned on this model's (BigVGAN-style) mels", VOCOS_REPO,
                             "vocos_ft.pt", "runs/vocos_bigvgan/vocos_ft.pt", context=32),
    "vocos": VocoderEntry("vocos", "charactr/vocos-mel-24khz, for models trained on Vocos mels", VOCOS_REPO,
                          context=32, mel="vocos"),
    "griffin-lim": VocoderEntry("griffin-lim", "mel pseudo-inverse + NNLS, then fast Griffin-Lim (no weights)",
                                context=None, mel=None),
    "revox": VocoderEntry("revox", f"{REVOX_CREDIT}; 48 kHz, F0 from the Griffin-Lim audio of the mel "
                          "('revox:<F0 source>[:dio|harvest]': griffin-lim, none or a registry vocoder)", REVOX_REPO,
                          context=None),
}


def build_vocos(init: str):
    """``init``: a HF repo id (pretrained weights) or a local Vocos YAML config (random initialisation)."""
    from vocos import Vocos

    if init.endswith((".yaml", ".yml")) and Path(init).exists():
        return Vocos.from_hparams(init)
    return Vocos.from_pretrained(init)


def load_vocos(init: str = VOCOS_REPO, state: dict | None = None, device: str = "cuda"):
    """Inference Vocos: ``init`` (see :func:`build_vocos`), then a fine-tuned ``state`` dict if given."""
    model = build_vocos(init)
    if state is not None:
        model.load_state_dict(state)
    return model.to(device).eval()


def input_conv(model: nn.Module) -> nn.Conv1d:
    """The first convolution, which reads the mel: ``backbone.embed`` (Vocos) or ``conv_pre`` (BigVGAN)."""
    return model.backbone.embed if hasattr(model, "backbone") else model.conv_pre


def extend_input_conv(conv: nn.Conv1d, extra: int) -> None:
    """Append ``extra`` zero-initialised input channels to ``conv`` in place (plain or ``weight_g`` / ``weight_v``
    weight norm), so the pretrained mapping is unchanged until the new weights learn (noise-conditioned vocoders)."""
    name = "weight_v" if hasattr(conv, "weight_v") else "weight"
    w = getattr(conv, name)
    setattr(conv, name, nn.Parameter(torch.cat([w.data, w.data.new_zeros(w.shape[0], extra, *w.shape[2:])], 1)))
    conv.in_channels += extra


def fold_noise_channels(conv: nn.Conv1d, extra: int) -> None:
    """Drop the last ``extra`` input channels: exact for input noise ``z = 0``. Weight norm must be removed first."""
    if hasattr(conv, "weight_v"):
        raise ValueError("remove the weight norm before folding the noise channels")
    conv.weight = nn.Parameter(conv.weight.data[:, : conv.in_channels - extra].clone())
    conv.in_channels -= extra


def bigvgan_snapshot(repo: str = BIGVGAN_REPO, weights: bool = True) -> tuple[types.ModuleType, str]:
    """NVIDIA BigVGAN (MIT): its code ships with each HF repo (``pip install drifting-tts[bigvgan]``; the v1 and v2
    repos ship the same code). Returns the imported ``bigvgan`` module and the snapshot path (with the generator
    weights if ``weights``)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, allow_patterns=["*.py", "config.json", "LICENSE", "alias_free_activation/**"]
                             + (["bigvgan_generator.pt"] if weights else []))
    sys.path.insert(0, path)  # bigvgan.py imports its helpers as top-level modules
    try:
        import bigvgan
    finally:
        sys.path.remove(path)
    return bigvgan, path


def _install_cuda_kernel(path: str) -> None:
    """Build BigVGAN's fused anti-aliased activation (inference only) for the local GPU and use it in place of the
    repo's loader, which hard-codes ``sm_70`` / ``sm_80`` SASS (no code for newer GPUs such as sm_120)."""
    import alias_free_activation.cuda as pkg  # importable once bigvgan is
    from torch.utils import cpp_extension

    major, minor = torch.cuda.get_device_capability()
    cc = f"{major}{minor}"
    if getattr(pkg, "load", None) is not None and getattr(pkg.load, "arch", None) == cc:
        return
    src = Path(path) / "alias_free_activation" / "cuda"
    build = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache" / "torch_extensions")) / f"bigvgan_{cc}"
    build.mkdir(parents=True, exist_ok=True)
    ext = cpp_extension.load(
        name=f"bigvgan_anti_alias_activation_sm{cc}", build_directory=str(build),
        sources=[str(src / "anti_alias_activation.cpp"), str(src / "anti_alias_activation_cuda.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode", f"arch=compute_{cc},code=sm_{cc}", "--use_fast_math",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "--expt-relaxed-constexpr", "--expt-extended-lambda"])
    loader = types.ModuleType("alias_free_activation.cuda.load")
    loader.load, loader.arch = (lambda: ext), cc
    pkg.load = sys.modules[loader.__name__] = loader


def build_bigvgan(repo: str = BIGVGAN_REPO, hparams: dict | None = None, pretrained: bool = True,
                  cuda_kernel: bool = False):
    """BigVGAN generator *with* weight norm, as trained. ``hparams`` override ``config.json`` (e.g. a tiny model
    for tests); ``pretrained`` loads the released generator weights; ``cuda_kernel``: fused activation (inference)."""
    bigvgan, path = bigvgan_snapshot(repo, weights=pretrained)
    h = bigvgan.load_hparams_from_json(f"{path}/config.json")
    h.update(hparams or {})
    if cuda_kernel:
        _install_cuda_kernel(path)
    # BigVGAN.from_pretrained predates current huggingface_hub (missing kwargs), so build it directly
    model = bigvgan.BigVGAN(h, use_cuda_kernel=cuda_kernel)
    if pretrained:
        model.load_state_dict(torch.load(f"{path}/bigvgan_generator.pt", map_location="cpu",
                                         weights_only=False)["generator"])
    return model


def load_bigvgan(repo: str = BIGVGAN_REPO, device: str = "cuda", finetuned: str | dict | None = None,
                 cuda_kernel: bool = False, fold_noise: bool = True):
    """Inference generator (weight norm removed): the released weights, or a ``bigvgan_ft.pt`` written by
    ``drifting-tts finetune-vocoder`` (same ``{"generator": ...}`` layout as ``bigvgan_generator.pt``; a path or the
    loaded dict). A generator trained with ``noise_channels`` input noise (``vocoder.objective: drift``) runs at
    ``z = 0``: those channels are folded away (``fold_noise``), which leaves a standard BigVGAN; otherwise they stay
    and the caller appends ``z``."""
    if finetuned:
        ck = finetuned if isinstance(finetuned, dict) else torch.load(finetuned, map_location="cpu",
                                                                       weights_only=False)
        model = build_bigvgan(ck.get("repo", repo), ck.get("hparams"), pretrained=False, cuda_kernel=cuda_kernel)
        noise = int(ck.get("noise_channels", 0))
        if noise:
            extend_input_conv(model.conv_pre, noise)
        model.load_state_dict(ck["generator"])
    else:
        model, noise = build_bigvgan(repo, cuda_kernel=cuda_kernel), 0
    model.remove_weight_norm()
    if noise and fold_noise:
        fold_noise_channels(model.conv_pre, noise)
    return model.to(device).eval()


def ola_istft(spec: Tensor, window: Tensor, centred: bool = False) -> Tensor:
    """Least-squares inverse STFT ``[B, n_fft // 2 + 1, T]`` -> ``[B, samples]`` (overlap-add, divided by the
    squared-window envelope) with no host synchronisation, so that it can run inside a CUDA graph. ``centred``:
    ``(T - 1) * hop`` samples, as ``torch.istft(center=True)``; else BigVGAN's framing (Vocos's ``same`` head),
    frame ``i`` centred on sample ``i * hop + hop / 2``, ``T * hop`` samples."""
    t = spec.shape[-1]
    size, pad = (t - 1) * HOP_LENGTH + N_FFT, N_FFT // 2 if centred else (N_FFT - HOP_LENGTH) // 2
    frames = torch.fft.irfft(spec, N_FFT, dim=1) * window[:, None]
    y = F.fold(frames, (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))[:, 0, 0]
    env = F.fold(window.square()[None, :, None].expand(1, -1, t), (1, size), (1, N_FFT), stride=(1, HOP_LENGTH))
    return (y / env[0, 0, 0].clamp_min(1e-11))[:, pad: size - pad]


class GriffinLim(torch.nn.Module):
    """Weight-free baseline: the mel filterbank is inverted by non-negative least squares (projected gradient with
    Nesterov momentum, started from the clipped pseudo-inverse), then the phase comes from fast Griffin-Lim
    (Perraudin et al. 2013) on the same STFT as the mel front end. ``mel``: ``bigvgan`` (reflect-padded uncentred
    frames, ``T`` frames -> ``T * hop`` samples) or ``vocos`` (centred, ``(T - 1) * hop``)."""

    def __init__(self, mel: str = "bigvgan", n_iter: int = 64, momentum: float = 0.99, nnls_iter: int = 100,
                 seed: int = 0):
        from .audio import make_logmel

        super().__init__()
        front = make_logmel(mel)
        fb = front.fb if mel == "bigvgan" else front.mel.mel_scale.fb.T  # [n_mels, n_fft // 2 + 1]
        self.register_buffer("fb", fb.contiguous(), persistent=False)
        self.register_buffer("pinv", torch.linalg.pinv(fb), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.centred, self.n_iter, self.momentum, self.nnls_iter, self.seed = mel == "vocos", n_iter, momentum, \
            nnls_iter, seed
        self.step = 1.0 / torch.linalg.matrix_norm(fb, ord=2).item() ** 2

    def magnitude(self, log_mel: Tensor) -> Tensor:
        """``[B, n_mels, T]`` log-mel -> ``[B, n_fft // 2 + 1, T]`` linear magnitude: ``min ||fb s - mel||, s >= 0``."""
        m = log_mel.exp()
        s = (self.pinv @ m).clamp_min(0)
        y, t = s, 1.0
        for _ in range(self.nnls_iter):
            s_next = (y - self.step * (self.fb.T @ (self.fb @ y - m))).clamp_min(0)
            t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
            y, s, t = s_next + (t - 1) / t_next * (s_next - s), s_next, t_next
        return s

    def stft(self, y: Tensor) -> Tensor:
        if self.centred:
            return torch.stft(y, N_FFT, HOP_LENGTH, N_FFT, self.window, center=True, pad_mode="reflect",
                              return_complex=True)
        pad = (N_FFT - HOP_LENGTH) // 2
        y = F.pad(y[:, None], (pad, pad), mode="reflect" if y.shape[-1] > pad else "constant")[:, 0]
        return torch.stft(y, N_FFT, HOP_LENGTH, N_FFT, self.window, center=False, return_complex=True)

    def istft(self, spec: Tensor) -> Tensor:
        return ola_istft(spec, self.window, self.centred)

    @torch.no_grad()
    def forward(self, log_mel: Tensor) -> Tensor:
        return self.reconstruct(self.magnitude(log_mel.float()))

    def reconstruct(self, mag: Tensor) -> Tensor:
        """Linear magnitude ``[B, n_fft // 2 + 1, T]`` -> waveform, phase by fast Griffin-Lim."""
        g = torch.Generator().manual_seed(self.seed)
        angles = torch.polar(torch.ones(mag.shape), 2 * math.pi * torch.rand(mag.shape, generator=g)).to(mag.device)
        prev = torch.zeros_like(angles)
        for _ in range(self.n_iter):
            rebuilt = self.stft(self.istft(mag * angles))
            angles = rebuilt - self.momentum / (1 + self.momentum) * prev
            angles, prev = angles / (angles.abs() + 1e-16), rebuilt
        return self.istft(mag * angles)


REVOX_RATE, REVOX_FFT, REVOX_HOP, REVOX_MELS = 48_000, 2048, 480, 128


def revox_frames(t: int) -> int:
    """Revox frames (10 ms, ``ceil(samples / 480)`` at 48 kHz) of the audio of ``t`` BigVGAN frames."""
    return -(-t * 2 * HOP_LENGTH // REVOX_HOP)


def to_revox_frames(x: Tensor) -> Tensor:
    """``[..., T]`` on BigVGAN frames (centred at ``(i + 1/2) * 256 / 24000`` s) -> ``[..., revox_frames(T)]`` on
    Revox frames (centred at ``k / 100`` s), by linear interpolation; frames outside the first / last centre are
    held."""
    t = x.shape[-1]
    pos = (torch.arange(revox_frames(t), device=x.device) * (REVOX_HOP / 2 / HOP_LENGTH) - 0.5).clamp(0, t - 1)
    lo = pos.floor().long()
    w = pos - lo
    return x[..., lo] * (1 - w) + x[..., (lo + 1).clamp(max=t - 1)] * w


def resample_sharp(wav: Tensor, sr: int, target: int) -> Tensor:
    """Kaiser-windowed sinc resampling, flat to 11.5 kHz between 24 and 48 kHz (torchaudio's default rolls off
    above 8 kHz)."""
    import torchaudio

    return torchaudio.functional.resample(wav, sr, target, lowpass_filter_width=64, rolloff=0.995,
                                          resampling_method="sinc_interp_kaiser", beta=14.77)


class RevoxLogMel(torch.nn.Module):
    """Revox's mel front end on 48 kHz audio: centred, zero-padded STFT (periodic Hann 2048, hop 480), magnitude,
    128 Slaney filters (area norm, 0-24 kHz), ``ln(max(., 1e-5))``, ``ceil(samples / 480)`` frames at ``k * 10`` ms."""

    def __init__(self):
        import torchaudio

        super().__init__()
        fb = torchaudio.functional.melscale_fbanks(REVOX_FFT // 2 + 1, 0.0, REVOX_RATE / 2, REVOX_MELS, REVOX_RATE,
                                                   norm="slaney", mel_scale="slaney")
        self.register_buffer("fb", fb.T.contiguous(), persistent=False)
        self.register_buffer("window", torch.hann_window(REVOX_FFT), persistent=False)

    @torch.no_grad()
    def forward(self, wav: Tensor) -> Tensor:
        """``[B, samples]`` at 48 kHz -> ``[B, 128, ceil(samples / 480)]``."""
        spec = torch.stft(wav, REVOX_FFT, REVOX_HOP, REVOX_FFT, self.window, center=True, pad_mode="constant",
                          return_complex=True)
        return (self.fb @ spec.abs())[..., : -(-wav.shape[-1] // REVOX_HOP)].clamp_min(1e-5).log()

    def convert(self, mag: Tensor) -> Tensor:
        """24 kHz linear magnitude on BigVGAN frames ``[B, 513, T]`` -> Revox log-mel ``[B, 128, revox_frames(T)]``.

        Doubled, it is the 48 kHz magnitude of the audio upsampled x2 (same 23.4375 Hz bins, a window as long), empty
        above 12 kHz. Time is interpolated in the linear domain, where it commutes with the filters (the geometric
        mean of log-domain interpolation is biased low at onsets)."""
        return to_revox_frames(self.fb[:, : mag.shape[1]] @ (2 * mag)).clamp_min(1e-5).log()


def revox_pitch(wav: Tensor, sample_rate: int, frames: int, method: str = "dio") -> tuple[Tensor, Tensor, Tensor]:
    """``f0_hz``, ``voiced``, ``pitch_valid`` (``[frames]`` each) from WORLD on ``wav`` ``[samples]``, at Revox's
    frame centres ``k * 10`` ms. WORLD's unvoiced frames are reliable unvoiced decisions (``pitch_valid``). dio runs
    with a looser voicing threshold (``allowed_range`` 0.2): Revox cannot voice a frame without F0, and dio's default
    misses voiced frames of Griffin-Lim audio (docs/VOCODERS.md)."""
    from .audio import world_f0

    f0 = torch.zeros(frames)
    raw = torch.from_numpy(world_f0(wav.double().cpu().numpy(), sample_rate, 1000 * REVOX_HOP / REVOX_RATE, method,
                                    allowed_range=0.2))
    f0[: min(frames, len(raw))] = raw[:frames].float()
    return f0, f0 > 0, torch.ones(frames, dtype=torch.bool)


class Revox(torch.nn.Module):
    """Revox Vocoder 1.0 by Minori Live (``REVOX_CREDIT``; CC BY-NC-SA 4.0, **non-commercial**): a 4.46 M-parameter
    48 kHz PC-NSF-Vocos conditioned on a 128-band log-mel and frame-level F0, here on this model's BigVGAN mels. The
    ONNX graph is downloaded from the original repo at runtime (never redistributed) and runs in ONNX Runtime on the
    CPU (its CUDA provider was slower on an RTX 5090).

    Mel conversion: :class:`GriffinLim`'s NNLS inverts the 100-band mel to the 513-bin magnitude, which
    :meth:`RevoxLogMel.convert` turns into Revox's 128 bands on 100 Hz frames centred at ``k * 10`` ms.

    ``f0``: the source of the frame-level F0 (the model predicts pitch per token only): WORLD ``method`` on the
    Griffin-Lim audio of the same mel (``griffin-lim``, no weights), on the audio of another registry vocoder (e.g.
    ``bigvgan-v2-ft``), or ``none`` (every frame pitch-invalid, so Revox reads the pitch from the mel). The 48 kHz
    output is resampled to 24 kHz (the mel carries nothing above 12 kHz); ``seed`` fixes the source noise."""

    num_params = 4_463_874

    def __init__(self, f0: str = "griffin-lim", method: str = "dio", seed: int = 0, path: str | None = None,
                 device: str = "cpu"):
        """``path``: a local ``vocoder.onnx`` (default: downloaded, pinned revision); ``device``: the F0 vocoder's."""
        import onnxruntime as ort

        super().__init__()
        self.gl, self.front = GriffinLim("bigvgan"), RevoxLogMel()
        self.f0, self.method, self.seed = f0, method, seed
        self.f0_vocoder = None if f0 in ("griffin-lim", "none") else load_vocoder(f0, device, backend="bigvgan")
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(REVOX_REPO, "vocoder.onnx", revision=REVOX_REVISION)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = torch.get_num_threads(), 1
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    def pitch(self, log_mel: Tensor, mag: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``f0_hz``, ``voiced``, ``pitch_valid`` ``[B, revox_frames(T)]`` from the ``f0`` source."""
        b, k = log_mel.shape[0], revox_frames(log_mel.shape[-1])
        if self.f0 == "none":
            return torch.zeros(b, k), torch.zeros(b, k, dtype=torch.bool), torch.zeros(b, k, dtype=torch.bool)
        wav = self.gl.reconstruct(mag) if self.f0_vocoder is None else self.f0_vocoder(log_mel)
        return tuple(torch.stack(x) for x in zip(*(revox_pitch(w, 24_000, k, self.method) for w in wav)))

    @torch.no_grad()
    def generate(self, mel: Tensor, f0: Tensor, voiced: Tensor, valid: Tensor) -> Tensor:
        """Revox inputs ``[B, 128, K]`` and ``[B, K]`` -> 48 kHz waveform ``[B, 480 K]`` (centred ISTFT)."""
        k, wavs = mel.shape[-1], []
        noise = torch.randn(mel.shape[0], 1, REVOX_HOP * k, generator=torch.Generator().manual_seed(self.seed))
        for i in range(mel.shape[0]):
            real, imag = self.session.run(None, {
                "mel": mel[i: i + 1].float().cpu().numpy(), "f0_hz": f0[i: i + 1].float().cpu().numpy(),
                "voiced": voiced[i: i + 1].cpu().numpy(), "pitch_valid": valid[i: i + 1].cpu().numpy(),
                "noise": noise[i].numpy()})
            spec = torch.complex(torch.from_numpy(real), torch.from_numpy(imag)).to(mel.device)
            wavs.append(torch.istft(spec, REVOX_FFT, REVOX_HOP, REVOX_FFT, self.front.window, center=True,
                                    length=REVOX_HOP * k))
        return torch.cat(wavs)

    @torch.no_grad()
    def forward(self, log_mel: Tensor) -> Tensor:
        """BigVGAN log-mel ``[B, 100, T]`` -> 24 kHz waveform ``[B, T * 256]``."""
        log_mel = log_mel.float()
        mag = self.gl.magnitude(log_mel)
        wav = self.generate(self.front.convert(mag), *self.pitch(log_mel, mag))
        return resample_sharp(wav, REVOX_RATE, 24_000)[..., : log_mel.shape[-1] * HOP_LENGTH]


class Vocoder:
    """Unnormalised log-mel ``[B, 100, T]`` -> waveform ``[B, samples]`` at 24 kHz, clamped to [-1, 1].

    Attributes: ``kind`` (``bigvgan`` / ``vocos`` / ``griffin-lim`` / ``revox``), ``mel`` (the front end it expects),
    ``name``, ``context`` (frames of context for streaming windows, ``None``: vocode each sentence whole) and
    ``graphs`` (whether fixed-size windows may run as CUDA graphs). Build one with :func:`load_vocoder`."""

    def __init__(self, device: str = "cuda", repo: str = VOCOS_REPO, finetuned: str | None = None,
                 backend: str = "vocos", cuda_kernel: bool = False, noise_seed: int | None = None):
        """``backend``: ``vocos`` or ``bigvgan`` (must match the mels the model was trained on).
        ``finetuned``: a ``vocos_ft.pt`` / ``bigvgan_ft.pt`` written by ``drifting-tts finetune-vocoder``.
        ``cuda_kernel``: BigVGAN's fused CUDA activation (~3x faster inference; built with nvcc on first use).
        ``noise_seed``: see :func:`load_vocoder`."""
        if finetuned or backend == "bigvgan":
            v = load_vocoder(finetuned, device, cuda_kernel, backend, noise_seed=noise_seed)
        else:
            v = Vocoder.wrap(load_vocos(repo, device=device), "vocos", device, "vocos", repo, VOCODERS["vocos"].context)
        self.__dict__.update(v.__dict__)

    @classmethod
    def wrap(cls, model, kind: str, device: str, mel: str = "bigvgan", name: str = "", context: int | None = 32):
        self = cls.__new__(cls)
        self.model, self.kind, self.device, self.mel, self.name, self.context = model, kind, device, mel, name, context
        self.noise, self.noise_seed = 0, None  # input-noise channels left unfolded (vocoder.objective: drift)
        return self

    @property
    def graphs(self) -> bool:
        return self.kind not in ("griffin-lim", "revox")

    @property
    def num_params(self) -> int:
        return getattr(self.model, "num_params", None) or sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def __call__(self, log_mel: Tensor) -> Tensor:
        """Unnormalised log-mel ``[B, 100, T]`` -> waveform ``[B, samples]`` at 24 kHz."""
        x = log_mel.to(self.device).float()
        if self.noise:
            g = torch.Generator(device=self.device).manual_seed(self.noise_seed)
            x = torch.cat([x, torch.randn(x.shape[0], self.noise, x.shape[-1], generator=g, device=self.device)], 1)
        if self.kind == "bigvgan":
            wav = self.model(x)[:, 0]
        elif self.kind == "vocos" and self.mel == "bigvgan":
            wav = self._vocos_bigvgan(x)
        elif self.kind == "vocos":
            wav = self.model.decode(x)
        else:
            wav = self.model(x)
        return wav.clamp(-1, 1)

    def _vocos_bigvgan(self, x: Tensor) -> Tensor:
        """Vocos backbone + ISTFT head on BigVGAN-style mels, ``T * hop`` samples (a ``center`` head gives
        ``(T - 1) * hop``, so it gets one more, repeated frame)."""
        n, head = x.shape[-1] * HOP_LENGTH, self.model.head
        centred = head.istft.padding == "center"
        h = head.out(self.model.backbone(F.pad(x, (0, 1), mode="replicate") if centred else x)).transpose(1, 2)
        mag, p = h.chunk(2, dim=1)
        spec = torch.exp(mag).clip(max=1e2) * (torch.cos(p) + 1j * torch.sin(p))
        return ola_istft(spec, head.istft.window, centred)[..., :n]


def checkpoint_kind(ck: dict) -> str:
    """``bigvgan`` for ``{"generator", "repo"?, "hparams"?}``, ``vocos`` for ``{"vocos", "init"?, "mel"?}``."""
    for key, kind in (("generator", "bigvgan"), ("vocos", "vocos")):
        if key in ck:
            return kind
    raise ValueError(f"not a vocoder checkpoint (keys {sorted(ck)[:8]}): expected a BigVGAN 'generator' or a "
                     "'vocos' state dict")


def _from_checkpoint(ck: dict, device: str, cuda_kernel: bool, name: str, context: int | None = None,
                     noise_seed: int | None = None) -> Vocoder:
    noise = int(ck.get("noise_channels", 0))  # drift-trained: folded away at z = 0 unless noise_seed
    if checkpoint_kind(ck) == "bigvgan":
        repo = ck.get("repo", BIGVGAN_REPO)  # the context of this repo's fine-tune entry, else of the stock one
        same = sorted((e for e in VOCODERS.values() if e.repo == repo), key=lambda e: not e.hub_file)
        base = same[0] if same else VOCODERS["bigvgan-v2-ft"]
        model = load_bigvgan(repo, device, finetuned=ck, cuda_kernel=cuda_kernel, fold_noise=noise_seed is None)
        return _with_noise(Vocoder.wrap(model, "bigvgan", device, "bigvgan", name, context or base.context), noise,
                           noise_seed)
    # a Vocos fine-tune: "mel": "bigvgan" when trained on BigVGAN-style mels (with "head_padding": "same", BigVGAN's
    # framing), else Vocos's own mels
    model = build_vocos(ck.get("init", VOCOS_REPO))
    model.head.istft.padding = ck.get("head_padding", model.head.istft.padding)
    if noise:
        extend_input_conv(model.backbone.embed, noise)
    model.load_state_dict(ck["vocos"])
    if noise and noise_seed is None:
        fold_noise_channels(model.backbone.embed, noise)
    v = Vocoder.wrap(model.to(device).eval(), "vocos", device, ck.get("mel", "vocos"), name,
                     context or VOCODERS["vocos-ft"].context)
    return _with_noise(v, noise, noise_seed)


def _with_noise(v: Vocoder, noise: int, noise_seed: int | None) -> Vocoder:
    """Keep a drift-trained vocoder's input-noise channels and feed them fixed noise (``noise_seed``)."""
    if noise and noise_seed is not None:
        v.noise, v.noise_seed = noise, noise_seed
    return v


def _hub_checkpoint(name: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(HUB_REPO, filename)
    except EntryNotFoundError as e:
        raise FileNotFoundError(f"{name}: {HUB_REPO}/{filename} not found (not published yet, or offline); pass "
                                "the checkpoint path instead") from e


def load_vocoder(spec: str | None = None, device: str = "cuda", cuda_kernel: bool = False,
                 backend: str = "bigvgan", noise_seed: int | None = None) -> Vocoder:
    """A registry name (:data:`VOCODERS`), a checkpoint path, or ``None`` for the stock vocoder of ``backend`` (the
    mel front end of the TTS model: ``bigvgan`` or ``vocos``; it also picks Griffin-Lim's filterbank).

    Checkpoints are recognised by their keys: ``{"generator", "repo"?, "hparams"?}`` is a BigVGAN-family fine-tune,
    ``{"vocos", "init"?, "mel"?}`` a Vocos fine-tune (``mel: "bigvgan"``: trained on BigVGAN-style mels).
    ``revox:<F0 source>[:<WORLD method>]`` sets the F0 of :class:`Revox` (non-commercial license: :data:`REVOX_CREDIT`).
    ``cuda_kernel``: BigVGAN's fused activation kernel (ignored by the others). A vocoder trained with input noise
    (``vocoder.objective: drift``) runs at ``z = 0`` (the noise channels folded away, a standard network), or with
    fixed Gaussian noise from ``noise_seed``."""
    if spec is None:
        spec = "bigvgan-v2" if backend == "bigvgan" else "vocos"
    name, _, f0 = spec.partition(":")
    if name in VOCODERS and VOCODERS[name].kind == "revox":
        f0, _, method = f0.partition(":")
        model = Revox(f0 or "griffin-lim", method or "dio", device=device)
        return Vocoder.wrap(model.to(device), "revox", device, "bigvgan", spec, VOCODERS[name].context)
    if spec in VOCODERS:
        e = VOCODERS[spec]
        if e.kind == "griffin-lim":
            return Vocoder.wrap(GriffinLim(backend).to(device), e.kind, device, backend, spec, e.context)
        if e.hub_file:
            path = e.local if e.local and Path(e.local).is_file() else _hub_checkpoint(spec, e.hub_file)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            return _from_checkpoint(ck, device, cuda_kernel, spec, e.context, noise_seed)
        if e.kind == "bigvgan":
            return Vocoder.wrap(load_bigvgan(e.repo, device, cuda_kernel=cuda_kernel), e.kind, device, e.mel, spec,
                                e.context)
        return Vocoder.wrap(load_vocos(e.repo, device=device), e.kind, device, e.mel, spec, e.context)
    if not Path(spec).is_file():
        raise ValueError(f"unknown vocoder {spec!r}: expected a checkpoint path or one of {', '.join(VOCODERS)}")
    return _from_checkpoint(torch.load(spec, map_location="cpu", weights_only=False), device, cuda_kernel, spec,
                            noise_seed=noise_seed)
