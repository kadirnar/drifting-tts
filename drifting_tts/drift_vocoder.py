"""A GAN-free vocoder: the drifting objective in a frozen feature space replaces the discriminators (issue #18).

A GAN vocoder learns content from a reconstruction loss (multi-scale mel L1) and realism from an adversarial loss,
whose discriminator judges local patches. Here the realism term is the drifting field of Deng et al. (``drift.py``)
in a **frozen** multi-scale feature space, as the paper uses a frozen pretrained MAE:

* the generator ``wave = G(mel, z)`` is a Vocos or BigVGAN with ``noise_channels`` Gaussian input channels at the frame
  rate (:class:`NoisyVocoder`), so it can draw ``S`` different waveforms per mel segment;
* feature maps (every map is ``L`` locations x ``D`` channels): the released, frozen BigVGAN discriminators
  (:class:`DiscriminatorFeatures`), multi-resolution log-magnitude STFT patches (:class:`STFTFeatures`) and optionally
  hidden states of a frozen speech SSL encoder (:class:`SSLFeatures`);
* pairing (:func:`vocoder_drift_loss`): ``conditional`` (one drift problem per segment and location: the real
  waveform's features are the positive, the other samples the negatives) and / or ``pooled`` (one problem per map over
  all locations of all segments: the distribution of local patches, which is what a patch discriminator judges).

Feature scales need no hand normalisation: ``drift.py`` divides every map by its mean pairwise distance.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

from .audio import SAMPLE_RATE
from .drift import DEFAULT_TEMPERATURES, feature_drift_loss, key_weight
from .models import bigvgan_disc as bd
from .vocoder import BIGVGAN_REPO, extend_input_conv, input_conv

PAIRINGS = ("conditional", "pooled")
Part = Callable[[Tensor], dict[str, Tensor]]  # waveforms [N, samples] -> {map: [N, L, D]}


class NoisyVocoder(nn.Module):
    """``wave = G([mel; z])``: a Vocos (``backbone`` + ``head``) or BigVGAN generator whose input convolution reads
    ``noise_channels`` extra Gaussian channels at the frame rate. Their weights start at zero, so the pretrained vocoder
    is unchanged at initialisation; ``z = None`` means ``z = 0`` (inference)."""

    def __init__(self, net: nn.Module, noise_channels: int = 0):
        super().__init__()
        self.net, self.noise_channels = net, noise_channels
        if noise_channels:
            extend_input_conv(input_conv(net), noise_channels)

    def forward(self, mel: Tensor, z: Tensor | None = None) -> Tensor:
        """``mel`` ``[B, n_mels, F]``, ``z`` ``[B, noise_channels, F]`` -> waveform ``[B, samples]``."""
        if self.noise_channels:
            z = mel.new_zeros(mel.shape[0], self.noise_channels, mel.shape[-1]) if z is None else z.to(mel)
            mel = torch.cat([mel, z], 1)
        if hasattr(self.net, "backbone"):
            return self.net.head(self.net.backbone(mel))
        return self.net(mel)[:, 0]


def _locations(f: Tensor) -> Tensor:
    """``[N, C, *spatial] -> [N, L, C]``: every spatial position is a location with ``C`` channels."""
    return f.flatten(2).transpose(1, 2)


def _remove_weight_norm(module: nn.Module) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for m in module.modules():
            if hasattr(m, "weight_g"):
                nn.utils.remove_weight_norm(m)


class DiscriminatorFeatures(nn.Module):
    """Every layer's feature map of NVIDIA's released, frozen BigVGAN discriminators: the multi-period discriminator
    (``mpd{p}_{i}``: time x period locations) and the sub-band CQT discriminator of v2 (``cqtd{j}_{i}``, its output
    map included) or the multi-resolution discriminator of v1 / base (``mrd{j}_{i}``): time x frequency locations.

    ``repo``: the HF repo whose ``config.json`` and ``bigvgan_discriminator_optimizer.pt`` are used (``None``: only
    ``hparams``, for tests). ``pretrained: false`` keeps a random initialisation (tests)."""

    def __init__(self, repo: str | None = BIGVGAN_REPO, mpd: bool = True, spectral: bool = True,
                 hparams: dict | None = None, pretrained: bool = True):
        super().__init__()
        h = {}
        if repo is not None:
            from huggingface_hub import hf_hub_download

            h = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        h = {"sampling_rate": SAMPLE_RATE, **h, **(hparams or {})}
        self.cqt = bool(h.get("use_cqtd_instead_of_mrd", False) or h.get("discriminator") != "mrd")
        self.mpd = bd.MultiPeriodDiscriminator(h) if mpd else None
        spec = bd.MultiScaleSubbandCQTDiscriminator if self.cqt else bd.MultiResolutionDiscriminator
        self.spec = spec(h) if spectral else None
        if pretrained:
            from huggingface_hub import hf_hub_download

            ck = torch.load(hf_hub_download(repo, "bigvgan_discriminator_optimizer.pt"), map_location="cpu",
                            weights_only=False)
            if self.mpd is not None:
                self.mpd.load_state_dict(ck["mpd"])
            if self.spec is not None:
                self.spec.load_state_dict(ck["mrd"])  # the released key for both CQT-D and MRD
        _remove_weight_norm(self)

    def parts(self) -> list[Part]:
        """One part per sub-discriminator."""
        out: list[Part] = []
        for d in self.mpd.discriminators if self.mpd is not None else []:
            out.append(lambda wave, d=d: {f"mpd{d.period}_{i}": _locations(f)
                                          for i, f in enumerate(d(wave[:, None])[1])})
        name = "cqtd" if self.cqt else "mrd"
        for j, d in enumerate(self.spec.discriminators if self.spec is not None else []):
            def part(wave: Tensor, d=d, j=j) -> dict[str, Tensor]:
                score, fmap = d(wave[:, None])
                maps = fmap + [score] if self.cqt else fmap  # the CQT-D leaves its output map out of fmap
                return {f"{name}{j}_{i}": _locations(f) for i, f in enumerate(maps)}
            out.append(part)
        return out

    def forward(self, wave: Tensor) -> dict[str, Tensor]:
        return {k: v for part in self.parts() for k, v in part(wave).items()}


class STFTFeatures(nn.Module):
    """Multi-resolution log-magnitude STFT patches (``stft{n_fft}``): ``patch = (bins, frames)`` tiles of
    ``log(|X| + eps)`` (Hann window, hop ``n_fft / 4``) -> locations x ``bins * frames`` channels. No phase."""

    def __init__(self, n_ffts: Sequence[int] = (512, 1024, 2048), patch: Sequence[int] = (16, 4), eps: float = 1e-5):
        super().__init__()
        self.n_ffts, self.patch, self.eps = list(n_ffts), tuple(patch), eps
        for n in self.n_ffts:
            self.register_buffer(f"win{n}", torch.hann_window(n), persistent=False)

    def forward(self, wave: Tensor) -> dict[str, Tensor]:
        out, (pf, pt) = {}, self.patch
        for n in self.n_ffts:
            spec = torch.view_as_real(torch.stft(wave.float(), n, n // 4, window=getattr(self, f"win{n}"),
                                                 center=True, return_complex=True))
            spec = 0.5 * (spec.pow(2).sum(-1) + self.eps**2).log()  # log |X|, smooth at |X| = 0
            N, fr, t = spec.shape
            fr, t = fr // pf, t // pt
            x = spec[:, : fr * pf, : t * pt].reshape(N, fr, pf, t, pt).permute(0, 1, 3, 2, 4)
            out[f"stft{n}"] = x.reshape(N, fr * t, pf * pt)
        return out


class SSLFeatures(nn.Module):
    """Hidden states (``ssl{layer}``: 50 Hz frames x channels) of a frozen self-supervised speech encoder from
    Hugging Face ``transformers`` (WavLM, wav2vec 2.0, HuBERT), fed at 16 kHz. ``normalize``: zero-mean, unit-variance
    input, for encoders trained that way (``do_normalize`` of their feature extractor)."""

    def __init__(self, name: str = "microsoft/wavlm-base-plus", layers: Sequence[int] = (3, 6, 9, 12),
                 normalize: bool = False, model: nn.Module | None = None):
        super().__init__()
        if model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(name)
        self.model, self.layers, self.normalize = model.eval(), list(layers), normalize
        self.resample = torchaudio.transforms.Resample(SAMPLE_RATE, 16_000)

    def forward(self, wave: Tensor) -> dict[str, Tensor]:
        x = self.resample(wave.float())
        if self.normalize:
            x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + 1e-7).sqrt()
        hs = self.model(x, output_hidden_states=True).hidden_states
        return {f"ssl{i}": hs[i] for i in self.layers}


class FeatureSpace(nn.Module):
    """Frozen multi-scale feature space: the union of the extractors' maps, ``{name: [N, L, D]}`` in fp32.

    :meth:`parts` splits it into independent parts (a sub-discriminator, the STFT set, the SSL encoder), so a training
    step can extract, drift and back-propagate one part at a time and hold only that part's graph in memory.
    ``global_stats``: also the mean and std of every map over its locations (``{name}_mean`` / ``_std``, ``L = 1``).
    ``dtype: bf16`` runs the extractors under bf16 autocast (CUDA)."""

    def __init__(self, extractors: dict[str, nn.Module], global_stats: bool = False, dtype: str = "fp32"):
        super().__init__()
        self.extractors = nn.ModuleDict(extractors)
        self.global_stats, self.bf16 = global_stats, dtype == "bf16"
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> FeatureSpace:  # always frozen: no dropout in the SSL encoder
        return super().train(False)

    def _wrap(self, part: Part) -> Part:
        def run(wave: Tensor) -> dict[str, Tensor]:
            with torch.autocast(wave.device.type, torch.bfloat16) if self.bf16 else nullcontext():
                out = {k: v.float() for k, v in part(wave).items()}
            if self.global_stats:
                for k, v in list(out.items()):
                    out[f"{k}_mean"], out[f"{k}_std"] = v.mean(1, keepdim=True), (v.var(1, keepdim=True) + 1e-6).sqrt()
            return out
        return run

    def parts(self) -> list[Part]:
        return [self._wrap(p) for ex in self.extractors.values()
                for p in (ex.parts() if hasattr(ex, "parts") else [ex])]

    def forward(self, wave: Tensor) -> dict[str, Tensor]:
        return {k: v for part in self.parts() for k, v in part(wave).items()}


def build_feature_space(fc: dict) -> FeatureSpace:
    """From the ``features`` config: ``discriminators`` / ``stft`` / ``ssl`` sections (null or absent: off)."""
    ex: dict[str, nn.Module] = {}
    if fc.get("discriminators"):
        ex["disc"] = DiscriminatorFeatures(**fc["discriminators"])
    if fc.get("stft"):
        ex["stft"] = STFTFeatures(**fc["stft"])
    if fc.get("ssl"):
        ex["ssl"] = SSLFeatures(**fc["ssl"])
    if not ex:
        raise ValueError("features: enable at least one of discriminators, stft, ssl")
    return FeatureSpace(ex, global_stats=fc.get("global_stats", False), dtype=fc.get("dtype", "fp32"))


def split_samples(feats: dict[str, Tensor], B: int) -> dict[str, Tensor]:
    """``[B * S, L, D] -> [B, S, L, D]`` (rows ordered segment-major, as ``repeat_interleave`` makes them)."""
    return {k: v.reshape(B, -1, *v.shape[1:]) for k, v in feats.items()}


def shifted_views(audio: Tensor, shifts: Sequence[int] = ()) -> Tensor:
    """``[B, n] -> [B, 1 + len(shifts), n]``: the segment and copies delayed by ``shifts`` samples (reflect-padded)."""
    if not shifts:
        return audio[:, None]
    m, n = max(abs(s) for s in shifts), audio.shape[-1]
    x = F.pad(audio[:, None], (m, m), mode="reflect")[:, 0]
    return torch.stack([audio] + [x[:, m - s: m - s + n] for s in shifts], 1)


def _pool(t: Tensor, idx: Tensor | None) -> Tensor:
    """``[B, S, L, D] -> [1, B * S * L', 1, D]``: one drift problem over every location of every segment."""
    if idx is not None:
        t = t[:, :, idx]
    return t.reshape(1, -1, 1, t.shape[-1])


def vocoder_drift_loss(
    gen: dict[str, Tensor],
    pos: dict[str, Tensor],
    pairing: dict[str, float],
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    key_weights: dict[str, float] | None = None,
    reduce: str = "mean",
    max_locations: int | None = 1024,
    pooled_locations: int | None = 128,
    taus: dict[str, Tensor] | None = None,
    affinity_floor: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted sum of the drift losses of the enabled pairings.

    Args:
        gen: ``{map: [B, S, L, D]}``: ``S`` generated samples per segment. pos: ``{map: [B, P, L, D]}``: the real
            segment (``P = 1``) or the segment and shifted views of it.
        pairing: ``{"conditional": w, "pooled": w}``. ``conditional``: a problem per (segment, location) with the
            ``S`` samples (siblings as negatives) and the ``P`` real views; ``max_locations`` subsamples large maps.
            ``pooled``: a problem per map over all segments and ``pooled_locations`` random locations: ``B * S * L'``
            generated and ``B * P * L'`` real patches, the local-patch distribution (a patch discriminator's view).
        reduce: ``mean`` over the maps (weighted) or ``sum`` (a step that back-propagates the feature space part by
            part divides the summed gradients by :func:`map_weight_sum` of all parts).
        taus: ``drift.mode: kyutai``: one learned temperature per pairing (else the fixed ``temperatures``).

    Returns the loss and an info dict keyed ``{pairing}/{stat}/{map}`` (see :func:`feature_drift_loss`).
    """
    total, info = None, {}
    for mode, w in pairing.items():
        if mode not in PAIRINGS:
            raise ValueError(f"drift.pairing keys must be in {PAIRINGS}, got {mode!r}")
        if not w:
            continue
        g, p, ml = gen, pos, max_locations
        if mode == "pooled":
            g, p, ml = {}, {}, None
            for k, v in gen.items():
                L = v.shape[2]
                idx = torch.randperm(L, device=v.device)[:pooled_locations] if pooled_locations and \
                    L > pooled_locations else None
                g[k], p[k] = _pool(v, idx), _pool(pos[k], idx)
        loss, inf = feature_drift_loss(g, p, temperatures=tuple(temperatures), key_weights=key_weights,
                                       affinity_floor=affinity_floor, reduce=reduce,
                                       taus=None if taus is None else taus[mode], max_locations=ml)
        total = w * loss if total is None else total + w * loss
        info.update({f"{mode}/{k}": v for k, v in inf.items()})
    if total is None:
        raise ValueError("drift.pairing: no pairing with a non-zero weight")
    return total, info


def map_weight_sum(keys, key_weights: dict[str, float] | None = None) -> float:
    """The normaliser of ``reduce: mean``: the summed weights of the feature maps."""
    return sum(key_weight(k, key_weights) for k in keys)


def summarize(info: dict[str, Tensor]) -> dict[str, float]:
    """Mean over maps of the raw (pre-normalisation) drift norms, temperatures and data mass, per pairing."""
    groups: dict[str, list[Tensor]] = {}
    for k, v in info.items():
        mode, stat = k.split("/")[:2]
        if stat.startswith(("force", "tau", "p_data")):
            groups.setdefault(f"{mode[:4]}_{stat}", []).append(v.detach().float().reshape(()))
    return {k: torch.stack(v).mean().item() for k, v in groups.items()}
