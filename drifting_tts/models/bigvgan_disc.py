"""BigVGAN-v2 discriminators and losses, ported from NVIDIA/BigVGAN (MIT): ``discriminators.py`` / ``loss.py``.

The released ``bigvgan_discriminator_optimizer.pt`` stores a multi-period discriminator (``mpd``) and, under the
key ``mrd``, the multi-scale sub-band CQT discriminator (v2) or the multi-resolution discriminator (v1 and
BigVGAN-base, ``"discriminator": "mrd"`` in their ``config.json``). Module names and the (old-style) weight norm match
the reference, so those state dicts load with ``strict=True``. The CQT front end needs ``nnAudio``.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

Outputs = tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]


def _wn(m: nn.Module) -> nn.Module:
    """``torch.nn.utils.weight_norm`` (``weight_g`` / ``weight_v``, as in the released checkpoints)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return nn.utils.weight_norm(m)


def _pad2d(k: tuple[int, int], d: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    return (k[0] - 1) * d[0] // 2, (k[1] - 1) * d[1] // 2


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, mult: float = 1.0):
        super().__init__()
        self.period = period
        ch = [1] + [int(c * mult) for c in (32, 128, 512, 1024, 1024)]
        self.convs = nn.ModuleList([_wn(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1) if i < 4 else 1, padding=(2, 0)))
                                    for i in range(5)])
        self.conv_post = _wn(nn.Conv2d(ch[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
        x = x.view(b, c, -1, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, h: dict):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p, h.get("discriminator_channel_mult", 1))
                                             for p in h["mpd_reshapes"]])

    def forward(self, y: Tensor, y_hat: Tensor) -> Outputs:
        return _run(self.discriminators, y, y_hat)


class DiscriminatorCQT(nn.Module):
    """Sub-band CQT discriminator: complex CQT of the 2x-upsampled signal, one conv per octave, 2-D conv stack."""

    def __init__(self, h: dict, hop_length: int, n_octaves: int, bins_per_octave: int):
        super().__init__()
        from nnAudio.features.cqt import CQT2010v2

        sr, filters, max_filters, scale = h["sampling_rate"], h["cqtd_filters"], h["cqtd_max_filters"], \
            h["cqtd_filters_scale"]
        in_ch, out_ch = h.get("cqtd_in_channels", 1), h.get("cqtd_out_channels", 1)
        self.n_octaves, self.bins_per_octave = n_octaves, bins_per_octave
        self.normalize_volume = h.get("cqtd_normalize_volume", False)
        self.cqt_transform = CQT2010v2(sr=sr * 2, hop_length=hop_length, n_bins=bins_per_octave * n_octaves,
                                       bins_per_octave=bins_per_octave, output_format="Complex", pad_mode="constant",
                                       verbose=False)
        k = (3, 9)
        self.conv_pres = nn.ModuleList([nn.Conv2d(in_ch * 2, in_ch * 2, k, padding=_pad2d(k))
                                        for _ in range(n_octaves)])
        convs = [nn.Conv2d(in_ch * 2, filters, k, padding=_pad2d(k))]
        c_in = min(scale * filters, max_filters)
        for i, d in enumerate(h["cqtd_dilations"]):
            c_out = min(scale ** (i + 1) * filters, max_filters)
            convs.append(_wn(nn.Conv2d(c_in, c_out, k, stride=(1, 2), dilation=(d, 1), padding=_pad2d(k, (d, 1)))))
            c_in = c_out
        c_out = min(scale ** (len(h["cqtd_dilations"]) + 1) * filters, max_filters)
        convs.append(_wn(nn.Conv2d(c_in, c_out, (3, 3), padding=(1, 1))))
        self.convs = nn.ModuleList(convs)
        self.conv_post = _wn(nn.Conv2d(c_out, out_ch, (3, 3), padding=(1, 1)))
        self.resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr * 2)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        if self.normalize_volume:
            x = x - x.mean(dim=-1, keepdim=True)
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        with torch.autocast(x.device.type, enabled=False):  # resampling + CQT in fp32
            z = self.cqt_transform(self.resample(x.float()))  # [B, bins, T', (re, im)]
        z = z.permute(0, 3, 2, 1)  # [B, 2, T', bins]
        bpo = self.bins_per_octave
        z = torch.cat([conv(z[..., i * bpo: (i + 1) * bpo]) for i, conv in enumerate(self.conv_pres)], dim=-1)
        fmap = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), 0.1)
            fmap.append(z)
        return self.conv_post(z), fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, h: dict):
        super().__init__()
        h = {"cqtd_filters": 32, "cqtd_max_filters": 1024, "cqtd_filters_scale": 1, "cqtd_dilations": [1, 2, 4],
             "cqtd_hop_lengths": [512, 256, 256], "cqtd_n_octaves": [9, 9, 9],
             "cqtd_bins_per_octaves": [24, 36, 48], **h}
        self.discriminators = nn.ModuleList([DiscriminatorCQT(h, hop, n, b) for hop, n, b in zip(
            h["cqtd_hop_lengths"], h["cqtd_n_octaves"], h["cqtd_bins_per_octaves"])])

    def forward(self, y: Tensor, y_hat: Tensor) -> Outputs:
        return _run(self.discriminators, y, y_hat)


class DiscriminatorR(nn.Module):
    """Magnitude-spectrogram discriminator at one STFT resolution ``(n_fft, hop, win)`` (BigVGAN v1, UniVNet)."""

    def __init__(self, resolution: list[int], mult: float = 1.0):
        super().__init__()
        self.resolution = resolution
        c = int(32 * mult)
        self.convs = nn.ModuleList([_wn(nn.Conv2d(1, c, (3, 9), padding=(1, 4)))]
                                   + [_wn(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
                                   + [_wn(nn.Conv2d(c, c, (3, 3), padding=(1, 1)))])
        self.conv_post = _wn(nn.Conv2d(c, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x: Tensor) -> Tensor:
        n_fft, hop, win = self.resolution
        x = F.pad(x, ((n_fft - hop) // 2, (n_fft - hop) // 2), mode="reflect").squeeze(1)
        # no window argument: a rectangular window, as in the reference
        return torch.stft(x, n_fft, hop, win, center=False, return_complex=True).abs()

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        fmap = []
        x = self.spectrogram(x)[:, None]
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, h: dict):
        super().__init__()
        mult = h.get("mrd_channel_mult", h.get("discriminator_channel_mult", 1))
        self.discriminators = nn.ModuleList([DiscriminatorR(r, mult) for r in h["resolutions"]])

    def forward(self, y: Tensor, y_hat: Tensor) -> Outputs:
        return _run(self.discriminators, y, y_hat)


def _run(discriminators: nn.ModuleList, y: Tensor, y_hat: Tensor) -> Outputs:
    """Real / generated scores and feature maps of every sub-discriminator (``y``, ``y_hat``: ``[B, 1, T]``)."""
    out: Outputs = ([], [], [], [])
    for d in discriminators:
        (s_r, f_r), (s_g, f_g) = d(y), d(y_hat)
        for lst, v in zip(out, (s_r, s_g, f_r, f_g)):
            lst.append(v)
    return out


def discriminator_loss(real: list[Tensor], fake: list[Tensor]) -> Tensor:
    """LSGAN: ``sum_k mean((1 - D_k(y))^2) + mean(D_k(y_hat)^2)``."""
    return sum(torch.mean((1 - r.float()) ** 2) + torch.mean(g.float() ** 2) for r, g in zip(real, fake))


def generator_loss(fake: list[Tensor]) -> Tensor:
    return sum(torch.mean((1 - g.float()) ** 2) for g in fake)


def feature_loss(fmap_r: list[list[Tensor]], fmap_g: list[list[Tensor]]) -> Tensor:
    """L1 feature matching summed over layers and sub-discriminators, weight 2 as in the reference."""
    return 2 * sum(torch.mean(torch.abs(r.float() - g.float())) for dr, dg in zip(fmap_r, fmap_g)
                   for r, g in zip(dr, dg))


class MultiScaleMelLoss(nn.Module):
    """BigVGAN-v2 (DAC) multi-scale log10-mel L1: 7 STFTs (windows 32..2048, hop w/4, centred, Hann), 5..320 Slaney
    mel bands, ``log10(clamp(mel, 1e-5))``, summed over scales."""

    def __init__(self, sample_rate: int = 24_000, n_mels=(5, 10, 20, 40, 80, 160, 320),
                 windows=(32, 64, 128, 256, 512, 1024, 2048)):
        super().__init__()
        self.windows = list(windows)
        with warnings.catch_warnings():  # the small windows have more mel bands than FFT bins (empty filters)
            warnings.simplefilter("ignore", UserWarning)
            for w, m in zip(windows, n_mels):
                fb = torchaudio.functional.melscale_fbanks(w // 2 + 1, 0.0, sample_rate / 2, m, sample_rate,
                                                           norm="slaney", mel_scale="slaney")
                self.register_buffer(f"fb{w}", fb, persistent=False)
                self.register_buffer(f"win{w}", torch.hann_window(w), persistent=False)

    def log_mel(self, x: Tensor, w: int) -> Tensor:
        spec = torch.stft(x, w, w // 4, window=getattr(self, f"win{w}"), center=True, return_complex=True).abs()
        return torch.log10((spec.transpose(-1, -2) @ getattr(self, f"fb{w}")).clamp_min(1e-5))

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """``y_hat``, ``y``: ``[B, 1, T]`` or ``[B, T]`` waveforms."""
        y_hat, y = y_hat.reshape(-1, y_hat.shape[-1]).float(), y.reshape(-1, y.shape[-1]).float()
        return sum(F.l1_loss(self.log_mel(y_hat, w), self.log_mel(y, w)) for w in self.windows)
