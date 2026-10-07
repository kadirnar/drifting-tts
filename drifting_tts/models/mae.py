"""Mel-MAE: a 1-D ResNet masked autoencoder whose multi-scale activations define the drift kernel.

Mel-domain analogue of the latent-MAE ResNet of the official release (``models/mae_model.py``):
GroupNorm basic blocks in four stages, a U-Net decoder for masked reconstruction, an optional
classification head (speaker classification plays the role of ImageNet fine-tuning), and
``get_activations`` producing the same family of multi-scale features (per-location, global
mean / std, patch mean / std, every-k-block outputs) plus raw-mel features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gn(c: int) -> nn.GroupNorm:
    g = min(32, c)
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


class BasicBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, kernel: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride, kernel // 2, bias=False)
        self.gn1 = _gn(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, 1, kernel // 2, bias=False)
        self.gn2 = _gn(c_out)
        self.proj = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), _gn(c_out))

    def forward(self, x: Tensor) -> Tensor:
        y = self.gn2(self.conv2(F.relu(self.gn1(self.conv1(x)))))
        return F.relu(y + (x if self.proj is None else self.proj(x)))


class UpBlock1d(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.norm = _gn(c_in + c_skip)
        self.conv1 = nn.Sequential(nn.Conv1d(c_in + c_skip, c_out, 3, padding=1, bias=False), _gn(c_out), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(c_out, c_out, 3, padding=1, bias=False), _gn(c_out), nn.ReLU())

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.conv2(self.conv1(self.norm(torch.cat([x, skip], 1))))


def _std(x: Tensor, dim, eps: float = 1e-6) -> Tensor:
    x = x.float()
    return (x.var(dim=dim, unbiased=False) + eps).sqrt()


def spectral_detail(mel: Tensor, kernel: int = 5) -> Tensor:
    """Mel minus its frequency-smoothed envelope: harmonic peaks / valleys (``[B, n_mels, T]``)."""
    pad = kernel // 2
    x = mel.transpose(1, 2)  # [B, T, F] -> pool along frequency
    smooth = F.avg_pool1d(F.pad(x.reshape(-1, 1, x.shape[-1]), (pad, pad), mode="replicate"), kernel, 1)
    return mel - smooth.reshape(x.shape).transpose(1, 2)


def mel_features(mel: Tensor, mel_patch: int = 4, detail: bool = False) -> dict[str, Tensor]:
    """Raw-mel features for the drift kernel (the official code adds the flattened input as "global").

    With ``detail=True`` the same features are added for :func:`spectral_detail`, so that the depth of
    harmonic peaks / valleys enters the kernel distance directly.
    """
    out: dict[str, Tensor] = {}
    sources = [("mel", mel)] + ([("detail", spectral_detail(mel))] if detail else [])
    for name, x in sources:
        B, C, T = x.shape
        out[f"{name}_global"] = x.reshape(B, 1, C * T)
        if T % mel_patch == 0:
            out[f"{name}_patch{mel_patch}"] = x.reshape(B, C, T // mel_patch, mel_patch).permute(0, 2, 1, 3).reshape(
                B, T // mel_patch, C * mel_patch)
        out[f"{name}_mean"] = x.mean(-1)[:, None]
        out[f"{name}_std"] = _std(x, -1)[:, None]
    return out


class MelMAE(nn.Module):
    def __init__(
        self,
        n_mels: int = 100,
        base_channels: int = 64,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        kernel: int = 5,
        num_classes: int = 0,
        mask_patch: int = 8,
    ):
        super().__init__()
        self.n_mels, self.mask_patch = n_mels, mask_patch
        self.conv_in = nn.Sequential(nn.Conv1d(n_mels, base_channels, kernel, padding=kernel // 2, bias=False),
                                     _gn(base_channels), nn.ReLU())
        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        chans = [base_channels]
        c = base_channels
        for i, n in enumerate(layers):
            c_out = base_channels * 2**i
            blocks = [BasicBlock1d(c, c_out, stride=1 if i == 0 else 2, kernel=kernel)]
            blocks += [BasicBlock1d(c_out, c_out, kernel=kernel) for _ in range(n - 1)]
            self.stages.append(nn.ModuleList(blocks))
            self.stage_norms.append(_gn(c_out))
            chans.append(c_out)
            c = c_out
        # U-Net decoder: layer4 -> layer3 -> layer2 -> layer1 -> conv_in
        self.bridge = nn.Sequential(nn.Conv1d(c, c, 3, padding=1, bias=False), _gn(c), nn.ReLU())
        ups = []
        for skip in reversed(chans[:-1]):
            ups.append(UpBlock1d(c, skip, skip))
            c = skip
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv1d(c, n_mels, 1)
        self.cls_head = nn.Linear(chans[-1], num_classes) if num_classes > 0 else None

    # ------------------------------------------------------------------ encoder
    def encode(self, x: Tensor, return_blocks: bool = False):
        feats = {"conv1": self.conv_in(x)}
        blocks: dict[str, list[Tensor]] = {}
        h = feats["conv1"]
        for i, (stage, norm) in enumerate(zip(self.stages, self.stage_norms)):
            outs = []
            for block in stage:
                h = block(h)
                outs.append(h)
            blocks[f"layer{i + 1}"] = outs
            h = norm(h)
            feats[f"layer{i + 1}"] = h
        return (feats, blocks) if return_blocks else feats

    # ------------------------------------------------------------------ MAE
    def make_mask(self, B: int, T: int, ratio: float, device) -> Tensor:
        """``[B, 1, T]`` mask (1 = hidden) over contiguous time patches."""
        n = -(-T // self.mask_patch)
        m = (torch.rand(B, n, device=device) < ratio).float()
        return m.repeat_interleave(self.mask_patch, dim=1)[:, None, :T]

    def forward(self, mel: Tensor, mask_ratio: float = 0.5, labels: Tensor | None = None,
                cls_weight: float = 0.0) -> tuple[Tensor, dict[str, Tensor]]:
        """Masked reconstruction (+ optional classification) loss on ``[B, n_mels, T]`` mels."""
        mask = self.make_mask(mel.shape[0], mel.shape[-1], mask_ratio, mel.device)
        feats = self.encode(mel * (1 - mask))
        x = self.bridge(feats["layer4"])
        skips = [feats[k] for k in ("layer3", "layer2", "layer1", "conv1")]
        for up, skip in zip(self.ups, skips):
            x = up(x, skip)
        recon = self.head(x)
        err = ((recon - mel) ** 2).mean(1, keepdim=True)
        recon_loss = (err * mask).sum() / mask.sum().clamp_min(1.0)
        metrics = {"recon_loss": recon_loss.detach(), "visible_loss": ((err * (1 - mask)).sum()
                   / (1 - mask).sum().clamp_min(1.0)).detach()}
        loss = recon_loss
        if self.cls_head is not None and labels is not None and cls_weight > 0:
            logits = self.cls_head(feats["layer4"].mean(-1))
            cls_loss = F.cross_entropy(logits, labels)
            loss = (1 - cls_weight) * recon_loss + cls_weight * cls_loss
            metrics["cls_loss"] = cls_loss.detach()
            metrics["cls_acc"] = (logits.argmax(-1) == labels).float().mean()
        return loss, metrics

    # ------------------------------------------------------------------ drift features
    def get_activations(
        self,
        mel: Tensor,
        patch_sizes: tuple[int, ...] = (2, 4),
        every_k_block: int = 0,
        use_mean: bool = True,
        use_std: bool = True,
        mel_patch: int = 4,
        spectral_detail: bool = False,
    ) -> dict[str, Tensor]:
        """Multi-scale features for the drift loss. Every value is ``[B, L, D]`` (L locations).

        Mirrors ``MAEResNetJAX.get_activations``: per-location features of every stage, their
        global mean / std, and mean / std over non-overlapping windows; plus raw-mel features.
        """
        out = mel_features(mel, mel_patch, spectral_detail)

        need_blocks = every_k_block >= 1
        enc = self.encode(mel, return_blocks=need_blocks)
        feats, blocks = enc if need_blocks else (enc, {})

        def add(name: str, f: Tensor) -> None:  # f: [B, D, T']
            t = f.shape[-1]
            out[name] = f.transpose(1, 2)
            if use_mean:
                out[f"{name}_mean"] = f.mean(-1)[:, None]
            if use_std:
                out[f"{name}_std"] = _std(f, -1)[:, None]
            for s in patch_sizes:
                if t % s == 0 and t // s >= 1:
                    w = f.reshape(f.shape[0], f.shape[1], t // s, s)
                    if use_mean:
                        out[f"{name}_mean{s}"] = w.mean(-1).transpose(1, 2)
                    if use_std:
                        out[f"{name}_std{s}"] = _std(w, -1).transpose(1, 2)

        for name, f in feats.items():
            add(name, f)
        if need_blocks:
            for lname, outs in blocks.items():
                for j, f in enumerate(outs, start=1):  # pre-GroupNorm block outputs, the last one included
                    if j % every_k_block == 0:
                        add(f"{lname}_blk{j}", f)
        return out
