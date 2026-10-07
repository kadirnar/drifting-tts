"""2-D Mel-MAE: a ResNet masked autoencoder over (frequency x time).

The 1-D Mel-MAE treats the 100 mel bins as channels and compresses them in its first layer, so
harmonic structure along frequency barely reaches the drift kernel. This variant follows the
paper's latent-MAE more literally (a 2-D GroupNorm ResNet on an image-like input, with an input
patchification as in the official pixel-space MAE) and DriftTTS's MelMAE: mels are treated as
single-channel images, so local frequency patterns (harmonic stripes, formants) become features.
``get_activations`` returns the same families of multi-scale features as the 1-D model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mae import _gn, _std, mel_features


class BasicBlock2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.gn1 = _gn(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.gn2 = _gn(c_out)
        self.proj = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), _gn(c_out))

    def forward(self, x: Tensor) -> Tensor:
        y = self.gn2(self.conv2(F.relu(self.gn1(self.conv1(x)))))
        return F.relu(y + (x if self.proj is None else self.proj(x)))


class UpBlock2d(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.norm = _gn(c_in + c_skip)
        self.conv1 = nn.Sequential(nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), _gn(c_out), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), _gn(c_out), nn.ReLU())

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(self.norm(torch.cat([x, skip], 1))))


class MelMAE2d(nn.Module):
    def __init__(
        self,
        n_mels: int = 100,
        base_channels: int = 32,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        input_patch: tuple[int, int] = (2, 2),
        num_classes: int = 0,
        mask_patch: tuple[int, int] = (10, 8),
    ):
        super().__init__()
        self.n_mels, self.input_patch, self.mask_patch = n_mels, tuple(input_patch), tuple(mask_patch)
        pf, pt = self.input_patch
        self.conv_in = nn.Sequential(nn.Conv2d(pf * pt, base_channels, 3, padding=1, bias=False),
                                     _gn(base_channels), nn.ReLU())
        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        chans = [base_channels]
        c = base_channels
        for i, n in enumerate(layers):
            c_out = base_channels * 2**i
            blocks = [BasicBlock2d(c, c_out, stride=1 if i == 0 else 2)]
            blocks += [BasicBlock2d(c_out, c_out) for _ in range(n - 1)]
            self.stages.append(nn.ModuleList(blocks))
            self.stage_norms.append(_gn(c_out))
            chans.append(c_out)
            c = c_out
        self.bridge = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), _gn(c), nn.ReLU())
        ups = []
        for skip in reversed(chans[:-1]):
            ups.append(UpBlock2d(c, skip, skip))
            c = skip
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(c, pf * pt, 1)
        self.cls_head = nn.Linear(chans[-1], num_classes) if num_classes > 0 else None

    # ------------------------------------------------------------------ patchify
    def patchify(self, mel: Tensor) -> Tensor:
        """``[B, F, T] -> [B, pf*pt, F/pf, T/pt]`` (pads F and T to multiples of the patch)."""
        pf, pt = self.input_patch
        B, Fr, T = mel.shape
        mel = F.pad(mel, (0, (-T) % pt, 0, (-Fr) % pf))
        Fr2, T2 = mel.shape[1:]
        x = mel.reshape(B, Fr2 // pf, pf, T2 // pt, pt).permute(0, 2, 4, 1, 3)
        return x.reshape(B, pf * pt, Fr2 // pf, T2 // pt)

    def unpatchify(self, x: Tensor, Fr: int, T: int) -> Tensor:
        pf, pt = self.input_patch
        B, _, f, t = x.shape
        x = x.reshape(B, pf, pt, f, t).permute(0, 3, 1, 4, 2).reshape(B, f * pf, t * pt)
        return x[:, :Fr, :T]

    # ------------------------------------------------------------------ encoder
    def encode(self, mel: Tensor, return_blocks: bool = False):
        feats = {"conv1": self.conv_in(self.patchify(mel))}
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
    def make_mask(self, B: int, Fr: int, T: int, ratio: float, device) -> Tensor:
        """``[B, F, T]`` mask (1 = hidden) over (frequency x time) blocks."""
        mf, mt = self.mask_patch
        nf, nt = -(-Fr // mf), -(-T // mt)
        m = (torch.rand(B, nf, nt, device=device) < ratio).float()
        return m.repeat_interleave(mf, 1).repeat_interleave(mt, 2)[:, :Fr, :T]

    def forward(self, mel: Tensor, mask_ratio: float = 0.5, labels: Tensor | None = None,
                cls_weight: float = 0.0) -> tuple[Tensor, dict[str, Tensor]]:
        B, Fr, T = mel.shape
        mask = self.make_mask(B, Fr, T, mask_ratio, mel.device)
        feats = self.encode(mel * (1 - mask))
        x = self.bridge(feats["layer4"])
        for up, skip in zip(self.ups, [feats[k] for k in ("layer3", "layer2", "layer1", "conv1")]):
            x = up(x, skip)
        recon = self.unpatchify(self.head(x), Fr, T)
        err = (recon - mel) ** 2
        recon_loss = (err * mask).sum() / mask.sum().clamp_min(1.0)
        metrics = {"recon_loss": recon_loss.detach(),
                   "visible_loss": ((err * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1.0)).detach()}
        loss = recon_loss
        if self.cls_head is not None and labels is not None and cls_weight > 0:
            logits = self.cls_head(feats["layer4"].mean((-1, -2)))
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
        """Multi-scale features ``[B, L, D]``: per-location (frequency x time), windowed and global
        mean / std (as in the official ``get_activations``), plus a frequency profile per stage
        (time-averaged, one location per frequency row) that describes the spectral shape."""
        out = mel_features(mel, mel_patch, spectral_detail)
        need_blocks = every_k_block >= 1
        enc = self.encode(mel, return_blocks=need_blocks)
        feats, blocks = enc if need_blocks else (enc, {})

        def add(name: str, f: Tensor) -> None:  # f: [B, C, F', T']
            B, C, fr, t = f.shape
            out[name] = f.flatten(2).transpose(1, 2)
            out[f"{name}_fprofile"] = f.mean(-1).transpose(1, 2)  # [B, F', C]
            if use_mean:
                out[f"{name}_mean"] = f.mean((-1, -2))[:, None]
            if use_std:
                out[f"{name}_std"] = _std(f.flatten(2), -1)[:, None]
            for s in patch_sizes:
                if fr >= s and t >= s:  # drop the incomplete last row / column of windows
                    g = f[:, :, : fr // s * s, : t // s * s]
                    w = g.reshape(B, C, fr // s, s, t // s, s).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1, s * s)
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
