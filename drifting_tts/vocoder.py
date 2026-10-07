"""Vocoders for the 24 kHz, 100-bin log-mel backends: Vocos (default) and BigVGAN-v2."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
from torch import Tensor


def build_vocos(init: str):
    """``init``: a HF repo id (pretrained weights) or a local Vocos YAML config (random initialisation)."""
    from vocos import Vocos

    if init.endswith((".yaml", ".yml")) and Path(init).exists():
        return Vocos.from_hparams(init)
    return Vocos.from_pretrained(init)


BIGVGAN_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"


def bigvgan_snapshot(repo: str = BIGVGAN_REPO, weights: bool = True) -> tuple[types.ModuleType, str]:
    """NVIDIA BigVGAN-v2 (MIT): its code ships with the HF repo (``pip install drifting-tts[bigvgan]``).
    Returns the imported ``bigvgan`` module and the snapshot path (with the generator weights if ``weights``)."""
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
    """BigVGAN-v2 generator *with* weight norm, as trained. ``hparams`` override ``config.json`` (e.g. a tiny model
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


def load_bigvgan(repo: str = BIGVGAN_REPO, device: str = "cuda", finetuned: str | None = None,
                 cuda_kernel: bool = False):
    """Inference generator (weight norm removed): the released weights, or a ``bigvgan_ft.pt`` written by
    ``drifting-tts finetune-vocoder`` (same ``{"generator": ...}`` layout as ``bigvgan_generator.pt``)."""
    if finetuned:
        ck = torch.load(finetuned, map_location="cpu", weights_only=False)
        model = build_bigvgan(ck.get("repo", repo), ck.get("hparams"), pretrained=False, cuda_kernel=cuda_kernel)
        model.load_state_dict(ck["generator"])
    else:
        model = build_bigvgan(repo, cuda_kernel=cuda_kernel)
    model.remove_weight_norm()
    return model.to(device).eval()


class Vocoder:
    def __init__(self, device: str = "cuda", repo: str = "charactr/vocos-mel-24khz", finetuned: str | None = None,
                 backend: str = "vocos", cuda_kernel: bool = False):
        """``backend``: ``vocos`` or ``bigvgan`` (must match the mels the model was trained on).
        ``finetuned``: a ``vocos_ft.pt`` / ``bigvgan_ft.pt`` written by ``drifting-tts finetune-vocoder``.
        ``cuda_kernel``: BigVGAN's fused CUDA activation (~3x faster inference; built with nvcc on first use)."""
        self.backend, self.device = backend, device
        if backend == "bigvgan":
            self.model = load_bigvgan(device=device, finetuned=finetuned, cuda_kernel=cuda_kernel)
            return
        if finetuned:
            ck = torch.load(finetuned, map_location="cpu", weights_only=False)
            self.model = build_vocos(ck.get("init", repo))
            self.model.load_state_dict(ck["vocos"])
        else:
            self.model = build_vocos(repo)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, log_mel: Tensor) -> Tensor:
        """Unnormalised log-mel ``[B, 100, T]`` -> waveform ``[B, samples]`` at 24 kHz."""
        x = log_mel.to(self.device).float()
        wav = self.model(x)[:, 0] if self.backend == "bigvgan" else self.model.decode(x)
        return wav.clamp(-1, 1)
