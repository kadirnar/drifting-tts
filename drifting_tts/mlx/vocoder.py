"""The MLX vocoders: converted ``.safetensors`` files in the ``mlx/`` folder of the model repo, each recording its
architecture in the safetensors metadata (:data:`METADATA_KEY`)."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .bigvgan import BigVGAN
from .vocos import Vocos

# registry name (as in drifting_tts.vocoder.VOCODERS) -> file in the mlx/ folder
VOCODERS = {
    "bigvgan-v2-ft": "vocoder.safetensors",
    "bigvgan-base-ft": "bigvgan_base_ft.safetensors",
    "vocos-ft": "vocos_ft.safetensors",
}
DEFAULT_VOCODER = "bigvgan-v2-ft"
METADATA_KEY = "drifting_tts.vocoder"
KINDS = {"bigvgan": BigVGAN, "vocos": Vocos}


def vocoder_path(spec: str | Path | None, root: str | Path) -> Path:
    """A registry name (its file in ``root``; ``None``: the default) or the path of a converted vocoder."""
    spec = DEFAULT_VOCODER if spec is None else spec
    if str(spec) in VOCODERS:
        path = Path(root) / VOCODERS[str(spec)]
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: not published at this revision, or convert it with "
                                    f"python -m drifting_tts.mlx.convert --vocoder {spec} --out {root}")
        return path
    if Path(spec).is_file():
        return Path(spec)
    raise ValueError(f"unknown MLX vocoder {spec!r}: expected a .safetensors path or one of {', '.join(VOCODERS)}")


def save_vocoder(path: str | Path, weights: dict[str, np.ndarray], kind: str, hparams: dict, fp16: bool) -> None:
    """Write converted weights with their architecture. ``fp16``: float16 storage, except BigVGAN's snake
    ``alpha`` / ``beta`` (a few kB, 59 vs 56 dB parity for BigVGAN-v2)."""
    if kind not in KINDS:
        raise ValueError(f"unknown vocoder kind {kind!r}")

    def dtype(k: str):
        return mx.float16 if fp16 and not k.endswith((".act.alpha", ".act.beta")) else mx.float32

    meta = {METADATA_KEY: json.dumps({"kind": kind, "hparams": hparams})}
    mx.save_safetensors(str(path), {k: mx.array(v).astype(dtype(k)) for k, v in weights.items()}, metadata=meta)


def load_vocoder(path: str | Path, hparams: dict | None = None, fused_activations: bool = False):
    """A converted vocoder (:class:`BigVGAN` or :class:`Vocos`), weights cast to float32 and materialised.

    ``hparams``: BigVGAN hyper-parameters for a file without metadata (``vocoder.safetensors`` of the first release,
    described by ``config.json``). ``fused_activations``: BigVGAN's anti-aliased activations as Metal kernels
    (:meth:`BigVGAN.use_fused_activations`)."""
    weights, meta = mx.load(str(path), return_metadata=True)
    if METADATA_KEY in meta:
        spec = json.loads(meta[METADATA_KEY])
    elif hparams is not None:
        spec = {"kind": "bigvgan", "hparams": hparams}
    else:
        raise ValueError(f"{path} does not describe its vocoder; pass its BigVGAN hparams")
    model = KINDS[spec["kind"]](spec["hparams"])
    model.load_weights([(k, v.astype(mx.float32)) for k, v in weights.items()])
    mx.eval(model.parameters())
    model.kind = spec["kind"]
    if fused_activations and spec["kind"] == "bigvgan":
        model.use_fused_activations()
    return model.prepare_for_inference()
