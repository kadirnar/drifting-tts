"""Convert the PyTorch checkpoints to MLX weights (needs torch; run once).

Writes ``model.safetensors`` (acoustic model), ``vocoder.safetensors`` (BigVGAN-v2, weight norm removed) and
``config.json`` (architecture, mel statistics, voices, duration factors). The acoustic model is stored in float32,
since float16 rounding of its weights costs audible precision (31.9 dB end-to-end SNR). The vocoder is stored in
float16 except its snake parameters (59 dB), unless ``--fp32``. The loaders cast everything back to float32.

    python -m drifting_tts.mlx.convert --model drifting_tts_v3.1.pt --vocoder bigvgan_v2_ft.pt --out mlx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..text import SYMBOLS
from ..voices import DEFAULT_VOICE, VOICES


def convert_acoustic(state_dict: dict) -> dict[str, np.ndarray]:
    """PyTorch ``DriftingTTS`` state dict -> MLX arrays keyed like :class:`drifting_tts.mlx.model.DriftingTTS`."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("generator.step_embed"):  # multi-step drifting only
            continue
        a = v.float().numpy() if hasattr(v, "numpy") else np.asarray(v, dtype=np.float32)
        if a.ndim == 3 and k.endswith(".weight"):  # Conv1d [out, in, k] -> MLX [out, k, in]
            a = a.transpose(0, 2, 1)
        out[k] = np.ascontiguousarray(a)
    return out


def save(weights: dict[str, np.ndarray], path: Path, fp16: bool) -> None:
    """``fp16``: store in float16, except the BigVGAN snake ``alpha`` / ``beta`` (a few kB; 59 vs 56 dB SNR)."""
    import mlx.core as mx

    def dtype(k: str):
        return mx.float16 if fp16 and not k.endswith((".act.alpha", ".act.beta")) else mx.float32

    mx.save_safetensors(str(path), {k: mx.array(v).astype(dtype(k)) for k, v in weights.items()})


def main() -> None:
    import torch

    from ..vocoder import load_bigvgan
    from .bigvgan import convert_bigvgan

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", required=True, help="fine-tuned BigVGAN-v2 (bigvgan_ft.pt)")
    p.add_argument("--out", default="mlx")
    p.add_argument("--fp32", action="store_true", help="store the vocoder in float32 too (450 MB instead of 225 MB)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    save(convert_acoustic(ck["ema"]), out / "model.safetensors", fp16=False)
    bigvgan = load_bigvgan(device="cpu", finetuned=args.vocoder)
    save(convert_bigvgan(bigvgan.state_dict()), out / "vocoder.safetensors", not args.fp32)
    hparams = {k: v for k, v in dict(bigvgan.h).items() if k in VOCODER_KEYS}

    config = {
        "model": ck["config"]["model"], "num_speakers": ck["num_speakers"], "n_vocab": len(SYMBOLS), "n_mels": 100,
        "sample_rate": 24000, "hop_length": 256, "stats": {k: ck["stats"][k] for k in ("mean", "std")},
        "duration_scale": float(ck.get("duration_scale", 1.0)),
        "duration_scales": {str(k): float(v) for k, v in ck.get("duration_scales", {}).items()},
        "temperature": float(ck.get("temperature") or 0.3),
        "voices": {k: v["id"] for k, v in VOICES.items()}, "default_voice": DEFAULT_VOICE, "vocoder": hparams,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"wrote {out}: " + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.0f} MB" for f in sorted(out.iterdir())))


# the BigVGAN hyper-parameters that define the generator (the rest of its config.json is about training)
VOCODER_KEYS = ("resblock", "upsample_rates", "upsample_kernel_sizes", "upsample_initial_channel",
                "resblock_kernel_sizes", "resblock_dilation_sizes", "use_tanh_at_final", "use_bias_at_final",
                "activation", "snake_logscale", "num_mels", "hop_size", "sampling_rate")


if __name__ == "__main__":
    main()
