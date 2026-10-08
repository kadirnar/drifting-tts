"""Convert the PyTorch checkpoints to MLX weights (needs torch; run once).

``--model`` writes ``model.safetensors`` (acoustic model) and ``config.json`` (architecture, mel statistics, voices,
duration factors). Each ``--vocoder`` writes its file of :data:`drifting_tts.mlx.vocoder.VOCODERS` with its
architecture in the safetensors metadata. The acoustic model is stored in float32, since float16 rounding of its
weights costs audible precision (31.9 dB end-to-end SNR). BigVGAN-v2 is stored in float16 except its snake parameters
(59 dB, 225 MB instead of 450 MB); the small vocoders stay float32 (about 55 MB each; float16 storage gave Vocos only
51 dB). ``--fp32`` keeps every vocoder in float32. The loaders cast everything to float32.

    python -m drifting_tts.mlx.convert --model drifting_tts_v3.1.pt --vocoder bigvgan-v2-ft --out mlx
    python -m drifting_tts.mlx.convert --vocoder bigvgan-base-ft --vocoder vocos-ft --out mlx

A vocoder is a registry name (``drifting_tts.vocoder.load_vocoder`` finds the checkpoint locally or on the Hub),
``NAME=PATH`` for a local checkpoint of one of them, or a bare BigVGAN-v2 checkpoint path (``bigvgan-v2-ft``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..text import SYMBOLS
from ..voices import DEFAULT_VOICE, VOICES
from .vocoder import VOCODERS, save_vocoder

FP16_VOCODERS = {"bigvgan-v2-ft"}  # stored in float16 unless --fp32

# the BigVGAN hyper-parameters that define the generator (the rest of its config.json is about training)
VOCODER_KEYS = ("resblock", "upsample_rates", "upsample_kernel_sizes", "upsample_initial_channel",
                "resblock_kernel_sizes", "resblock_dilation_sizes", "use_tanh_at_final", "use_bias_at_final",
                "activation", "snake_logscale", "num_mels", "hop_size", "sampling_rate")


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


def convert_vocoder(voc) -> tuple[str, dict, dict[str, np.ndarray]]:
    """A PyTorch :class:`drifting_tts.vocoder.Vocoder` -> ``(kind, hparams, weights)`` for :func:`save_vocoder`."""
    if voc.kind == "bigvgan":
        from .bigvgan import convert_bigvgan

        return "bigvgan", {k: v for k, v in dict(voc.model.h).items() if k in VOCODER_KEYS}, \
            convert_bigvgan(voc.model.state_dict())
    if voc.kind == "vocos" and voc.mel == "bigvgan":
        from .vocos import convert_vocos, vocos_hparams

        return "vocos", vocos_hparams(voc.model), convert_vocos(voc.model.state_dict())
    raise ValueError(f"vocoder {voc.name!r} ({voc.kind} on {voc.mel} mels) has no MLX port")


def parse_vocoder(spec: str) -> tuple[str, str]:
    """``NAME``, ``NAME=PATH`` or a BigVGAN-v2 checkpoint path -> ``(registry name, load_vocoder spec)``."""
    name, _, path = spec.partition("=")
    if not path and name not in VOCODERS:
        if not Path(name).is_file():
            raise ValueError(f"unknown vocoder {spec!r}: expected NAME, NAME=PATH or a checkpoint path, with NAME "
                             f"one of {', '.join(VOCODERS)}")
        return "bigvgan-v2-ft", name
    if name not in VOCODERS:
        raise ValueError(f"no MLX vocoder file for {name!r} (one of {', '.join(VOCODERS)})")
    return name, path or name


def main() -> None:
    import torch

    from ..vocoder import load_vocoder

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", help="acoustic checkpoint (drifting_tts_v3.1.pt): model.safetensors and config.json")
    p.add_argument("--vocoder", action="append", default=None,
                   help=f"repeatable: {' | '.join(VOCODERS)}, NAME=PATH or a BigVGAN-v2 checkpoint path "
                        "(default with --model: bigvgan-v2-ft)")
    p.add_argument("--out", default="mlx")
    p.add_argument("--fp32", action="store_true", help="store every vocoder in float32")
    args = p.parse_args()
    vocoders = [parse_vocoder(v) for v in (args.vocoder or (["bigvgan-v2-ft"] if args.model else []))]
    if not args.model and not vocoders:
        p.error("nothing to convert: pass --model and/or --vocoder")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    default_hparams = None
    for name, spec in vocoders:
        kind, hparams, weights = convert_vocoder(load_vocoder(spec, "cpu"))
        save_vocoder(out / VOCODERS[name], weights, kind, hparams, fp16=name in FP16_VOCODERS and not args.fp32)
        if name == "bigvgan-v2-ft":
            default_hparams = hparams
        print(f"{name}: {out / VOCODERS[name]}")

    if args.model:
        import mlx.core as mx

        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        weights = {k: mx.array(v, dtype=mx.float32) for k, v in convert_acoustic(ck["ema"]).items()}
        mx.save_safetensors(str(out / "model.safetensors"), weights)
        config = {
            "model": ck["config"]["model"], "num_speakers": ck["num_speakers"], "n_vocab": len(SYMBOLS),
            "n_mels": 100, "sample_rate": 24000, "hop_length": 256,
            "stats": {k: ck["stats"][k] for k in ("mean", "std")},
            "duration_scale": float(ck.get("duration_scale", 1.0)),
            "duration_scales": {str(k): float(v) for k, v in ck.get("duration_scales", {}).items()},
            "temperature": float(ck.get("temperature") or 0.3),
            "voices": {k: v["id"] for k, v in VOICES.items()}, "default_voice": DEFAULT_VOICE,
        }
        if default_hparams is not None:  # older MLX releases read the default vocoder's architecture from here
            config["vocoder"] = default_hparams
        (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"wrote {out}: " + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.0f} MB" for f in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
