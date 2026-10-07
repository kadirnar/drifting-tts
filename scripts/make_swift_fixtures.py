"""Generate small, deterministic CPU fixtures for native Swift/Python MLX parity.

Run with the repository's MLX environment; no model download or GPU work is required.
The predictors have fixed 256-channel widths, so exact float32 weights require about 2 MB.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from drifting_tts.mlx.bigvgan import BigVGAN
from drifting_tts.mlx.model import DriftingTTS
from drifting_tts.mlx.synthesize import Synthesizer
from drifting_tts.text import SYMBOLS, text_to_ids

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "swift/DriftingTTS/Tests/DriftingTTSTests/Fixtures"
MODEL = {
    "text": {"d": 16, "heads": 2, "layers": 2, "ffn": 32, "spk_dim": 8, "dropout": 0.0},
    "gen": {"hidden": 32, "depth": 2, "heads": 2, "patch": 2, "mlp_ratio": 2.0,
            "n_registers": 4, "noise_classes": 8, "noise_coords": 3, "residual_prior": True, "num_steps": 1},
    "pitch": {"enabled": True},
}
VOCODER = {
    "num_mels": 100, "upsample_rates": [4, 2], "upsample_kernel_sizes": [8, 4],
    "upsample_initial_channel": 16, "resblock": "1", "resblock_kernel_sizes": [3, 5],
    "resblock_dilation_sizes": [[1, 3], [1, 5]], "activation": "snakebeta", "snake_logscale": True,
    "use_tanh_at_final": True, "use_bias_at_final": True,
}


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    mx.set_default_device(mx.cpu)
    mx.random.seed(123)
    model = DriftingTTS(MODEL, num_speakers=3, n_vocab=len(SYMBOLS), n_mels=100)
    # Stay away from ceil boundaries: every token takes exactly two frames at scale 1.
    model.encoder.duration.proj.weight = mx.zeros_like(model.encoder.duration.proj.weight)
    model.encoder.duration.proj.bias = mx.array([np.log(1.25)], dtype=mx.float32)
    model.generator.registers = mx.random.normal(model.generator.registers.shape) * 0.1
    vocoder = BigVGAN(VOCODER)
    weights = dict(tree_flatten(vocoder.parameters()))
    # Nonzero, distinct coefficients exercise SnakeBeta exp() and reciprocal conversion.
    for name, value in weights.items():
        if name.endswith((".alpha", ".beta")):
            weights[name] = mx.random.uniform(-0.3, 0.3, value.shape)
    vocoder.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters(), vocoder.parameters())
    mx.save_safetensors(str(destination / "model.safetensors"), dict(tree_flatten(model.parameters())))
    mx.save_safetensors(str(destination / "vocoder.safetensors"), dict(tree_flatten(vocoder.parameters())))
    config = {
        "model": MODEL, "vocoder": VOCODER, "num_speakers": 3, "n_vocab": len(SYMBOLS), "n_mels": 100,
        "sample_rate": 24000, "hop_length": 8, "stats": {"mean": -2.0, "std": 1.25},
        "duration_scale": 1.0, "duration_scales": {}, "temperature": 0.3,
        "voices": {"studio": 0, "female": 1, "male": 2}, "default_voice": "studio",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tensors, encoders, generators, vocoders = {}, [], [], []
    for name, text, speaker, pitch in [
        ("short", "Merhaba.", 0, 0.0), ("pitched", "İyi günler!", 1, 2.0),
        ("lowered", "Merhaba dünya.", 2, -3.0),
    ]:
        prefix = f"encode.{name}"
        ids, spk = mx.array([text_to_ids(text)]), mx.array([speaker])
        tokens, logw = model.encode(ids, spk, pitch)
        tensors.update({f"{prefix}.ids": ids, f"{prefix}.speaker": spk,
                        f"{prefix}.tokens": tokens, f"{prefix}.log_durations": logw})
        encoders.append({"name": name, "pitch_shift": pitch})
    for frames in (17, 22):
        prefix = f"generate.frames{frames}"
        noise = mx.random.normal((1, frames, 100)) * 0.3
        condition = mx.random.normal((1, frames, 116))
        speaker, cfg, labels = mx.array([2]), mx.array([1.7]), mx.array([[1, 4, 7]])
        output = model.generate(noise, condition, speaker, cfg, labels)
        tensors.update({f"{prefix}.{key}": value for key, value in {
            "noise": noise, "condition": condition, "speaker": speaker, "cfg": cfg, "labels": labels,
            "output": output,
        }.items()})
        generators.append({"name": f"frames{frames}"})
    for frames in (1, 7, 79):
        prefix = f"vocoder.frames{frames}"
        mel = mx.random.normal((1, frames, 100)) * 1.5 - 2
        output = vocoder(mel.transpose(0, 2, 1))
        tensors.update({f"{prefix}.mel": mel, f"{prefix}.output": output})
        vocoders.append({"name": f"frames{frames}"})
    # A genuine interior chunk checks both context margins, independently of utterance boundaries.
    context = vocoder.context_frames
    start, end = context + 3, context + 10
    count = end + context + 4
    mel = mx.random.normal((1, count, 100)) * 1.5 - 2
    full = vocoder(mel.transpose(0, 2, 1))
    cropped = vocoder(mel[:, start - context:end + context].transpose(0, 2, 1))
    cropped = cropped[:, context * 8:(context + end - start) * 8]
    np.testing.assert_allclose(np.array(cropped), np.array(full[:, start * 8:end * 8]), atol=2e-6, rtol=2e-5)
    tensors.update({"chunk.mel": mel, "chunk.output": full[:, start * 8:end * 8]})
    pipeline = {"text": "Merhaba. İyi günler!", "speaker": "female", "seed": 19,
                "cfg_scale": 1.5, "temperature": 0.4, "length_scale": 1.0, "pause": 0.002,
                "chunk_frames": 11, "first_chunk_frames": 3}
    synth = Synthesizer(destination, compile=False)
    chunks = list(synth.stream(**pipeline))
    waveform = np.concatenate([audio for audio, _ in chunks])
    tensors["pipeline.waveform"] = mx.array(waveform)
    pipeline["sample_count"] = len(waveform)
    pipeline["silence_chunks"] = sum(info["is_silence"] for _, info in chunks)
    # Record Threefry draws separately to localize a seed/key-splitting mismatch.
    key, noise_key, label_key = mx.random.split(mx.random.key(pipeline["seed"]), 3)
    tensors["random.next_key"] = key
    tensors["random.noise"] = mx.random.normal((1, 17, 100), key=noise_key)
    tensors["random.labels"] = mx.random.randint(0, 8, (1, 3), key=label_key)
    mx.eval(tensors)
    mx.save_safetensors(str(destination / "references.safetensors"), tensors)
    metadata = {"format_version": 1, "python_mlx_version": importlib.metadata.version("mlx"),
                "encode_cases": encoders, "generate_cases": generators, "vocoder_cases": vocoders,
                "context_frames": context, "chunk_start": start, "chunk_end": end, "pipeline": pipeline}
    (destination / "cases.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    size = sum(path.stat().st_size for path in destination.glob("*.safetensors"))
    print(f"Wrote {len(tensors)} reference tensors and deterministic float32 model/vocoder weights: {size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    generate(parser.parse_args().output)
