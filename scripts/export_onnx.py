"""Export the model to ONNX for the browser (onnxruntime-web, WebGPU) and check every graph against PyTorch.

Three graphs; the glue between them is a few lines of JavaScript (web/tts.js):

* ``text_encoder.onnx``: token ids ``[1, N]``, speaker ``[1]`` -> token condition ``[1, 100 + d, N]`` (prior mean and
  pitch-conditioned text features) and log-durations ``[1, 1, N]``;
* ``generator.onnx``: noise ``[1, 100, T]``, frame condition ``[1, 100 + d, T]``, speaker, guidance scale ``[1]``
  and style codes ``[1, noise_coords]`` -> log-mel ``[1, 100, T]`` (one DriftDiT pass, de-normalised);
* ``vocoder.onnx``: log-mel ``[1, 100, T]`` -> waveform ``[1, T * 256]`` at 24 kHz (BigVGAN-v2, weight norm removed).

Between the encoder and the generator, each token's condition column is repeated ``ceil(exp(logw) * scale)`` times
(the hard alignment of ``durations_to_alignment``). The graphs are simplified with onnxslim when it is installed.
``--fp16`` also writes ``*_fp16.onnx`` copies that store the weights in half precision (half the download) and cast
them back to fp32 when the session is created, so they compute exactly like the fp32 graphs on any WebGPU device.

    python scripts/export_onnx.py --model drifting_tts_v3.1.pt --vocoder bigvgan_v2_ft.pt --out web/models
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from drifting_tts.text import SYMBOLS
from drifting_tts.train import load_tts
from drifting_tts.vocoder import load_bigvgan
from drifting_tts.voices import DEFAULT_VOICE, VOICES


class TextEncoder(nn.Module):
    def __init__(self, tts):
        super().__init__()
        self.tts = tts

    def forward(self, text: torch.Tensor, speaker: torch.Tensor):
        lengths = torch.ones_like(speaker) * text.shape[1]
        h, mu, logw, x_mask = self.tts.encoder(text, lengths, speaker)
        if self.tts.pitch_enabled:
            h, _ = self.tts.pitch_condition(h, x_mask, speaker)
        return torch.cat([mu, h], 1), logw


class Generator(nn.Module):
    def __init__(self, tts, mean: float, std: float):
        super().__init__()
        self.tts, self.mean, self.std = tts, mean, std

    def forward(self, z, cond, speaker, cfg_scale, noise_labels):
        mel = self.tts.generate(z, cond, speaker, cfg_scale, noise_labels=noise_labels)
        return mel * self.std + self.mean


class Vocoder(nn.Module):
    def __init__(self, bigvgan):
        super().__init__()
        self.bigvgan = bigvgan
        static_channel_filters(bigvgan)

    def forward(self, mel):
        return self.bigvgan(mel)[:, 0].clamp(-1, 1)


def static_channel_filters(bigvgan: nn.Module) -> None:
    """BigVGAN's anti-aliased resampling expands its filter to ``x.shape[1]`` channels, which the ONNX tracer keeps
    symbolic ("convolution for kernel of unknown shape"). The channel count is fixed, so freeze it at trace time."""
    import torch.nn.functional as F

    def up(self, x):
        c = int(x.shape[1])
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c)
        return x[..., self.pad_left: -self.pad_right]

    def lowpass(self, x):
        c = int(x.shape[1])
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c)

    for m in bigvgan.modules():
        name = type(m).__name__
        if name == "UpSample1d":
            m.forward = up.__get__(m)
        elif name == "LowPassFilter1d":
            m.forward = lowpass.__get__(m)


def export(module: nn.Module, args: tuple, path: Path, names: list[str], outputs: list[str], dynamic: dict) -> None:
    torch.onnx.export(module, args, str(path), input_names=names, output_names=outputs, dynamic_axes=dynamic,
                      opset_version=18, dynamo=False)


def simplify(path: Path) -> None:
    try:
        import onnxslim
    except ImportError:
        return
    import onnx

    onnx.save(onnxslim.slim(onnx.load(str(path))), str(path))


def half_weights(src: Path, dst: Path, min_size: int = 1024) -> None:
    """Store large fp32 initializers as fp16, each followed by a Cast back to fp32 (folded at session creation)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    keep, casts = [], []
    for init in m.graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < min_size:
            keep.append(init)
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def check(path: Path, module: nn.Module, feeds: dict, name: str) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})
    with torch.no_grad():
        ref = module(*feeds.values())
    ref = ref if isinstance(ref, tuple) else (ref,)
    err = max(float(np.abs(g - r.numpy()).max() / (np.abs(r.numpy()).max() + 1e-9)) for g, r in zip(got, ref))
    print(f"{name}: max relative error vs PyTorch {err:.2e}", flush=True)
    return err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", required=True, help="fine-tuned BigVGAN-v2 (bigvgan_ft.pt)")
    p.add_argument("--out", default="web/models")
    p.add_argument("--fp16", action="store_true", help="also write *_fp16.onnx copies with half-precision weights")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tts, cfg, stats = load_tts(args.model, "cpu")
    gen = tts.generator
    enc, generator = TextEncoder(tts).eval(), Generator(tts, stats["mean"], stats["std"]).eval()
    vocoder = Vocoder(load_bigvgan(device="cpu", finetuned=args.vocoder)).eval()

    n, t = 23, 61  # odd lengths: the generator pads T to its patch size
    text = torch.randint(2, len(SYMBOLS), (1, n))
    spk = torch.tensor([VOICES[DEFAULT_VOICE]["id"]])
    cond, logw = enc(text, spk)
    gen_args = (torch.randn(1, 100, t), torch.randn(1, cond.shape[1], t), spk, torch.tensor([2.0]),
                torch.randint(0, gen.noise_classes, (1, max(1, gen.noise_coords))))
    mel = torch.randn(1, 100, t) * 2 - 5

    export(enc, (text, spk), out / "text_encoder.onnx", ["text", "speaker"], ["cond", "logw"],
           {"text": {1: "N"}, "cond": {2: "N"}, "logw": {2: "N"}})
    export(generator, gen_args, out / "generator.onnx", ["z", "cond", "speaker", "cfg_scale", "noise_labels"], ["mel"],
           {"z": {2: "T"}, "cond": {2: "T"}, "mel": {2: "T"}})
    export(vocoder, (mel,), out / "vocoder.onnx", ["mel"], ["audio"], {"mel": {2: "T"}, "audio": {1: "S"}})

    for name in ("text_encoder", "generator", "vocoder"):
        simplify(out / f"{name}.onnx")

    # parity at other lengths than the export example (dynamic axes really dynamic)
    text2 = torch.randint(2, len(SYMBOLS), (1, 57))
    check(out / "text_encoder.onnx", enc, {"text": text2, "speaker": spk}, "text_encoder")
    cond2, _ = enc(text2, spk)
    for t2 in (150, 151):  # even and odd: with and without patch padding
        feeds = {"z": torch.randn(1, 100, t2), "cond": torch.randn(1, cond2.shape[1], t2), "speaker": spk,
                 "cfg_scale": torch.tensor([2.0]), "noise_labels": gen_args[4]}
        check(out / "generator.onnx", generator, feeds, f"generator T={t2}")
    check(out / "vocoder.onnx", vocoder, {"mel": torch.randn(1, 100, 97) * 2 - 5}, "vocoder")

    if args.fp16:
        for name in ("text_encoder", "generator", "vocoder"):
            half_weights(out / f"{name}.onnx", out / f"{name}_fp16.onnx")

    meta = {"sample_rate": 24000, "hop_length": 256, "n_mels": 100, "cond_channels": int(cond.shape[1]),
            "noise_classes": gen.noise_classes, "noise_coords": max(1, gen.noise_coords),
            "duration_scale": tts.duration_scale,
            "duration_scales": {str(k): v for k, v in tts.duration_scales.items()},
            "temperature": float(getattr(tts, "temperature", None) or 0.3),
            "voices": {k: v["id"] for k, v in VOICES.items()}, "default_voice": DEFAULT_VOICE,
            "files": {f.name: f.stat().st_size for f in sorted(out.glob("*.onnx"))}}
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: " + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.0f} MB" for f in sorted(out.glob("*.onnx"))))


if __name__ == "__main__":
    main()
