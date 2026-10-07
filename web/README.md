---
title: Drifting TTS WebGPU
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: One-step Turkish text-to-speech in the browser with WebGPU
models:
- Vyvo/drifting-tts-tr
---

# Drifting TTS in the browser (WebGPU)

The Turkish [drifting-tts](https://github.com/kadirnar/drifting-tts) model running fully client-side with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) on WebGPU. Without WebGPU it falls back to WASM on the CPU, which is much slower. No server is involved: the text never leaves the device.

| file | contents |
|---|---|
| `text.js` | the Turkish text frontend (numbers, dates, units, abbreviations), a line-by-line port of `drifting_tts/text.py` |
| `tts.js` | the pipeline: text encoder → duration expansion → one DriftDiT pass → BigVGAN-v2 |
| `app.js`, `index.html` | the demo page |

The models are the `onnx/*_fp16.onnx` graphs of [Vyvo/drifting-tts-tr](https://huggingface.co/Vyvo/drifting-tts-tr).
Together they are about 370 MB. The browser caches them after the first visit. They store their weights in fp16 and compute in fp32.

## Run locally

```bash
python scripts/export_onnx.py --model drifting_tts_v3.1.pt --vocoder bigvgan_v2_ft.pt --out web/models --fp16
cd web && python -m http.server 8000
# open http://localhost:8000/?models=http://localhost:8000/models
```

## Tests

- **Text frontend:** `node web/tests/text.test.mjs` checks `text.js` against fixtures written by
  `scripts/make_web_fixtures.py` from the Python frontend.
- **Graphs:** `scripts/export_onnx.py` checks every exported graph against PyTorch.
