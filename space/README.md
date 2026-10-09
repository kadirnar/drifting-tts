---
title: Drifting TTS Turkish
emoji: 🗣️
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: One-step Turkish TTS trained with a drifting objective
models:
- Vyvo/drifting-tts-tr
---

One-step Turkish text to speech: one DriftDiT pass per sentence. Release v3.2 (default): durations and pitch sampled
by a prosody model trained with drifting, Vocos v2, punctuation-aware pauses; v3.1 (BigVGAN-v2) for comparison.
Model: [Vyvo/drifting-tts-tr](https://huggingface.co/Vyvo/drifting-tts-tr) · Code: [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts)
