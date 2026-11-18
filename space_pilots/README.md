---
title: Drifting TTS English pilots (private test)
emoji: 🧪
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: ARDrift v2 and voice cloning pilots (private test)
---

Private test of two research pilots of [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts): ARDrift v2
(the text as a prefix, no aligner) and the aligned English model with a voice from a recording. Not a release: the
released model's demo is [Vyvo/drifting-tts-en-demo](https://huggingface.co/spaces/Vyvo/drifting-tts-en-demo).

`scripts/deploy_pilots_space.sh` puts the weights into the Space (fp16). Without them the app downloads them from
the pilots' model repo, and a private repo then needs the secret `HF_TOKEN` (a token that can read it). On a CPU
Space set the variable `DEMO_THREADS` to the number of vCPUs (default 2).
