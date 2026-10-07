#!/usr/bin/env bash
# Deploy the WebGPU demo: web/ -> a static Hugging Face Space (the ONNX graphs come from the model repo).
#   scripts/deploy_webgpu_space.sh [space id]        default: Vyvo/drifting-tts-tr-webgpu
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=${1:-Vyvo/drifting-tts-tr-webgpu}
STAGE=$(mktemp -d)
cp web/index.html web/app.js web/tts.js web/text.js web/README.md "$STAGE/"
hf repo create "$SPACE" --repo-type space --space-sdk static --exist-ok >/dev/null
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
