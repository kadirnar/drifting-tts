#!/usr/bin/env bash
# Deploy the Gradio demo: space/ + the drifting_tts package -> a Hugging Face Space (weights come from the model repo).
#   scripts/deploy_space.sh [space id]        default: Vyvo/drifting-tts-tr-demo
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=${1:-Vyvo/drifting-tts-tr-demo}
STAGE=$(mktemp -d)
cp space/app.py space/requirements.txt space/README.md "$STAGE/"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
