#!/usr/bin/env bash
# Deploy the private test Space of the pilots (space_pilots/): ARDrift v2 (#103) and the voice-reference model with
# guidance (#96, #99). The app and the package are copied, and the two checkpoints go into the Space repo itself in
# fp16 (0.6 GB; a Space stores at most 1 GB), so the Space needs no token. The vocoder and the voices come from the
# released model's repo.
#   scripts/deploy_pilots_space.sh <space id> <ARDrift v2 checkpoint> <voice-reference checkpoint>
# Without the two checkpoints only the app is deployed: it then downloads the weights from the pilots' model repo
# (DRIFTING_TTS_PILOTS_REPO; a private repo needs the Space secret HF_TOKEN).
# The Space is created private if it does not exist yet; the script refuses a Space that is not private. Its
# hardware is set separately (a CPU Space speaks a sentence in about half a minute, ZeroGPU in a second or two).
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1 V2=${2:-} VOICE=${3:-}
python -c "import sys; from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$SPACE', repo_type='space', space_sdk='gradio', private=True, exist_ok=True)
sys.exit(0 if api.space_info('$SPACE').private else '$SPACE is not private: refusing to deploy')"
STAGE=$(mktemp -d)
cp space_pilots/app.py space_pilots/engine.py space_pilots/requirements.txt space_pilots/README.md "$STAGE/"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
if [ -n "$V2" ]; then
  python -c "import sys, torch; from drifting_tts.publish import export_tts
for src, dst in zip(sys.argv[1:3], sys.argv[3:5]):
    torch.save(export_tts(torch.load(src, map_location='cpu', weights_only=False), half=True), dst)" \
    "$V2" "$VOICE" "$STAGE/ardrift_v2_clock_40k.pt" "$STAGE/ardrift_en_voice.pt"
fi
du -sh "$STAGE"
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
