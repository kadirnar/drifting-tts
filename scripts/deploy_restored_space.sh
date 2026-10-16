#!/usr/bin/env bash
# Deploy the private Space of the restored voices (space_restored/, #71): the voices model fine-tuned on the recordings
# restored with Sidon next to the one trained on the original recordings (with its mel bandwidth extension). The app is
# space_test/voices_app.py, staged as app.py. The test Space has no room for two more models: a Space stores at most
# 1 GB, every stored file version included.
#   scripts/deploy_restored_space.sh <space id> <voices checkpoint> <restored voices checkpoint> <MelBWE checkpoint> \
#       <voices.json>
# Both checkpoints go into the Space repo itself in fp16, with the duration predictor they carry. The Space is created
# private if it does not exist yet (its hardware is set separately); the script refuses a Space that is not private.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1 MODEL=$2 RESTORED=$3 BWE=$4 VOICES=$5
hf repo create "$SPACE" --repo-type space --space-sdk gradio --private --exist-ok
python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().space_info('$SPACE').private else
'$SPACE is not private: refusing to deploy')"
STAGE=$(mktemp -d)
cp space_test/requirements.txt space_restored/README.md "$STAGE/"
cp space_test/voices_app.py "$STAGE/app.py"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
for pair in "$MODEL:ardrift_voices.pt" "$RESTORED:ardrift_voices_restored.pt"; do
  python -c "import sys, torch; from drifting_tts.publish import export_tts
torch.save(export_tts(torch.load(sys.argv[1], map_location='cpu', weights_only=False), half=True), sys.argv[2])" \
    "${pair%%:*}" "$STAGE/${pair#*:}"
done
cp "$BWE" "$STAGE/bwe.pt"
cp "$VOICES" "$STAGE/voices.json"
du -sh "$STAGE"
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
