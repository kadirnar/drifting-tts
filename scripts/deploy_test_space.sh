#!/usr/bin/env bash
# Deploy the private test Space (space_test/): one tab per model under test, with their files:
#   scripts/deploy_test_space.sh <space id> <ARDrift checkpoint> <ARDrift voices checkpoint> <MelBWE checkpoint> <voices.json> \
#       [<English ARDrift checkpoint> <english_voices.json>]
# DURATIONS=<train-durations checkpoint of the voices model> and ARDRIFT_DURATIONS=<... of the ARDrift checkpoint> add
# only their duration predictors (durations_tr.pt, durations_ardrift.pt; 2 MB each), offered next to the models' own.
# RESTORED=<voices checkpoint fine-tuned on the restored recordings (#71)> adds it to the voices tab, next to the voices
# model (as ardrift_voices_restored.pt).
# The ARDrift tab is the public demo's app (space/app.py, staged as demo_app.py) with the ARDrift checkpoint beside it.
# The checkpoints go into the Space repo itself, the voices models in fp16 (a Space stores at most 1 GB, every stored
# file version included; the English checkpoint is copied as given, exported in fp16 already). The script refuses a
# Space that is not private.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1 ARDRIFT=$2 MODEL=$3 BWE=$4 VOICES=$5 ENGLISH=${6:-} ENGLISH_VOICES=${7:-}
python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().space_info('$SPACE').private else
'$SPACE is not private: refusing to deploy')"
if grep -n '{{' space/app.py; then  # result placeholders of a release not filled in yet
  echo "fill in the {{...}} placeholders above before deploying" >&2; exit 1
fi
STAGE=$(mktemp -d)
cp space_test/app.py space_test/voices_app.py space_test/english_app.py space_test/requirements.txt \
  space_test/README.md "$STAGE/"
cp space/app.py "$STAGE/demo_app.py"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$ARDRIFT" "$STAGE/ardrift.pt"
cp "$BWE" "$STAGE/bwe.pt"
cp "$VOICES" "$STAGE/voices.json"
half() {  # <checkpoint> <staged name>: exported in fp16
  python -c "import sys, torch; from drifting_tts.publish import export_tts
torch.save(export_tts(torch.load(sys.argv[1], map_location='cpu', weights_only=False), half=True), sys.argv[2])" \
    "$1" "$STAGE/$2"
}
half "$MODEL" ardrift_voices.pt
if [ -n "$ENGLISH" ]; then cp "$ENGLISH" "$STAGE/ardrift_en.pt"; cp "$ENGLISH_VOICES" "$STAGE/english_voices.json"; fi
if [ -n "${RESTORED:-}" ]; then half "$RESTORED" ardrift_voices_restored.pt; fi
if [ -n "${DURATIONS:-}" ]; then
  python -c "import sys, torch; from drifting_tts.publish import export_durations
torch.save(export_durations(torch.load(sys.argv[1], map_location='cpu', weights_only=False)), sys.argv[2])" \
    "$DURATIONS" "$STAGE/durations_tr.pt"
fi
if [ -n "${ARDRIFT_DURATIONS:-}" ]; then
  python -c "import sys, torch; from drifting_tts.publish import export_durations
torch.save(export_durations(torch.load(sys.argv[1], map_location='cpu', weights_only=False)), sys.argv[2])" \
    "$ARDRIFT_DURATIONS" "$STAGE/durations_ardrift.pt"
fi
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
