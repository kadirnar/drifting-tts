#!/usr/bin/env bash
# End-to-end improved recipe (docs/TRAINING.md). Usage: scripts/train_v2.sh [tts config] [extra overrides...]
#   scripts/train_v2.sh configs/tts_v2.yaml
#   scripts/train_v2.sh configs/tts_v2_large_gpu.yaml train.num_workers=8
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/tts_v2.yaml}
shift || true
RUN=runs/$(basename "$CFG" .yaml)

# 1. data: mels + F0 (pitch conditioning) + waveforms (vocoder fine-tuning, SIM against recordings),
#    with a disjoint dev split for tuning (val is kept for the reported numbers)
[ -f data/train/f0.bin ] && [ -f data/train/audio.bin ] || \
    drifting-tts prepare --out data/train --f0 --save-audio --dev-size 200

# 2. 2-D Mel-MAE feature encoder for the drift kernel
[ -f runs/mae2d/mae_ema.pt ] || drifting-tts train-mae --config configs/mae2d.yaml --workdir runs/mae2d

# 3. drifting TTS
drifting-tts train --config "$CFG" --workdir "$RUN" "$@"

# 4. duration calibration, vocoder fine-tuning on the model's own mels
drifting-tts calibrate-durations --model "$RUN/model_ema.pt"
drifting-tts finetune-vocoder --workdir "$RUN/vocoder" tts.path="$RUN/model_ema.pt"

# 5. tuning sweeps on the dev split (if the data has one): harmonic contrast (fast), then the full judges.
#    Store the chosen temperature with: drifting-tts calibrate-durations --model $RUN/model_ema.pt --temperature T
if grep -q '"split": "dev"' data/train/index.jsonl; then
    drifting-tts evaluate --model "$RUN/model_ema.pt" --harmonic --split dev --num 200 \
        --temperature 0.3 0.5 0.7 1.0 --out "$RUN/eval_dev_harmonic"
    drifting-tts evaluate --model "$RUN/model_ema.pt" --split dev --num 100 --temperature 0.3 0.5 0.7 1.0 \
        --cfg 1.0 1.5 --out "$RUN/eval_dev"
fi

# 6. reported numbers on val (temperature: the stored one, else 0.5), stock vs fine-tuned vocoder
drifting-tts evaluate --model "$RUN/model_ema.pt" --num 200 --out "$RUN/eval"
drifting-tts evaluate --model "$RUN/model_ema.pt" --vocoder "$RUN/vocoder/vocos_ft.pt" --num 200 --out "$RUN/eval_ftvoc"
