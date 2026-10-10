#!/usr/bin/env bash
# Final evaluation of a release against the v3.1 reference rows (docs/RESULTS.md, "v3.2").
#
#   scripts/eval_release.sh MODEL PROSODY VOCODER OUT
#
#   MODEL    acoustic checkpoint of the release (v3.2: runs/release/drifting_tts_v3.1.pt, or a fine-tuned one)
#   PROSODY  prosody predictor: a train-prosody checkpoint or `drift` (the published one)
#   VOCODER  vocoder: a checkpoint (vocos_ft_<step>.pt) or a registry name (`vocos-v2`)
#   OUT      output directory; every run is skipped when its results.json exists (resumable)
#
# Systems: `v3.1-bigvgan` (v3.1 as released: BigVGAN-v2-ft, regressors, 0.15 s pauses), `v3.1-vocos` (vocos-ft) and
# `v3.2` (MODEL + PROSODY + VOCODER + punctuation pauses). Protocols (docs/EXPERIMENTS.md, "Protocols and judges"):
#   freya495  all 495 Freya-TR-Eval sentences for the studio, male and female voices (the README table)
#   freya100  the first 100 sentences, speaker 722 (reproduces known rows: v3.1-bigvgan 0.66% / 2.934,
#             v3.1-vocos 1.10% / 2.627 in WER / UTMOSv2)
#   prosody   `drifting-tts prosody` on the 100 studio `val` utterances: recording, predicted (v3.1 + vocos-ft),
#             predicted@voc=bigvgan-v2-ft (v3.1 as released) and release (v3.2)
# Then scripts/release_summary.py writes OUT/summary.md (with DNSMOS OVRL from the saved audio).
#
# Environment: STAGES (default "freya100 freya495 prosody"; the dry run: STAGES=freya100), REF_MODEL (default
# runs/release/drifting_tts_v3.1.pt), DATA (prepared data for the prosody stage; default: the model's data.root),
# NEED_GB (GPU memory to wait for before each run, default 7; the judges need ~5-6 GB), MAX_GB (the shared-GPU
# budget, default 28), DEVICE (default cuda), PY (default python). Run from the directory the relative paths refer to;
# the drifting_tts package of this repository is put first on PYTHONPATH.
set -euo pipefail
if [ $# -ne 4 ]; then sed -n 2,27p "$0"; exit 1; fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
MODEL=$1 PROSODY=$2 VOCODER=$3 OUT=$4
STAGES=${STAGES:-freya100 freya495 prosody}
PROSODY_DURATIONS=${PROSODY_DURATIONS:-regressor}  # v3.2 samples the token pitch only (hub.RELEASES)
REF_MODEL=${REF_MODEL:-runs/release/drifting_tts_v3.1.pt}
NEED_GB=${NEED_GB:-7} MAX_GB=${MAX_GB:-28} DEVICE=${DEVICE:-cuda} PY=${PY:-python}
mkdir -p "$OUT"
LOG="$OUT/eval_release.log"
echo "$(date -u +%FT%TZ) eval_release: model=$MODEL prosody=$PROSODY vocoder=$VOCODER stages=[$STAGES]" | tee -a "$LOG"

wait_gpu() {  # a GPU shared with other jobs: start only when used + NEED_GB <= MAX_GB
  [ "$DEVICE" = cuda ] || return 0
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ $((used + NEED_GB * 1024)) -le $((MAX_GB * 1024)) ] && return 0
    echo "$(date -u +%T) waiting for GPU memory (used ${used} MiB, need ${NEED_GB} GB)" >> "$LOG"
    sleep 60
  done
}

bench() {  # bench <name> <args...>: one `drifting-tts benchmark` run into $OUT/<name>
  local name=$1; shift
  if [ -f "$OUT/$name/results.json" ]; then echo "skip $name (done)" | tee -a "$LOG"; return 0; fi
  wait_gpu
  echo "$(date -u +%T) run $name" | tee -a "$LOG"
  $PY -m drifting_tts.cli benchmark --save-wavs 1000 --device "$DEVICE" --out "$OUT/$name" "$@" >> "$LOG" 2>&1
}

systems() {  # systems <protocol> <voice> <extra benchmark args...>
  local proto=$1 voice=$2; shift 2
  bench "${proto}_${voice}_v3.1-bigvgan" --model "$REF_MODEL" --vocoder bigvgan-v2-ft --speaker "$voice" "$@"
  bench "${proto}_${voice}_v3.1-vocos" --model "$REF_MODEL" --vocoder vocos-ft --speaker "$voice" "$@"
  bench "${proto}_${voice}_v3.2" --model "$MODEL" --vocoder "$VOCODER" --prosody "$PROSODY" --pause punct \
    --prosody-durations "$PROSODY_DURATIONS" --speaker "$voice" "$@"
}

for stage in $STAGES; do
  case $stage in
    freya100) systems freya100 studio --num 100 ;;
    freya495) for voice in studio male female; do systems freya495 "$voice"; done ;;
    prosody)
      if [ -f "$OUT/prosody_val722/results.json" ]; then echo "skip prosody (done)" | tee -a "$LOG"; continue; fi
      wait_gpu
      echo "$(date -u +%T) run prosody_val722" | tee -a "$LOG"
      $PY -m drifting_tts.cli prosody --model "$REF_MODEL" ${DATA:+--data "$DATA"} --split val --speaker 722 \
        --num 100 --vocoder vocos-ft --systems recording predicted predicted@voc=bigvgan-v2-ft release \
        --release-model "$MODEL" --release-prosody "$PROSODY" --release-vocoder "$VOCODER" --workers 2 \
        --release-prosody-durations "$PROSODY_DURATIONS" \
        --device "$DEVICE" --out "$OUT/prosody_val722" >> "$LOG" 2>&1 ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
wait_gpu
$PY "$REPO/scripts/release_summary.py" "$OUT" --device "$DEVICE" | tee -a "$LOG"
