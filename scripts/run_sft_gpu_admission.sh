#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-$ROOT/../runs/sft-gpu-admission-single-persona-v1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/sft-gpu-admission-single-persona-v1.log"
if [[ -e "$OUT_DIR" ]]; then
  echo "SFT_GPU_ADMISSION: FAIL"
  echo "FAILURE_PHASE: preflight"
  echo "FAILURE_REASON: OUTPUT_DIR_ALREADY_EXISTS_REFUSING_OVERWRITE"
  exit 2
fi
cd "$ROOT"
set +e
"$PYTHON_BIN" -u scripts/run_sft_gpu_admission.py --model-path "$MODEL_PATH" --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
if [[ -d "$OUT_DIR" ]]; then
  cp "$LOG" "$OUT_DIR/sft_gpu_admission.log"
fi
exit "$status"
