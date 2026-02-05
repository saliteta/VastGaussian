#!/usr/bin/env bash
# Run one train_vast.py job per subfolder in DATA_ROOT.
# Jobs are distributed across the GPU pool: one job per GPU per wave; next wave starts when the current wave finishes.
# Usage: ./train_multi_gpu.sh [optional extra args for train_vast.py]

set -e

# Data root: each subfolder is one scene → one training job
DATA_ROOT="/mnt/c/Users/bxiong/Desktop/data/cp_result"

# Output base: each job writes to OUTPUT_BASE/<subfolder_name>
OUTPUT_BASE="./output"

# GPU pool: one job per GPU per wave
GPUS=(0 1 2 3)

# Optional: extra arguments passed to train_vast.py (e.g. --iterations 60000)
EXTRA_ARGS=("$@")

# Resolve script dir so we can run train_vast.py from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Collect subfolder names (directories only), sorted so order is deterministic
SUBFOLDERS=()
for d in "$DATA_ROOT"/*; do
  [[ -d "$d" ]] || continue
  SUBFOLDERS+=("$(basename "$d")")
done
readarray -t SUBFOLDERS < <(printf '%s\n' "${SUBFOLDERS[@]}" | sort -V)

if [[ ${#SUBFOLDERS[@]} -eq 0 ]]; then
  echo "No subfolders found in $DATA_ROOT"
  exit 1
fi

NUM_GPUS=${#GPUS[@]}
TOTAL=${#SUBFOLDERS[@]}
echo "Found $TOTAL scene(s), GPU pool: ${GPUS[*]}"

wave=0
for (( start = 0; start < TOTAL; start += NUM_GPUS )); do
  echo "--- Wave $(( wave + 1 )) (scenes $(( start + 1 ))-$(( start + NUM_GPUS < TOTAL ? start + NUM_GPUS : TOTAL )) / $TOTAL) ---"
  pids=()
  for (( g = 0; g < NUM_GPUS && start + g < TOTAL; g++ )); do
    subfolder="${SUBFOLDERS[$(( start + g ))]}"
    gpu_id="${GPUS[$g]}"
    source_path="$DATA_ROOT/$subfolder"
    model_path="$OUTPUT_BASE/$subfolder"
    echo "[$(date +%H:%M:%S)] Launching '$subfolder' on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python train_vast.py \
      -s "$source_path" \
      -m "$model_path" \
      "${EXTRA_ARGS[@]}" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  (( wave++ )) || true
done

echo "All jobs finished."
