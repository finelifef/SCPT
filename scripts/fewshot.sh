#!/usr/bin/env bash
# The only experiment script: one dataset, configurable shots, three seeds.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: DATA_ROOT=/path/to/datasets bash scripts/fewshot.sh DATASET [KEY VALUE ...]" >&2
  exit 2
fi
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the dataset folders}"
DATASET="$1"
shift
SHOTS="${SHOTS:-16}"
SEEDS="${SEEDS:-1 2 3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
read -r -a SEED_LIST <<< "$SEEDS"
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "SEEDS must contain at least one nonnegative integer" >&2
  exit 2
fi
for SEED in "${SEED_LIST[@]}"; do
  ARGS=(--root "$DATA_ROOT" --dataset "$DATASET" --shots "$SHOTS" --seed "$SEED"
        --output-dir "${OUTPUT_ROOT}/${DATASET}/${SHOTS}shot/seed${SEED}")
  if [[ -n "${CLIP_CHECKPOINT:-}" ]]; then
    ARGS+=(--clip-checkpoint "$CLIP_CHECKPOINT")
  fi
  if [[ -n "${SPLIT_DIR:-}" ]]; then
    ARGS+=(--split-dir "$SPLIT_DIR")
  fi
  "$PYTHON" "${REPO_ROOT}/train.py" "${ARGS[@]}" "$@"
done
