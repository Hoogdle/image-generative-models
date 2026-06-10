#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-stylegan2-bw}"
JSON="${JSON:-/home/tykim/WorkSpace/CelebV-HQ/celebvhq_info.json}"
WORKDIR="${WORKDIR:-/home/tykim/WorkSpace/datasets/celebv-hq}"
ZIP="${ZIP:-${ROOT_DIR}/datasets/celebv-hq-256x256.zip}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-1}"
MAX_CLIPS="${MAX_CLIPS:-0}"
START_INDEX="${START_INDEX:-0}"
NO_ZIP="${NO_ZIP:-0}"

ARGS=(
  "${ROOT_DIR}/scripts/prepare_celebvhq_256.py"
  --json "${JSON}"
  --workdir "${WORKDIR}"
  --zip "${ZIP}"
  --frames-per-clip "${FRAMES_PER_CLIP}"
  --max-clips "${MAX_CLIPS}"
  --start-index "${START_INDEX}"
)

if [[ "${NO_ZIP}" == "1" ]]; then
  ARGS+=(--no-zip)
fi

PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" python "${ARGS[@]}"
