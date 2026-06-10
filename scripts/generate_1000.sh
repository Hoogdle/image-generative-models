#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-stylegan2-bw}"
NETWORK="${NETWORK:-${ROOT_DIR}/pretrained/stylegan2-ffhq-256x256.pkl}"
OUTDIR="${OUTDIR:-${ROOT_DIR}/generated/stylegan2_ffhq256_1000}"

if [[ ! -s "${NETWORK}" ]]; then
  echo "Missing network: ${NETWORK}"
  echo "Set NETWORK=/path/to/network.pkl or download the pretrained model."
  exit 1
fi

PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" python "${ROOT_DIR}/scripts/generate_1000.py" \
  --network "${NETWORK}" \
  --outdir "${OUTDIR}" \
  --num-images "${NUM_IMAGES:-1000}" \
  --start-seed "${START_SEED:-0}" \
  --trunc "${TRUNC:-1.0}" \
  --noise-mode "${NOISE_MODE:-const}"
