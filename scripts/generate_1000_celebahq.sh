#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-stylegan2-celebahq-bw}"
NETWORK="${NETWORK:-${ROOT_DIR}/pretrained/stylegan2-celebahq-256x256.pkl}"
OUTDIR="${OUTDIR:-${ROOT_DIR}/generated/celebahq_stylegan2_1000}"

if [[ ! -s "${NETWORK}" ]]; then
  echo "Missing network: ${NETWORK}"
  exit 1
fi

if [[ -d "/home/tykim/.conda/envs/${CONDA_ENV}/targets/x86_64-linux/include" ]]; then
  export CPATH="/home/tykim/.conda/envs/${CONDA_ENV}/targets/x86_64-linux/include:${CPATH:-}"
fi

PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" python "${ROOT_DIR}/scripts/generate_1000.py" \
  --network "${NETWORK}" \
  --outdir "${OUTDIR}" \
  --num-images "${NUM_IMAGES:-1000}" \
  --start-seed "${START_SEED:-0}" \
  --trunc "${TRUNC:-1.0}" \
  --noise-mode "${NOISE_MODE:-const}"
