#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-stylegan2-celebahq-bw}"
DATA_ZIP="${DATA_ZIP:-${ROOT_DIR}/datasets/celebahq-256x256.zip}"
NETWORK="${NETWORK:-}"
METRICS="${METRICS:-fid50k_full,kid50k_full,is50k,pr50k3_full}"
GPUS="${GPUS:-1}"
MIRROR="${MIRROR:-1}"

if [[ -z "${NETWORK}" ]]; then
  echo "Usage:"
  echo "  NETWORK=/path/to/network-snapshot-XXXXXX.pkl bash scripts/eval_candidate_celebahq_metrics.sh"
  exit 1
fi

if [[ ! -s "${NETWORK}" ]]; then
  echo "Missing network snapshot: ${NETWORK}"
  exit 1
fi

if [[ ! -s "${DATA_ZIP}" ]]; then
  echo "Missing dataset zip: ${DATA_ZIP}"
  exit 1
fi

if [[ -d "/home/tykim/.conda/envs/${CONDA_ENV}/targets/x86_64-linux/include" ]]; then
  export CPATH="/home/tykim/.conda/envs/${CONDA_ENV}/targets/x86_64-linux/include:${CPATH:-}"
fi

PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" python "${ROOT_DIR}/calc_metrics.py" \
  --metrics="${METRICS}" \
  --data="${DATA_ZIP}" \
  --mirror="${MIRROR}" \
  --gpus="${GPUS}" \
  --network="${NETWORK}"
