#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ZIP="${DATA_ZIP:-${ROOT_DIR}/datasets/celebahq-256x256.zip}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs_celebahq}"
MODEL="${MODEL:-${ROOT_DIR}/pretrained/stylegan2-celebahq-256x256.pkl}"
CONDA_ENV="${CONDA_ENV:-stylegan2-celebahq-bw}"

if [[ -d "/home/tykim/.conda/envs/${CONDA_ENV}/targets/x86_64-linux/include" ]]; then
  export CPATH="/home/tykim/.conda/envs/${CONDA_ENV}/targets/x86_64-linux/include:${CPATH:-}"
elif [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/targets/x86_64-linux/include" ]]; then
  export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
fi

GPUS="${GPUS:-1}"
BATCH="${BATCH:-32}"
GAMMA="${GAMMA:-5}"
KIMG="${KIMG:-3000}"
SNAP="${SNAP:-5}"
TICK="${TICK:-2}"
METRICS="${METRICS:-fid50k_full}"
WORKERS="${WORKERS:-6}"
SEED="${SEED:-0}"
GLR="${GLR:-}"
DLR="${DLR:-}"
FREEZED="${FREEZED:-0}"
DESC="${DESC:-celebahq256-stylegan2-celebahq-ft-fidonly}"
CBASE="${CBASE:-16384}"
CMAX="${CMAX:-512}"

if [[ ! -s "${MODEL}" ]]; then
  echo "Missing pretrained model: ${MODEL}"
  echo "Run: scripts/download_pretrained_stylegan2_celebahq256.sh"
  exit 1
fi

if [[ ! -s "${DATA_ZIP}" ]]; then
  echo "Missing dataset zip: ${DATA_ZIP}"
  exit 1
fi

ARGS=(
  "${ROOT_DIR}/train.py"
  --outdir="${OUT_DIR}"
  --cfg=stylegan2
  --data="${DATA_ZIP}"
  --gpus="${GPUS}"
  --batch="${BATCH}"
  --cbase="${CBASE}"
  --cmax="${CMAX}"
  --gamma="${GAMMA}"
  --mirror=1
  --aug=ada
  --resume="${MODEL}"
  --kimg="${KIMG}"
  --snap="${SNAP}"
  --tick="${TICK}"
  --metrics="${METRICS}"
  --workers="${WORKERS}"
  --seed="${SEED}"
  --freezed="${FREEZED}"
  --desc="${DESC}"
)

if [[ -n "${GLR}" ]]; then
  ARGS+=(--glr="${GLR}")
fi
if [[ -n "${DLR}" ]]; then
  ARGS+=(--dlr="${DLR}")
fi

PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" python "${ARGS[@]}"
