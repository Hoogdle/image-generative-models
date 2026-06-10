#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/pretrained"
MODEL_NAME="stylegan2-celebahq-256x256.pkl"
MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/${MODEL_NAME}"

mkdir -p "${OUT_DIR}"

if [[ -s "${OUT_DIR}/${MODEL_NAME}" ]]; then
  echo "Already exists: ${OUT_DIR}/${MODEL_NAME}"
  exit 0
fi

curl -L --fail --retry 3 --retry-delay 5 \
  -o "${OUT_DIR}/${MODEL_NAME}" \
  "${MODEL_URL}"

echo "Downloaded: ${OUT_DIR}/${MODEL_NAME}"
