#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ZIP="${DATA_ZIP:-${ROOT_DIR}/datasets/celebv-hq-256x256.zip}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs_celebvhq}"
DESC="${DESC:-celebvhq256-stylegan2-ffhq-aligned-ft}"

DATA_ZIP="${DATA_ZIP}" OUT_DIR="${OUT_DIR}" DESC="${DESC}" METRICS="${METRICS:-fid50k_full,kid50k_full,is50k,pr50k3_full}" bash "${ROOT_DIR}/scripts/train_finetune_mmcelebahq_stylegan2.sh"
