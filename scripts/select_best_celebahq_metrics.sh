#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-stylegan2-celebahq-bw}" \
RUNS_DIR="${RUNS_DIR:-${ROOT_DIR}/runs_celebahq}" \
OUTDIR="${OUTDIR:-${ROOT_DIR}/best_metrics_celebahq}" \
TAG="${TAG:-stylegan2-celebahq}" \
TOPPR_KEY="${TOPPR_KEY:-pr50k3_full_recall}" \
bash "${ROOT_DIR}/scripts/select_best_by_metrics.sh"
