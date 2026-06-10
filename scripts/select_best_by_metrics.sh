#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-stylegan2-bw}"
RUN_DIR="${RUN_DIR:-}"
RUNS_DIR="${RUNS_DIR:-${ROOT_DIR}/runs_celebvhq}"
OUTDIR="${OUTDIR:-${ROOT_DIR}/best_metrics}"
TAG="${TAG:-stylegan2-celebvhq}"
TOPPR_KEY="${TOPPR_KEY:-pr50k3_full_recall}"

ARGS=(
  "${ROOT_DIR}/scripts/select_best_by_metrics.py"
  --runs-dir "${RUNS_DIR}"
  --outdir "${OUTDIR}"
  --tag "${TAG}"
  --toppr-key "${TOPPR_KEY}"
  --copy
)

if [[ -n "${RUN_DIR}" ]]; then
  ARGS+=(--run-dir "${RUN_DIR}")
fi

conda run --no-capture-output -n "${CONDA_ENV}" python "${ARGS[@]}"
