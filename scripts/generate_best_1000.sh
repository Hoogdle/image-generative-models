#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEST_LINE="$(cd "${ROOT_DIR}" && conda run -n stylegan2-bw python scripts/select_best_snapshot.py --runs-dir runs | grep '^best_source=')"
NETWORK="${BEST_LINE#best_source=}"
OUTDIR="${OUTDIR:-${ROOT_DIR}/generated/stylegan2_best_1000}"

NETWORK="${NETWORK}" OUTDIR="${OUTDIR}" bash "${ROOT_DIR}/scripts/generate_1000.sh"
