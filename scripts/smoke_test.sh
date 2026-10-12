#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_DATA=false
if [[ "${1:-}" == "--skip-data" ]]; then
  SKIP_DATA=true
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/plato-formal-matplotlib}"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m compileall -q external/Debias_IMU/BiasDy tools
"$PYTHON_BIN" external/Debias_IMU/BiasDy/plato/loss_smoke_test.py
"$PYTHON_BIN" external/Debias_IMU/BiasDy/plato/nll_smoke_test.py
"$PYTHON_BIN" external/Debias_IMU/BiasDy/plato/supervision_smoke_test.py

if [[ "$SKIP_DATA" == false ]]; then
  "$PYTHON_BIN" tools/verify_aqualux_layout.py
fi

printf 'All requested smoke tests passed.\n'
