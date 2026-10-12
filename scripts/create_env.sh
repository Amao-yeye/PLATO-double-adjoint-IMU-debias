#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$REPO_ROOT[test]"

printf 'Environment ready. Activate with: source %s/bin/activate\n' "$VENV_DIR"
