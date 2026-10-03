#!/usr/bin/env bash
set -euo pipefail

recreate=0
if [[ "${1:-}" == "--recreate" ]]; then
  recreate=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash runtime/bootstrap.sh [--recreate]" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.11}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Set PYTHON_BIN if it is installed under another name." >&2
  exit 1
fi
version="$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$version" != "3.11" ]]; then
  echo "Python 3.11 is required; found $version at $python_bin" >&2
  exit 1
fi

ensure_venv() {
  local venv_path="$1"
  local requirements_path="$2"
  local venv_python="$venv_path/bin/python"
  if [[ ! -x "$venv_python" ]]; then
    if [[ -e "$venv_path" ]]; then
      if [[ "$recreate" -ne 1 ]]; then
        echo "Existing non-Linux environment at $venv_path. Re-run with --recreate to replace it." >&2
        exit 1
      fi
      rm -rf -- "$venv_path"
    fi
    "$python_bin" -m venv "$venv_path"
  fi
  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install -r "$requirements_path"
}

main_venv="$root/.venv"
ensure_venv "$main_venv" "$root/runtime/requirements-main.txt"

sam3_venv="$root/runtime/sam3/.venv"
ensure_venv "$sam3_venv" "$root/runtime/sam3/requirements.txt"
"$sam3_venv/bin/python" -m pip install --no-deps -e "$root/runtime/sam3/source"

"$main_venv/bin/python" -B -c "import cv2, numpy, torch; from PyQt5 import QtCore; print('main runtime ready')"
PYTHONPATH="$root/runtime/sam3/source" "$sam3_venv/bin/python" -B -c "import cv2, numpy, torch, sam3; print('sam3 runtime ready')"
echo "Runtime setup complete. Start with: bash $root/start_cable_review.sh"
