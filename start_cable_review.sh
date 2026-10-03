#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing Linux runtime. Run: bash $root/runtime/bootstrap.sh --recreate" >&2
  exit 1
fi
cd "$root"
exec "$python" "$root/prototype/assembly_auto_review_dino_v2.py"
