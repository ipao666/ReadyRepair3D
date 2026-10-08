#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R3DGUARD_HOME="$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT"
if [[ "${1:-}" == "--full" && $# -eq 1 ]]; then
  python -m pytest tests -q --import-mode=importlib
elif [[ $# -eq 0 ]]; then
  python "$ROOT/../tools/check_cpu.py"
else
  echo "Usage: $0 [--full]" >&2
  exit 2
fi
python "$ROOT/ops/build_sha256s.py" --root "$ROOT" --verify
