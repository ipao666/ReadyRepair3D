#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R3DGUARD_HOME="$ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
python -m pytest tests -q --import-mode=importlib
python "$ROOT/ops/build_sha256s.py" --root "$ROOT" --verify
