#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
OUTPUT="$ROOT/tmp/hunyuan_restore_smoke"
LOG="$ROOT/logs/hunyuan_restore_smoke.log"

source "$ROOT/activate_hunyuan21.sh"
mkdir -p "$ROOT/logs"
rm -rf "$OUTPUT"

python "$ROOT/ops/run_hunyuan_ready3d.py" \
  --stage all \
  --manifest "$ROOT/data/ai_sana_200/manifest.jsonl" \
  --output-root "$OUTPUT" \
  --limit 1 \
  >"$LOG" 2>&1
