#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
LOG="$ROOT/logs/regenerate_sana200.log"
OUTPUT="$ROOT/data/ai_sana_200"

source "$ROOT/activate.sh"
mkdir -p "$ROOT/logs" "$OUTPUT"

echo "[$(date -Is)] SANA-200 regeneration started" >>"$LOG"
python "$ROOT/ops/generate_sana_200.py" \
  --max-groups 50 \
  --output-dir "$OUTPUT" \
  >>"$LOG" 2>&1
python "$ROOT/ops/verify_sana_200.py" "$OUTPUT" >>"$LOG" 2>&1
echo "[$(date -Is)] SANA-200 regeneration completed" >>"$LOG"
