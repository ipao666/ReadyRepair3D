#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
STATE="$ROOT/state"
FROZEN="$ROOT/data/hunyuan_validation64/auto_quality_v2/FROZEN"

mkdir -p "$STATE" "$ROOT/logs"
rm -f \
  "$STATE/validation64.FAILED" \
  "$STATE/validation64.SUCCESS" \
  "$STATE/features200.FAILED" \
  "$STATE/features200.SUCCESS" \
  "$STATE/ready3d_labels.FAILED" \
  "$STATE/ready3d_labels.SUCCESS" \
  "$STATE/ready3d_training.FAILED" \
  "$STATE/ready3d_training.SUCCESS" \
  "$STATE/full_chain.FAILED" \
  "$STATE/full_chain.SUCCESS" \
  "$FROZEN"

if "$ROOT/ops/finalize_validation64.sh"; then
  touch "$STATE/validation64.SUCCESS"
else
  code=$?
  printf '%s\n' "$code" >"$STATE/validation64.FAILED"
  exit "$code"
fi

"$ROOT/ops/run_features200_after_smoke.sh"
"$ROOT/ops/run_ready3d_labels_after_features.sh"
"$ROOT/ops/run_ready3d_training_after_labels.sh"
"$ROOT/ops/run_full_chain_after_training.sh"
