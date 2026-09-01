#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
FEATURE_SUCCESS="$ROOT/state/features200.SUCCESS"
FEATURE_FAILED="$ROOT/state/features200.FAILED"
STATE_DIR="$ROOT/state"
SUCCESS="$STATE_DIR/ready3d_labels.SUCCESS"
FAILED="$STATE_DIR/ready3d_labels.FAILED"
LOCK="$ROOT/locks/gpu0.lock"

mkdir -p "$STATE_DIR" "$ROOT/locks"
rm -f "$SUCCESS" "$FAILED"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "$code" >"$FAILED"; fi' EXIT

while [[ ! -f "$FEATURE_SUCCESS" ]]; do
  if [[ -f "$FEATURE_FAILED" ]]; then
    echo "features200 failed; refusing to generate Ready3D labels" >&2
    exit 1
  fi
  sleep 15
done

exec 9>"$LOCK"
flock -x 9
"$ROOT/ops/run_ready3d_labels_after_freeze.sh"
test -f "$ROOT/data/ready3d_labels/hunyuan_v2/COMPLETE"
touch "$SUCCESS"
