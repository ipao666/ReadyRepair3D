#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
TRAIN_SUCCESS="$ROOT/state/ready3d_training.SUCCESS"
TRAIN_FAILED="$ROOT/state/ready3d_training.FAILED"
STATE_DIR="$ROOT/state"
SUCCESS="$STATE_DIR/full_chain.SUCCESS"
FAILED="$STATE_DIR/full_chain.FAILED"
LOCK="$ROOT/locks/gpu0.lock"
OUTPUT="$ROOT/evaluation/restored_full_chain_smoke"

mkdir -p "$STATE_DIR" "$ROOT/locks"
rm -f "$SUCCESS" "$FAILED"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "$code" >"$FAILED"; fi' EXIT

while [[ ! -f "$TRAIN_SUCCESS" ]]; do
  if [[ -f "$TRAIN_FAILED" ]]; then
    echo "Ready3D training failed; refusing to run the full-chain smoke" >&2
    exit 1
  fi
  sleep 15
done

exec 9>"$LOCK"
flock -x 9

"$ROOT/ops/run_full_quality_pipeline.sh" \
  --prompt "a compact red retro desk radio with one tuning dial and a carry handle" \
  --output-dir "$OUTPUT" \
  --seed 2026072001

source "$ROOT/activate_hunyuan21.sh"
python "$ROOT/ops/verify_glb.py" "$OUTPUT/final.glb"
python - "$OUTPUT/pipeline_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if summary.get("round_count") not in {1, 2}:
    raise SystemExit(f"invalid full-chain round count: {summary.get('round_count')}")
if not Path(summary["final_glb"]).is_file():
    raise SystemExit("full-chain summary points to a missing GLB")
PY

touch "$SUCCESS"
