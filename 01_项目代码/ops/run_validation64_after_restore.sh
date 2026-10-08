#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
SMOKE_STATUS="$ROOT/tmp/hunyuan_restore_smoke/status.jsonl"
STATE_DIR="$ROOT/state"
SUCCESS="$STATE_DIR/validation64.SUCCESS"
FAILED="$STATE_DIR/validation64.FAILED"
LOCK="$ROOT/locks/gpu0.lock"

mkdir -p "$STATE_DIR" "$ROOT/locks"
rm -f "$SUCCESS" "$FAILED" \
  "$ROOT/data/hunyuan_validation64/auto_quality_v2/FROZEN"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "$code" >"$FAILED"; fi' EXIT

smoke_glb="$(
  /opt/conda/envs/hunyuan21/bin/python - "$SMOKE_STATUS" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
if len(rows) != 1 or rows[0].get("paint", {}).get("status") != "success":
    raise SystemExit("Hunyuan restore smoke did not finish successfully")
print(rows[0]["paint"]["artifact_path"])
PY
)"

source "$ROOT/activate_hunyuan21.sh"
python "$ROOT/ops/verify_glb.py" "$smoke_glb"

exec 9>"$LOCK"
flock -x 9
"$ROOT/ops/run_validation64_supervisor.sh"
"$ROOT/ops/finalize_validation64.sh"
touch "$SUCCESS"
