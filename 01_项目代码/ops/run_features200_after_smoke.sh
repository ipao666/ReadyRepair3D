#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
VALIDATION_PID_FILE="$ROOT/logs/validation64_restore.pid"
VALIDATION_FROZEN="$ROOT/data/hunyuan_validation64/auto_quality_v2/FROZEN"
VALIDATION_FAILED="$ROOT/state/validation64.FAILED"
STATE_DIR="$ROOT/state"
SUCCESS="$STATE_DIR/features200.SUCCESS"
FAILED="$STATE_DIR/features200.FAILED"
LOCK="$ROOT/locks/gpu0.lock"

mkdir -p "$STATE_DIR" "$ROOT/locks"
rm -f "$SUCCESS" "$FAILED"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "$code" >"$FAILED"; fi' EXIT

while [[ ! -f "$VALIDATION_FROZEN" ]]; do
  if [[ -f "$VALIDATION_FAILED" ]]; then
    echo "validation64 failed; refusing to start features200" >&2
    exit 1
  fi
  if [[ -f "$VALIDATION_PID_FILE" ]]; then
    validation_pid="$(cat "$VALIDATION_PID_FILE")"
    if ! kill -0 "$validation_pid" 2>/dev/null; then
      echo "validation64 exited without FROZEN marker" >&2
      exit 1
    fi
    if [[ -r "/proc/$validation_pid/cmdline" ]] &&
       ! tr '\0' ' ' <"/proc/$validation_pid/cmdline" |
         grep -q "run_validation64_after_restore.sh"; then
      echo "validation64 PID was reused by an unrelated process" >&2
      exit 1
    fi
  fi
  sleep 15
done

source "$ROOT/activate.sh"
exec 9>"$LOCK"
flock -x 9
python "$ROOT/ops/extract_ready3d_features.py" \
  --manifest "$ROOT/data/ai_sana_200/manifest.jsonl" \
  --output-dir "$ROOT/data/ready3d_features_200"

python - "$ROOT/data/ready3d_features_200" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "features.jsonl").read_text().splitlines() if line]
embeddings = np.load(root / "dino_embeddings.npy")
schema = json.loads((root / "feature_schema.json").read_text())
if len(rows) != 200 or embeddings.shape != (200, 1024) or schema.get("samples") != 200:
    raise SystemExit(
        f"invalid features200 output: rows={len(rows)} embeddings={embeddings.shape} "
        f"schema_samples={schema.get('samples')}"
    )
if len({row["sample_id"] for row in rows}) != 200:
    raise SystemExit("features200 has duplicate sample IDs")
PY

touch "$SUCCESS"
