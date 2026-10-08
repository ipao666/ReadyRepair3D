#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
LABEL_SUCCESS="$ROOT/state/ready3d_labels.SUCCESS"
LABEL_FAILED="$ROOT/state/ready3d_labels.FAILED"
STATE_DIR="$ROOT/state"
SUCCESS="$STATE_DIR/ready3d_training.SUCCESS"
FAILED="$STATE_DIR/ready3d_training.FAILED"
OUTPUT="$ROOT/evaluation/ready3d_v2_restore"
CHECKPOINT="$ROOT/checkpoints/ready3d_v2/ready3d_v2.joblib"

mkdir -p "$STATE_DIR" "$OUTPUT" "$(dirname "$CHECKPOINT")"
rm -f "$SUCCESS" "$FAILED"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "$code" >"$FAILED"; fi' EXIT

while [[ ! -f "$LABEL_SUCCESS" ]]; do
  if [[ -f "$LABEL_FAILED" ]]; then
    echo "Ready3D label generation failed; refusing to train" >&2
    exit 1
  fi
  sleep 15
done

source "$ROOT/activate.sh"

python "$ROOT/ops/build_ready3d_training_targets.py" \
  --features "$ROOT/data/ready3d_features_200/features.jsonl" \
  --train-manifest "$ROOT/data/ready3d_splits/train96.jsonl" \
  --train-labels "$ROOT/data/hunyuan_ready3d_train/auto_quality_v2/quality_labels.jsonl" \
  --validation-manifest "$ROOT/data/validation64/manifest.jsonl" \
  --validation-labels "$ROOT/data/hunyuan_validation64/auto_quality_v2/quality_labels.jsonl" \
  --test-manifest "$ROOT/data/ready3d_splits/test24.jsonl" \
  --test-labels "$ROOT/data/hunyuan_ready3d_test/auto_quality_v2/quality_labels.jsonl" \
  --calibration-in "$ROOT/data/hunyuan_validation64/auto_quality_v2/calibration.json" \
  --output-dir "$OUTPUT"

python "$ROOT/ops/train_ready3d.py" \
  --labels "$OUTPUT/targets.jsonl" \
  --features "$ROOT/data/ready3d_features_200/features.jsonl" \
  --embeddings "$ROOT/data/ready3d_features_200/dino_embeddings.npy" \
  --calibration "$OUTPUT/calibration.json" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --allow-unsupported-failures

python - "$CHECKPOINT" "$OUTPUT/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

import joblib

checkpoint_path, metrics_path = map(Path, sys.argv[1:])
checkpoint = joblib.load(checkpoint_path)
metrics = json.loads(metrics_path.read_text())
if checkpoint.get("schema_version") != "r3dguard.ready3d-checkpoint.v3":
    raise SystemExit("unexpected Ready3D checkpoint schema")
if metrics.get("sample_counts") != {"train": 96, "validation": 64, "test": 24}:
    raise SystemExit(f"unexpected Ready3D split counts: {metrics.get('sample_counts')}")
PY

touch "$SUCCESS"
