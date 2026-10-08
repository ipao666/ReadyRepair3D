#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/_env.sh"

RUN_ROOT="${R3D_V3_RUN_ROOT:-$ROOT/data/ready3d_v3}"
CALIBRATION="${R3D_V3_CALIBRATION:-$(resolve_calibration)}"
FEATURES="$RUN_ROOT/features"
QUALITY="$RUN_ROOT/auto_quality_v2"
SPLITS="$RUN_ROOT/splits"
TRAINING="$RUN_ROOT/training"
V3_CHECKPOINT="${R3D_V3_CHECKPOINT:-$ROOT/checkpoints/ready3d_v3_top1/ready3d_v3_top1.joblib}"
V2_CHECKPOINT="${R3D_V2_CHECKPOINT:-$ROOT/checkpoints/ready3d_v2/ready3d_v2.joblib}"

test -s "$RUN_ROOT/manifest.jsonl"
test -s "$RUN_ROOT/hunyuan/status.jsonl"
test -s "$CALIBRATION"
test -s "$V2_CHECKPOINT"
mkdir -p "$FEATURES" "$QUALITY" "$SPLITS" "$TRAINING" "$(dirname "$V3_CHECKPOINT")"

source "$ROOT/activate.sh"
python "$ROOT/ops/extract_ready3d_features.py" \
  --manifest "$RUN_ROOT/manifest.jsonl" \
  --output-dir "$FEATURES" \
  --dino "$R3DGUARD_MODELS/dinov2-large" \
  --depth "$R3DGUARD_MODELS/Depth-Anything-V2-Large-hf" \
  --birefnet "$R3DGUARD_MODELS/BiRefNet" \
  --device cuda

source "$ROOT/activate_hunyuan21.sh"
python "$ROOT/ops/score_hunyuan_outputs.py" \
  --status "$RUN_ROOT/hunyuan/status.jsonl" \
  --render-root "$RUN_ROOT/renders" \
  --output-dir "$QUALITY" \
  --dino-model "$R3DGUARD_MODELS/dinov2-large" \
  --birefnet-model "$R3DGUARD_MODELS/BiRefNet" \
  --expect 320 \
  --device cuda \
  --calibration-in "$CALIBRATION" \
  --fit-source ready3d_v3_fixed_80_groups

source "$ROOT/activate.sh"
python "$ROOT/ops/prepare_ready3d_v3_training_inputs.py" \
  --manifest "$RUN_ROOT/manifest.jsonl" \
  --labels "$QUALITY/quality_labels.jsonl" \
  --output-dir "$SPLITS"

python "$ROOT/ops/build_ready3d_training_targets.py" \
  --features "$FEATURES/features.jsonl" \
  --train-manifest "$SPLITS/train_manifest.jsonl" \
  --train-labels "$SPLITS/train_labels.jsonl" \
  --validation-manifest "$SPLITS/validation_manifest.jsonl" \
  --validation-labels "$SPLITS/validation_labels.jsonl" \
  --test-manifest "$SPLITS/test_manifest.jsonl" \
  --test-labels "$SPLITS/test_labels.jsonl" \
  --calibration-in "$CALIBRATION" \
  --output-dir "$TRAINING"

python "$ROOT/ops/train_ready3d_v3_top1.py" \
  --labels "$TRAINING/targets.jsonl" \
  --features "$FEATURES/features.jsonl" \
  --embeddings "$FEATURES/dino_embeddings.npy" \
  --checkpoint "$V3_CHECKPOINT" \
  --output-dir "$TRAINING/model"

python "$ROOT/ops/build_ready3d_v3_strategy_input.py" \
  --targets "$TRAINING/targets.jsonl" \
  --features "$FEATURES/features.jsonl" \
  --embeddings "$FEATURES/dino_embeddings.npy" \
  --ready-v2-checkpoint "$V2_CHECKPOINT" \
  --ready-v3-checkpoint "$V3_CHECKPOINT" \
  --split test \
  --output "$TRAINING/strategy_test_candidates.jsonl"

python "$ROOT/ops/evaluate_ready3d_v3_strategies.py" \
  --input "$TRAINING/strategy_test_candidates.jsonl" \
  --output-dir "$TRAINING/strategy_evaluation" \
  --bootstrap-draws 10000 \
  --seed 20260827

python "$ROOT/ops/run_ready3d_v3_server_preflight.py" \
  --manifest "$RUN_ROOT/manifests/candidates320.jsonl" \
  --output-root "$RUN_ROOT" \
  --labels "$QUALITY/quality_labels.jsonl" \
  --expect 320 \
  --report "$RUN_ROOT/preflight_complete.json"
