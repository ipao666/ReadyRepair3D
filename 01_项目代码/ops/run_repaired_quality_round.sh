#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
HUNYUAN_PYTHON="$(hunyuan_python)"
PROMPT=""
OUTPUT_DIR=""
SEED=20260717
QUALITY_THRESHOLD=0.806912747446761
CALIBRATION="$(resolve_calibration)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --quality-threshold) QUALITY_THRESHOLD="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROMPT" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --prompt TEXT --output-dir DIR [--seed N] [--quality-threshold FLOAT]" >&2
  exit 2
fi

OUTPUT_DIR=$(
  "$HUNYUAN_PYTHON" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "$OUTPUT_DIR"
)
REPAIR="$OUTPUT_DIR/repair3d"
POST="$OUTPUT_DIR/post_repair"
ROUND_FINAL="$OUTPUT_DIR/round_final"
mkdir -p "$OUTPUT_DIR"

"$ROOT/ops/run_top2_dual3d_stage1.sh" \
  --prompt "$PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED"

source "$ROOT/activate_hunyuan21.sh"

if [[ ! -f "$REPAIR/COMPLETE" ]]; then
  python "$ROOT/ops/run_repair3d_auto.py" \
    "$OUTPUT_DIR/selected_3d/final.glb" \
    --output-dir "$REPAIR"
  [[ -s "$REPAIR/final.glb" ]]
  [[ -s "$REPAIR/repair_report.json" ]]
  touch "$REPAIR/COMPLETE"
fi

REPAIR_CANDIDATE_READY=$(
  "$HUNYUAN_PYTHON" -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1])).get("candidate_ready_for_rescore") else "false")' \
    "$REPAIR/repair_report.json"
)

REPAIRED_LABEL_ARGS=()
if [[ "$REPAIR_CANDIDATE_READY" == "true" ]]; then
  if [[ ! -f "$POST/COMPLETE" ]]; then
    mkdir -p "$POST"
    python "$ROOT/ops/build_post_repair_status.py" \
      --labels "$OUTPUT_DIR/quality_v2/quality_labels.jsonl" \
      --selection-report "$OUTPUT_DIR/selected_3d/selection_report.json" \
      --repaired-glb "$REPAIR/candidate.glb" \
      --output "$POST/status.jsonl"
    python "$ROOT/ops/render_hunyuan_outputs.py" \
      --status "$POST/status.jsonl" \
      --output-root "$POST/renders" \
      --expect 1 \
      --workers 1
    python "$ROOT/ops/score_hunyuan_outputs.py" \
      --status "$POST/status.jsonl" \
      --render-root "$POST/renders" \
      --output-dir "$POST/quality_v2" \
      --dino-model "$R3DGUARD_MODELS/dinov2-large" \
      --birefnet-model "$R3DGUARD_MODELS/BiRefNet" \
      --expect 1 \
      --device cuda \
      --calibration-in "$CALIBRATION"
    [[ $(wc -l < "$POST/quality_v2/quality_labels.jsonl") -eq 1 ]]
    touch "$POST/COMPLETE"
  fi
  REPAIRED_LABEL_ARGS=(
    --repaired-labels "$POST/quality_v2/quality_labels.jsonl"
  )
fi

if [[ ! -f "$ROUND_FINAL/COMPLETE" ]]; then
  python "$ROOT/ops/finalize_repaired_round.py" \
    --original-labels "$OUTPUT_DIR/quality_v2/quality_labels.jsonl" \
    --selection-report "$OUTPUT_DIR/selected_3d/selection_report.json" \
    --repair-report "$REPAIR/repair_report.json" \
    "${REPAIRED_LABEL_ARGS[@]}" \
    --output-dir "$ROUND_FINAL" \
    --threshold "$QUALITY_THRESHOLD"
  [[ -s "$OUTPUT_DIR/round_final/final.glb" ]]
  [[ -s "$ROUND_FINAL/round_report.json" ]]
  touch "$ROUND_FINAL/COMPLETE"
fi

touch "$OUTPUT_DIR/ROUND_COMPLETE"
