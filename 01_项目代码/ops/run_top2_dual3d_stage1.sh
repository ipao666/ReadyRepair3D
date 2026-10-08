#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
HUNYUAN_PYTHON="$(hunyuan_python)"
PROMPT=""
OUTPUT_DIR=""
SEED=20260717
CALIBRATION="$(resolve_calibration)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROMPT" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --prompt TEXT --output-dir DIR [--seed INTEGER]" >&2
  exit 2
fi
if [[ ! -s "$CALIBRATION" ]]; then
  echo "Frozen validation calibration is missing: $CALIBRATION" >&2
  exit 1
fi

OUTPUT_DIR=$(
  "$HUNYUAN_PYTHON" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "$OUTPUT_DIR"
)
CANDIDATES="$OUTPUT_DIR/candidates"
OPTIMIZED_PROMPT="$OUTPUT_DIR/optimized_prompt.jsonl"
FEATURES="$OUTPUT_DIR/features"
READY="$OUTPUT_DIR/ready3d"
SELECTED="$OUTPUT_DIR/selected_manifest.jsonl"
HUNYUAN="$OUTPUT_DIR/hunyuan"
RENDER_ROOT="$OUTPUT_DIR/renders"
QUALITY="$OUTPUT_DIR/quality_v2"
SELECTED_3D="$OUTPUT_DIR/selected_3d"
STARTED_AT=$(date +%s)
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$CANDIDATES/COMPLETE" ]]; then
  source "$ROOT/activate.sh"
  python "$ROOT/ops/optimize_chinese_prompts.py" \
    --prompt "$PROMPT" \
    --output "$OPTIMIZED_PROMPT" \
    --model "$R3DGUARD_MODELS/Qwen3-8B"
  [[ $(wc -l < "$OPTIMIZED_PROMPT") -eq 1 ]]
  python "$ROOT/ops/generate_sana_from_optimized_prompts.py" \
    --input "$OPTIMIZED_PROMPT" \
    --output-dir "$CANDIDATES" \
    --model "$R3DGUARD_MODELS/SANA1.5_1.6B_1024px_diffusers" \
    --base-seed "$SEED"
  [[ $(wc -l < "$CANDIDATES/manifest.jsonl") -eq 4 ]]
  touch "$CANDIDATES/COMPLETE"
fi

source "$ROOT/activate.sh"

if [[ ! -f "$FEATURES/COMPLETE" ]]; then
  python "$ROOT/ops/extract_ready3d_features.py" \
    --manifest "$CANDIDATES/manifest.jsonl" \
    --output-dir "$FEATURES" \
    --dino "$R3DGUARD_MODELS/dinov2-large" \
    --depth "$R3DGUARD_MODELS/Depth-Anything-V2-Large-hf" \
    --birefnet "$R3DGUARD_MODELS/BiRefNet" \
    --device cuda
  [[ $(wc -l < "$FEATURES/features.jsonl") -eq 4 ]]
  touch "$FEATURES/COMPLETE"
fi

if [[ ! -f "$READY/COMPLETE" ]]; then
  python "$ROOT/scripts/score_external64.py" \
    --manifest "$CANDIDATES/manifest.jsonl" \
    --features "$FEATURES/features.jsonl" \
    --embeddings "$FEATURES/dino_embeddings.npy" \
    --checkpoint "$ROOT/checkpoints/ready3d_v2/ready3d_v2.joblib" \
    --output-dir "$READY" \
    --top-k 2
  python "$ROOT/ops/build_selected_manifest.py" \
    --manifest "$CANDIDATES/manifest.jsonl" \
    --predictions "$READY/predictions.jsonl" \
    --output "$SELECTED" \
    --expected-per-group 2
  [[ $(wc -l < "$SELECTED") -eq 2 ]]
  touch "$READY/COMPLETE"
fi

source "$ROOT/activate_hunyuan21.sh"

if [[ ! -f "$HUNYUAN/COMPLETE" ]]; then
  python "$ROOT/ops/run_hunyuan_ready3d.py" \
    --stage all \
    --manifest "$SELECTED" \
    --output-root "$HUNYUAN" \
    --stop-vram-mib 38000 \
    --allow-partial-groups
  [[ $(find "$HUNYUAN/glb" -maxdepth 1 -name '*.glb' -type f | wc -l) -eq 2 ]]
  touch "$HUNYUAN/COMPLETE"
fi

if [[ ! -f "$RENDER_ROOT/COMPLETE" ]]; then
  python "$ROOT/ops/render_hunyuan_outputs.py" \
    --status "$HUNYUAN/status.jsonl" \
    --output-root "$RENDER_ROOT" \
    --expect 2 \
    --workers 2
  [[ $(find "$RENDER_ROOT" -name 'shaded_*.png' -type f | wc -l) -eq 16 ]]
  touch "$RENDER_ROOT/COMPLETE"
fi

if [[ ! -f "$QUALITY/COMPLETE" ]]; then
  python "$ROOT/ops/score_hunyuan_outputs.py" \
    --status "$HUNYUAN/status.jsonl" \
    --render-root "$RENDER_ROOT" \
    --output-dir "$QUALITY" \
    --dino-model "$R3DGUARD_MODELS/dinov2-large" \
    --birefnet-model "$R3DGUARD_MODELS/BiRefNet" \
    --expect 2 \
    --device cuda \
    --calibration-in "$CALIBRATION"
  [[ $(wc -l < "$QUALITY/quality_labels.jsonl") -eq 2 ]]
  touch "$QUALITY/COMPLETE"
fi

if [[ ! -f "$SELECTED_3D/COMPLETE" ]]; then
  python "$ROOT/ops/select_best_hunyuan.py" \
    --labels "$QUALITY/quality_labels.jsonl" \
    --manifest "$SELECTED" \
    --output-dir "$SELECTED_3D" \
    --threshold 0.806912747446761
  [[ -s "$OUTPUT_DIR/selected_3d/final.glb" ]]
  [[ -s "$SELECTED_3D/selection_report.json" ]]
  touch "$SELECTED_3D/COMPLETE"
fi

"$HUNYUAN_PYTHON" - \
  "$PROMPT" "$SEED" "$OUTPUT_DIR" "$STARTED_AT" <<'PY'
import json
import sys
import time
from pathlib import Path

prompt = sys.argv[1]
seed = int(sys.argv[2])
output = Path(sys.argv[3])
started = int(sys.argv[4])
predictions = [
    json.loads(line)
    for line in (output / "ready3d/predictions.jsonl").read_text().splitlines()
    if line.strip()
]
selected = sorted(
    (row for row in predictions if row["selected"]),
    key=lambda row: row["selection_rank"],
)
selection = json.loads(
    (output / "selected_3d/selection_report.json").read_text()
)
summary = {
    "schema_version": "r3dguard.top2-dual3d-stage1.v1",
    "prompt": prompt,
    "seed": seed,
    "candidate_count": len(predictions),
    "selected_2d": [
        {
            "sample_id": row["sample_id"],
            "candidate_index": row["candidate_index"],
            "selection_rank": row["selection_rank"],
            "predicted_quality": row["predicted_quality"],
            "structural_input_pass": row["structural_input_pass"],
        }
        for row in selected
    ],
    "generated_3d_count": 2,
    "winner_sample_id": selection["winner_sample_id"],
    "winner_quality_score_v2": selection["winner_quality_score_v2"],
    "winner_technically_valid": not selection["round_failed"],
    "meets_quality_threshold": selection["meets_quality_threshold"],
    "final_glb": str((output / "selected_3d/final.glb").resolve()),
    "elapsed_seconds": time.time() - started,
    "repair3d_connected": False,
    "second_round_enabled": False,
}
(output / "pipeline_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False), flush=True)
PY

touch "$OUTPUT_DIR/STAGE1_COMPLETE"
