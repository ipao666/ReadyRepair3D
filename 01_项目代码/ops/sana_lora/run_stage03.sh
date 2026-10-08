#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/../.." && pwd)"
STAGE01="${STAGE01_FREEZE:-/root/readyrepair3d_runs/stages/stage01_ready3d_v3_r2}"
STAGE02="${STAGE02_FREEZE:-/root/readyrepair3d_runs/stages/stage02_sana_candidates}"
RUN_ROOT="${STAGE03_RUN_ROOT:-/root/readyrepair3d_runs/stages/stage03_hunyuan_labels}"

for stage in "$STAGE01" "$STAGE02"; do
  test -f "$stage/READY"
  (cd "$stage" && sha256sum -c --quiet SHA256SUMS)
done
test ! -f "$RUN_ROOT/READY"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/features"

source "$ROOT/activate.sh"
python "$ROOT/ops/extract_ready3d_features.py" \
  --manifest "$STAGE02/manifest.jsonl" \
  --output-dir "$RUN_ROOT/features" \
  --dino "$ROOT/models/dinov2-large" \
  --depth "$ROOT/models/Depth-Anything-V2-Large-hf" \
  --birefnet "$ROOT/models/BiRefNet" \
  --device cuda \
  --limit 840

python "$OPS_DIR/predict_ready3d_v3_batch.py" \
  --candidates "$STAGE02/manifest.jsonl" \
  --features "$RUN_ROOT/features/features.jsonl" \
  --embeddings "$RUN_ROOT/features/dino_embeddings.npy" \
  --checkpoint "$STAGE01/ready3d_v3_top1.joblib" \
  --output "$RUN_ROOT/ready3d_predictions.jsonl" \
  --selections "$RUN_ROOT/ready3d_selections.jsonl" \
  --splits train,validation

python "$OPS_DIR/build_hunyuan_queue.py" \
  --candidates "$STAGE02/manifest.jsonl" \
  --predictions "$RUN_ROOT/ready3d_predictions.jsonl" \
  --features "$RUN_ROOT/features/features.jsonl" \
  --output "$RUN_ROOT/hunyuan_queue.jsonl" \
  --coverage "$RUN_ROOT/coverage.json"

source "$ROOT/activate_hunyuan21.sh"
python "$ROOT/ops/run_hunyuan_ready3d.py" \
  --stage all \
  --manifest "$RUN_ROOT/hunyuan_queue.jsonl" \
  --output-root "$RUN_ROOT/hunyuan" \
  --allow-partial-groups \
  --stop-vram-mib "${R3D_STOP_VRAM_MIB:-38000}"

# A second resume-safe pass also validates that every Blender subprocess has
# fully published all eight views before quality scoring starts.
python "$ROOT/ops/render_hunyuan_outputs.py" \
  --status "$RUN_ROOT/hunyuan/status.jsonl" \
  --output-root "$RUN_ROOT/renders" \
  --expect 300 \
  --workers "${R3D_RENDER_WORKERS:-2}"
python "$ROOT/ops/render_hunyuan_outputs.py" \
  --status "$RUN_ROOT/hunyuan/status.jsonl" \
  --output-root "$RUN_ROOT/renders" \
  --expect 300 \
  --workers "${R3D_RENDER_WORKERS:-2}"

python "$ROOT/ops/score_hunyuan_outputs.py" \
  --status "$RUN_ROOT/hunyuan/status.jsonl" \
  --render-root "$RUN_ROOT/renders" \
  --output-dir "$RUN_ROOT/quality" \
  --dino-model "$ROOT/models/dinov2-large" \
  --birefnet-model "$ROOT/models/BiRefNet" \
  --expect 300 \
  --device cuda \
  --calibration-in "$STAGE01/scoring_calibration.json" \
  --fit-source sana_lora_stage03_fixed_queue

source "$ROOT/activate.sh"
allow_flag=()
if python - "$STAGE01/pseudolabel_gate.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["allow_ready3d_v3_pseudo_labels"] else 1)
PY
then
  allow_flag+=(--allow-pseudo-labels)
fi
python "$OPS_DIR/calibrate_ready3d_labels.py" \
  --predictions "$RUN_ROOT/ready3d_predictions.jsonl" \
  --hunyuan-labels "$RUN_ROOT/quality/quality_labels.jsonl" \
  --output "$RUN_ROOT/candidate_labels.jsonl" \
  --calibration "$RUN_ROOT/ready3d_calibration.json" \
  "${allow_flag[@]}"
