#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/../.." && pwd)"
STAGE01="${STAGE01_FREEZE:-/root/readyrepair3d_runs/stages/stage01_ready3d_v3_r2}"
STAGE02="${STAGE02_FREEZE:-/root/readyrepair3d_runs/stages/stage02_sana_candidates}"
STAGE04="${STAGE04_RUN_ROOT:-/root/readyrepair3d_runs/stages/stage04_lora_training}"
PROXY="$STAGE04/proxy_validation"

for stage in "$STAGE01" "$STAGE02"; do
  test -f "$stage/READY"
  (cd "$stage" && sha256sum -c --quiet SHA256SUMS)
done
test -f "$STAGE04/training_summary.json"
mkdir -p "$PROXY"

source /root/miniconda3/etc/profile.d/conda.sh
source "$ROOT/activate.sh"
python - "$STAGE02/manifest.jsonl" "$PROXY/validation_candidates.jsonl" <<'PY'
import json, sys
source, target = sys.argv[1:]
rows = [json.loads(line) for line in open(source) if line.strip()]
selected = [row for row in rows if row.get("split") == "validation"]
if len(selected) != 120:
    raise SystemExit(f"expected 120 validation candidates, found {len(selected)}")
with open(target, "w") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

metrics=()
for step in 250 500 750 1000 1250 1500; do
  checkpoint="$STAGE04/run_r1/checkpoint-$step"
  output="$PROXY/checkpoint-$step"
  python "$OPS_DIR/generate_with_lora.py" \
    --candidates "$PROXY/validation_candidates.jsonl" \
    --optimized-prompts "$STAGE02/manifests/optimized_prompts.jsonl" \
    --output-root "$output" \
    --model "$ROOT/models/SANA1.5_1.6B_1024px_diffusers" \
    --lora "$checkpoint"
  python "$OPS_DIR/score_prompt_adherence.py" \
    --manifest "$output/manifest.jsonl" \
    --output "$output/adherence.jsonl" \
    --model "$ROOT/models/Qwen3-VL-8B-Instruct" \
    --fail-fast
  python "$ROOT/ops/extract_ready3d_features.py" \
    --manifest "$output/manifest.jsonl" \
    --output-dir "$output/features" \
    --dino "$ROOT/models/dinov2-large" \
    --depth "$ROOT/models/Depth-Anything-V2-Large-hf" \
    --birefnet "$ROOT/models/BiRefNet" \
    --device cuda
  python "$OPS_DIR/predict_ready3d_v3_batch.py" \
    --candidates "$output/manifest.jsonl" \
    --features "$output/features/features.jsonl" \
    --embeddings "$output/features/dino_embeddings.npy" \
    --checkpoint "$STAGE01/ready3d_v3_top1.joblib" \
    --output "$output/ready3d_predictions.jsonl" \
    --selections "$output/ready3d_selections.jsonl" \
    --splits validation
  metric="$output/proxy_metrics.jsonl"
  python "$OPS_DIR/build_proxy_metrics.py" \
    --manifest "$output/manifest.jsonl" \
    --predictions "$output/ready3d_predictions.jsonl" \
    --adherence "$output/adherence.jsonl" \
    --checkpoint "checkpoint-$step" \
    --output "$metric"
  metrics+=("$metric")
done
python - "${metrics[@]}" "$PROXY/proxy_metrics.jsonl" <<'PY'
import sys
sources, target = sys.argv[1:-1], sys.argv[-1]
with open(target, "w") as out:
    for source in sources:
        out.write(open(source).read())
PY
python "$OPS_DIR/evaluate_lora_candidates.py" \
  --metrics "$PROXY/proxy_metrics.jsonl" \
  --output "$PROXY/checkpoint_ranking.jsonl" \
  --top-k 2
