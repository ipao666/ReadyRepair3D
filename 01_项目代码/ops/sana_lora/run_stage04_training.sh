#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/../.." && pwd)"
STAGE02="${STAGE02_FREEZE:-/root/readyrepair3d_runs/stages/stage02_sana_candidates}"
STAGE03="${STAGE03_FREEZE:-/root/readyrepair3d_runs/stages/stage03_hunyuan_labels}"
RUN_ROOT="${STAGE04_RUN_ROOT:-/root/readyrepair3d_runs/stages/stage04_lora_training}"
TRAINING_DIR="$RUN_ROOT/run_r1"

for stage in "$STAGE02" "$STAGE03"; do
  test -f "$stage/READY"
  (cd "$stage" && sha256sum -c SHA256SUMS)
done
test ! -f "$RUN_ROOT/READY"
mkdir -p "$RUN_ROOT/logs" "$TRAINING_DIR"
source "$ROOT/activate.sh"

expected="$({ python - "$STAGE03/candidate_labels.jsonl" <<'PY'
import json, sys
rows=[json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
print(sum(row.get("split") == "train" for row in rows))
PY
} )"
if [[ "$expected" != "720" && "$expected" != "180" ]]; then
  echo "unexpected training label count: $expected" >&2
  exit 1
fi

python "$OPS_DIR/build_training_manifest.py" \
  --candidates "$STAGE02/manifest.jsonl" \
  --labels "$STAGE03/candidate_labels.jsonl" \
  --adherence "$STAGE02/adherence_scores.jsonl" \
  --output "$RUN_ROOT/train_weighted.jsonl" \
  --expect "$expected"

cp "$ROOT/configs/sana_lora/train_rank16.yaml" "$RUN_ROOT/train_rank16.yaml"
python "$OPS_DIR/train_quality_lora.py" \
  --config "$RUN_ROOT/train_rank16.yaml" \
  --manifest "$RUN_ROOT/train_weighted.jsonl" \
  --output-dir "$TRAINING_DIR" \
  --expect "$expected" \
  --resume-from-checkpoint latest

python "$OPS_DIR/summarize_stage04.py" \
  --training-dir "$TRAINING_DIR" \
  --output "$RUN_ROOT/training_summary.json"
