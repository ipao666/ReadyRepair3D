#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/r3dguard
MANIFEST=${1:-$ROOT/evaluation/e2e16_full_loop/manifest.jsonl}
OUTPUT_ROOT=${2:-$ROOT/evaluation/e2e16_full_loop/runs}
LOG_ROOT=${3:-$ROOT/evaluation/e2e16_full_loop/logs}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
exec 9>"$OUTPUT_ROOT/.batch.lock"
if ! flock -n 9; then
  echo "Another e2e16 batch is already running." >&2
  exit 3
fi

mapfile -t ROWS < <(
  /opt/conda/envs/hunyuan21/bin/python - "$MANIFEST" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
for row in rows:
    print(f"{row['prompt_id']}\t{int(row['base_seed'])}\t{row['prompt']}")
PY
)

total=${#ROWS[@]}
index=0
for row in "${ROWS[@]}"; do
  index=$((index + 1))
  IFS=$'\t' read -r prompt_id seed prompt <<<"$row"
  sample_dir="$OUTPUT_ROOT/$prompt_id"
  sample_log="$LOG_ROOT/$prompt_id.log"
  mkdir -p "$sample_dir"

  if [[ -s "$sample_dir/final.glb" && -f "$sample_dir/PIPELINE_COMPLETE" ]]; then
    echo "[$index/$total] SKIP_COMPLETE $prompt_id"
    continue
  fi

  rm -f "$sample_dir/BATCH_SAMPLE_COMPLETE" "$sample_dir/BATCH_SAMPLE_FAILED"
  printf '%s\n' "$prompt_id" >"$OUTPUT_ROOT/CURRENT_SAMPLE"
  echo "[$index/$total] START $prompt_id seed=$seed"
  if "$ROOT/ops/run_full_quality_pipeline.sh" \
      --prompt "$prompt" \
      --output-dir "$sample_dir" \
      --seed "$seed" \
      >"$sample_log" 2>&1; then
    if [[ -s "$sample_dir/final.glb" && -f "$sample_dir/PIPELINE_COMPLETE" ]]; then
      touch "$sample_dir/BATCH_SAMPLE_COMPLETE"
      echo "[$index/$total] COMPLETE $prompt_id"
    else
      touch "$sample_dir/BATCH_SAMPLE_FAILED"
      echo "[$index/$total] FAILED_MISSING_ARTIFACT $prompt_id" >&2
    fi
  else
    exit_code=$?
    printf '%s\n' "$exit_code" >"$sample_dir/exit_code.txt"
    touch "$sample_dir/BATCH_SAMPLE_FAILED"
    echo "[$index/$total] FAILED $prompt_id exit=$exit_code" >&2
  fi
done

rm -f "$OUTPUT_ROOT/CURRENT_SAMPLE"
touch "$OUTPUT_ROOT/BATCH_COMPLETE"
echo "BATCH_COMPLETE total=$total"
