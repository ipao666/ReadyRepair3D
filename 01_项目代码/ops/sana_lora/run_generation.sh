#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/../.." && pwd)"
RUN_ROOT="${SANA_LORA_RUN_ROOT:-/root/readyrepair3d_runs/stages/stage02_sana_lora_data_work}"
MANIFEST_DIR="$RUN_ROOT/manifests"
PROMPTS="${SANA_LORA_PROMPTS:-$ROOT/examples/sana_lora/prompts240.jsonl}"
BASE_SEED="${SANA_LORA_BASE_SEED:-2026083100}"

mkdir -p "$MANIFEST_DIR" "$RUN_ROOT/logs"
source "$ROOT/activate.sh"

python "$OPS_DIR/build_candidate_manifest.py" \
  --prompts "$PROMPTS" \
  --output-dir "$MANIFEST_DIR" \
  --base-seed "$BASE_SEED"

python "$ROOT/ops/optimize_chinese_prompts.py" \
  --input "$MANIFEST_DIR/prompt_optimizer_input.jsonl" \
  --output "$MANIFEST_DIR/optimized_prompts.jsonl" \
  --prompt-field prompt_zh \
  --model "${R3D_QWEN_MODEL:-$ROOT/models/Qwen3-8B}"

python "$ROOT/ops/generate_ready3d_v3_sana.py" \
  --candidates "$MANIFEST_DIR/candidates960.jsonl" \
  --optimized-prompts "$MANIFEST_DIR/optimized_prompts.jsonl" \
  --output-root "$RUN_ROOT" \
  --model "${R3D_SANA_MODEL:-$ROOT/models/SANA1.5_1.6B_1024px_diffusers}"
