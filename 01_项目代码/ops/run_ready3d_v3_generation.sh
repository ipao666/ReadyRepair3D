#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/_env.sh"

RUN_ROOT="${R3D_V3_RUN_ROOT:-$ROOT/data/ready3d_v3}"
MANIFEST_DIR="$RUN_ROOT/manifests"
PROMPTS="${R3D_V3_PROMPTS:-$ROOT/examples/ready3d_v3/prompts80.jsonl}"
BASE_SEED="${R3D_V3_BASE_SEED:-2026082700}"
OPTIMIZED="$MANIFEST_DIR/optimized_prompts.jsonl"
CANDIDATES="$MANIFEST_DIR/candidates320.jsonl"

mkdir -p "$MANIFEST_DIR" "$RUN_ROOT/logs"

source "$ROOT/activate.sh"
python "$ROOT/ops/build_ready3d_v3_manifest.py" \
  --prompts "$PROMPTS" \
  --output-dir "$MANIFEST_DIR" \
  --base-seed "$BASE_SEED"

python "$ROOT/ops/optimize_chinese_prompts.py" \
  --input "$MANIFEST_DIR/prompt_optimizer_input.jsonl" \
  --output "$OPTIMIZED" \
  --prompt-field prompt_zh \
  --model "${R3D_QWEN_MODEL:-$R3DGUARD_MODELS/Qwen3-8B}"

python "$ROOT/ops/generate_ready3d_v3_sana.py" \
  --candidates "$CANDIDATES" \
  --optimized-prompts "$OPTIMIZED" \
  --output-root "$RUN_ROOT" \
  --model "${R3D_SANA_MODEL:-$R3DGUARD_MODELS/SANA1.5_1.6B_1024px_diffusers}"

source "$ROOT/activate_hunyuan21.sh"
python "$ROOT/ops/run_hunyuan_ready3d.py" \
  --stage all \
  --manifest "$RUN_ROOT/manifest.jsonl" \
  --output-root "$RUN_ROOT/hunyuan" \
  --stop-vram-mib "${R3D_STOP_VRAM_MIB:-38000}"

python "$ROOT/ops/render_hunyuan_outputs.py" \
  --status "$RUN_ROOT/hunyuan/status.jsonl" \
  --output-root "$RUN_ROOT/renders" \
  --expect 320 \
  --workers "${R3D_RENDER_WORKERS:-2}"

python "$ROOT/ops/run_ready3d_v3_server_preflight.py" \
  --manifest "$CANDIDATES" \
  --output-root "$RUN_ROOT" \
  --expect 320 \
  --report "$RUN_ROOT/preflight_after_generation.json"
