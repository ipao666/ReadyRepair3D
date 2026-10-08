#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/../.." && pwd)"
STAGE01="${STAGE01_FREEZE:-/root/readyrepair3d_runs/stages/stage01_ready3d_v3}"
RUN_ROOT="${SANA_LORA_RUN_ROOT:-/root/readyrepair3d_runs/stages/stage02_sana_candidates}"

test -f "$STAGE01/READY"
(cd "$STAGE01" && sha256sum -c SHA256SUMS)
test ! -f "$RUN_ROOT/READY"
mkdir -p "$RUN_ROOT/logs"

export SANA_LORA_RUN_ROOT="$RUN_ROOT"
bash "$OPS_DIR/run_generation.sh"

# The generation entrypoint runs in a child shell, so activate the scoring
# environment again in this parent shell before invoking Qwen3-VL.
source "$ROOT/activate.sh"
python "$OPS_DIR/score_prompt_adherence.py" \
  --manifest "$RUN_ROOT/manifest.jsonl" \
  --output "$RUN_ROOT/adherence_scores.jsonl" \
  --model "${R3D_QWEN_VL_MODEL:-$ROOT/models/Qwen3-VL-8B-Instruct}"

python "$OPS_DIR/summarize_stage02.py" \
  --manifest "$RUN_ROOT/manifest.jsonl" \
  --adherence "$RUN_ROOT/adherence_scores.jsonl" \
  --output "$RUN_ROOT/summary.json"
