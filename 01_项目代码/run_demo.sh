#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R3DGUARD_HOME="$ROOT"

PROMPT=${1:-"一个蓝白陶瓷茶壶，完整主体，干净背景"}
OUTPUT_DIR=${2:-"$ROOT/outputs/demo"}
SEED=${3:-20260722}

bash "$ROOT/ops/run_full_quality_pipeline.sh" \
  --prompt "$PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED"
