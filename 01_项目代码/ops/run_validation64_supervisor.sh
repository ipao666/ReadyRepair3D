#!/usr/bin/env bash
set -euo pipefail

source /root/r3dguard/activate_hunyuan21.sh

ROOT=/root/r3dguard/data/hunyuan_validation64
CONTROL="$ROOT/gpu_control.json"
mkdir -p "$ROOT"
echo $$ > "$ROOT/supervisor.pid"

export R3D_HUNYUAN_SOURCE_MANIFEST=/root/r3dguard/data/validation64/manifest.jsonl
export R3D_HUNYUAN_OUTPUT_ROOT="$ROOT"
export R3D_HUNYUAN_GROUP_IDS=sana_001,sana_008,sana_011,sana_013,sana_016,sana_019,sana_023,sana_024,sana_025,sana_027,sana_034,sana_037,sana_038,sana_040,sana_043,sana_045

python /root/r3dguard/ops/monitor_hunyuan_gpu.py \
  --output "$ROOT/gpu_memory.csv" \
  --control "$CONTROL" \
  --interval 1 \
  > "$ROOT/monitor.log" 2>&1 &
MONITOR_PID=$!
echo "$MONITOR_PID" > "$ROOT/monitor.pid"

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python /root/r3dguard/ops/run_hunyuan_16.py \
  --stage all \
  --limit 64 \
  --stop-vram-mib 38000 \
  --monitor-control "$CONTROL"
