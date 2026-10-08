#!/usr/bin/env bash
set -euo pipefail

source /root/r3dguard/activate_hunyuan21.sh

VALIDATION=/root/r3dguard/data/hunyuan_validation64/auto_quality_v2
SPLITS=/root/r3dguard/data/ready3d_splits
FINAL=/root/r3dguard/data/ready3d_labels/hunyuan_v2
mkdir -p "$FINAL"
echo $$ > "$FINAL/supervisor.pid"

while [[ ! -f "$VALIDATION/FROZEN" ]]; do
  sleep 60
done

run_split() {
  local split="$1"
  local manifest="$2"
  local count="$3"
  local root="/root/r3dguard/data/hunyuan_ready3d_${split}"
  local control="$root/gpu_control.json"
  local groups
  groups=$(/opt/conda/envs/hunyuan21/bin/python - "$manifest" <<'PY'
import json, sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x]
print(','.join(sorted({row['group_id'] for row in rows})))
PY
)
  mkdir -p "$root"
  export R3D_HUNYUAN_SOURCE_MANIFEST="$manifest"
  export R3D_HUNYUAN_OUTPUT_ROOT="$root"
  export R3D_HUNYUAN_GROUP_IDS="$groups"

  python /root/r3dguard/ops/monitor_hunyuan_gpu.py \
    --output "$root/gpu_memory.csv" --control "$control" --interval 1 \
    > "$root/monitor.log" 2>&1 &
  local monitor_pid=$!
  python /root/r3dguard/ops/run_hunyuan_16.py \
    --stage all --limit "$count" --stop-vram-mib 38000 --monitor-control "$control"
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  R3D_HUNYUAN_ROOT="$root" \
  R3D_TRELLIS_STATUS="$root/no_trellis_status.jsonl" \
  python /root/r3dguard/ops/verify_hunyuan_16.py --expect "$count"

  python /root/r3dguard/ops/score_hunyuan_outputs.py \
    --status "$root/status.jsonl" \
    --render-root "$root/renders" \
    --output-dir "$root/auto_quality_v2" \
    --dino-model /root/r3dguard/models/dinov2-large \
    --birefnet-model /root/r3dguard/models/BiRefNet \
    --expect "$count" \
    --calibration-in "$VALIDATION/calibration.json" \
    --refresh
}

run_split train "$SPLITS/train96.jsonl" 96
run_split test "$SPLITS/test24.jsonl" 24

/opt/conda/envs/hunyuan21/bin/python - <<'PY'
import json
from pathlib import Path

final=Path('/root/r3dguard/data/ready3d_labels/hunyuan_v2')
sources={
    'train': Path('/root/r3dguard/data/hunyuan_ready3d_train/auto_quality_v2/quality_labels.jsonl'),
    'test': Path('/root/r3dguard/data/hunyuan_ready3d_test/auto_quality_v2/quality_labels.jsonl'),
}
combined=[]
for split,path in sources.items():
    rows=[json.loads(x) for x in path.read_text().splitlines() if x]
    for row in rows:
        row['split']=split
        combined.append(row)
with (final/'quality_labels.jsonl').open('w',encoding='utf-8') as f:
    for row in combined:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
versions={row['scoring_version'] for row in combined}
if versions != {'ready3d-v2'}:
    raise SystemExit(f'unexpected scoring versions: {sorted(versions)}')
summary={
    'scoring_version':versions.pop(),
    'calibration_source':'independent_validation_64',
    'train_samples':sum(r['split']=='train' for r in combined),
    'test_samples':sum(r['split']=='test' for r in combined),
    'technically_valid':sum(r['technically_valid'] for r in combined),
    'high_quality':sum(r['high_quality'] for r in combined),
}
(final/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(final/'COMPLETE').write_text('ready3d-hunyuan-v2\n',encoding='utf-8')
print(summary)
PY
