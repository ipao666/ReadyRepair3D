#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard/data/hunyuan_validation64
SUPERVISOR_PID=$(cat "$ROOT/supervisor.pid")
echo $$ > "$ROOT/finalizer.pid"

while kill -0 "$SUPERVISOR_PID" 2>/dev/null; do
  sleep 30
done

source /root/r3dguard/activate_hunyuan21.sh

python - <<'PY'
import json
from pathlib import Path
path = Path('/root/r3dguard/data/hunyuan_validation64/status.jsonl')
rows = [json.loads(line) for line in path.read_text().splitlines() if line]
if len(rows) != 64:
    raise SystemExit(f'expected 64 status rows, found {len(rows)}')
shape = sum(row.get('shape', {}).get('status') == 'success' for row in rows)
paint = sum(row.get('paint', {}).get('status') == 'success' for row in rows)
if shape != 64 or paint != 64:
    raise SystemExit(f'generation incomplete: shape={shape}, paint={paint}')
print({'generation_verified': True, 'shape': shape, 'paint': paint})
PY

R3D_HUNYUAN_ROOT="$ROOT" \
R3D_TRELLIS_STATUS=/root/r3dguard/data/validation64/no_trellis_status.jsonl \
python /root/r3dguard/ops/verify_hunyuan_16.py --expect 64

python /root/r3dguard/ops/score_hunyuan_outputs.py \
  --status "$ROOT/status.jsonl" \
  --render-root "$ROOT/renders" \
  --output-dir "$ROOT/auto_quality_v2" \
  --dino-model /root/r3dguard/models/dinov2-large \
  --birefnet-model /root/r3dguard/models/BiRefNet \
  --expect 64 \
  --fit-source independent_validation_64 \
  --refresh

python - <<'PY'
import json
from pathlib import Path
root = Path('/root/r3dguard/data/hunyuan_validation64/auto_quality_v2')
rows = [json.loads(line) for line in (root / 'quality_labels.jsonl').read_text().splitlines() if line]
calibration = json.loads((root / 'calibration.json').read_text())
assert len(rows) == 64
assert calibration['source'] == 'independent_validation_64'
assert all(row['high_quality'] is not None for row in rows)
(root / 'FROZEN').write_text('ready3d-v2\n', encoding='utf-8')
print({'calibration_frozen': True, 'samples': len(rows), 'threshold': calibration['quality_threshold']})
PY
