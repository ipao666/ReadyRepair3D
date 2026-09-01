#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
HUNYUAN_PYTHON="$(hunyuan_python)"
PROMPT=""
OUTPUT_DIR=""
SEED=20260717
QUALITY_THRESHOLD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --quality-threshold) QUALITY_THRESHOLD="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROMPT" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --prompt TEXT --output-dir DIR [--seed N] [--quality-threshold FLOAT]" >&2
  exit 2
fi

if [[ -z "$QUALITY_THRESHOLD" ]]; then
  CALIBRATION="$(resolve_calibration)"
  QUALITY_THRESHOLD=$(
    "$HUNYUAN_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["quality_threshold"])' \
      "$CALIBRATION"
  )
fi

OUTPUT_DIR=$(
  "$HUNYUAN_PYTHON" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "$OUTPUT_DIR"
)
ROUND1="$OUTPUT_DIR/round_1"
ROUND2="$OUTPUT_DIR/round_2"
STARTED_AT=$(date +%s)
mkdir -p "$OUTPUT_DIR"

"$ROOT/ops/run_repaired_quality_round.sh" \
  --prompt "$PROMPT" \
  --output-dir "$ROUND1" \
  --seed "$SEED" \
  --quality-threshold "$QUALITY_THRESHOLD"

ROUND1_REPORT="$ROUND1/round_final/round_report.json"
ROUND1_MEETS=$(
  "$HUNYUAN_PYTHON" -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1]))["meets_quality_threshold"] else "false")' \
    "$ROUND1_REPORT"
)

REPORTS=("$ROUND1_REPORT")
SECOND_ROUND_EXECUTED=false
SECOND_ROUND_TRIGGER_REASON=null
if [[ "$ROUND1_MEETS" != "true" ]]; then
  ROUND2_SEED=$((SEED + 1000))
  SECOND_ROUND_EXECUTED=true
  SECOND_ROUND_TRIGGER_REASON=round1_below_threshold_or_invalid
  "$ROOT/ops/run_repaired_quality_round.sh" \
    --prompt "$PROMPT" \
    --output-dir "$ROUND2" \
    --seed "$ROUND2_SEED" \
    --quality-threshold "$QUALITY_THRESHOLD"
  REPORTS+=("$ROUND2/round_final/round_report.json")
fi

source "$ROOT/activate_hunyuan21.sh"
python "$ROOT/ops/select_best_round.py" \
  --round-report "${REPORTS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --threshold "$QUALITY_THRESHOLD"

[[ -s "$OUTPUT_DIR/final.glb" ]]
[[ -s "$OUTPUT_DIR/final_report.json" ]]

"$HUNYUAN_PYTHON" - \
  "$PROMPT" "$SEED" "$QUALITY_THRESHOLD" "$OUTPUT_DIR" "$STARTED_AT" \
  "$SECOND_ROUND_EXECUTED" "$SECOND_ROUND_TRIGGER_REASON" <<'PY'
import json
import sys
import time
from pathlib import Path

prompt = sys.argv[1]
seed = int(sys.argv[2])
threshold = float(sys.argv[3])
output = Path(sys.argv[4])
started = int(sys.argv[5])
second_executed = sys.argv[6].lower() == "true"
trigger_reason = None if sys.argv[7] == "null" else sys.argv[7]
final = json.loads((output / "final_report.json").read_text())
round_reports = [
    json.loads(path.read_text())
    for path in (
        output / "round_1/round_final/round_report.json",
        output / "round_2/round_final/round_report.json",
    )
    if path.is_file()
]
summary = {
    "schema_version": "r3dguard.full-quality-pipeline.v1",
    "prompt": prompt,
    "base_seed": seed,
    "quality_threshold": threshold,
    "round_count": len(round_reports),
    "second_round_executed": second_executed,
    "second_round_trigger_reason": trigger_reason,
    "rounds": [
        {
            "round_index": index,
            "seed": seed + (index - 1) * 1000,
            "quality_score_v2": report["final_quality_score_v2"],
            "technically_valid": report["final_technically_valid"],
            "meets_quality_threshold": report["meets_quality_threshold"],
            "repair3d_action": report["repair3d_action"],
            "repair3d_accepted": report["repair3d_accepted"],
            "v2_repair_accepted": report["v2_repair_accepted"],
        }
        for index, report in enumerate(round_reports, 1)
    ],
    "selected_round": final["selected_round"],
    "final_quality_score_v2": final["final_quality_score_v2"],
    "final_technically_valid": final["final_technically_valid"],
    "meets_quality_threshold": final["meets_quality_threshold"],
    "low_confidence": final["low_confidence"],
    "final_glb": str((output / "final.glb").resolve()),
    "elapsed_seconds": time.time() - started,
}
(output / "pipeline_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False), flush=True)
PY

touch "$OUTPUT_DIR/PIPELINE_COMPLETE"
