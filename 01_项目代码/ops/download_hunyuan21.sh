#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
REPO="$ROOT/repos/Hunyuan3D-2.1"
MODEL="$ROOT/models/Hunyuan3D-2.1"
LOG="$ROOT/logs/hunyuan21_download.log"

mkdir -p "$ROOT/repos" "$ROOT/models" "$ROOT/logs"
source "$ROOT/activate.sh"

echo "[$(date -Is)] Hunyuan3D-2.1 download started" | tee -a "$LOG"

if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 \
    https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git \
    "$REPO" 2>&1 | tee -a "$LOG"
else
  echo "[$(date -Is)] Repository already present" | tee -a "$LOG"
fi

hf download tencent/Hunyuan3D-2.1 \
  --local-dir "$MODEL" 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] Hunyuan3D-2.1 download completed" | tee -a "$LOG"
du -sh "$REPO" "$MODEL" | tee -a "$LOG"
