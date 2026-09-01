#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
CONDA=/opt/conda/bin/conda
LOG_DIR="$ROOT/logs"
MODEL_DIR="$ROOT/models"

mkdir -p "$LOG_DIR" "$MODEL_DIR"

start_download() {
  local name="$1"
  local repo="$2"
  local target="$3"
  local pidfile="$LOG_DIR/${name}_download_restore.pid"
  local logfile="$LOG_DIR/${name}_download_restore.log"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name already running: $(cat "$pidfile")"
    return
  fi

  nohup "$CONDA" run -n trellis2 env \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME="$MODEL_DIR/hf-cache" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    hf download "$repo" --local-dir "$MODEL_DIR/$target" \
    >"$logfile" 2>&1 &
  echo "$!" >"$pidfile"
  echo "$name started: pid=$!, repo=$repo"
}

start_download dinov2 facebook/dinov2-large dinov2-large
start_download depth depth-anything/Depth-Anything-V2-Large-hf Depth-Anything-V2-Large-hf
start_download biref ZhengPeng7/BiRefNet BiRefNet
