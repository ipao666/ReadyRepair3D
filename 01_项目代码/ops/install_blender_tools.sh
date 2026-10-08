#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
LOG="$ROOT/logs/blender_tools_install.log"
mkdir -p "$ROOT/logs"

echo "[$(date -Is)] Blender and mesh tools installation started" | tee -a "$LOG"
export DEBIAN_FRONTEND=noninteractive
apt-get update 2>&1 | tee -a "$LOG"
apt-get install -y --no-install-recommends blender 2>&1 | tee -a "$LOG"
echo "[$(date -Is)] Blender installation completed" | tee -a "$LOG"
blender --version | head -n 2 | tee -a "$LOG"
