#!/usr/bin/env bash
set -Eeuo pipefail

source /root/r3dguard/activate.sh

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/r3dguard/models/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

mkdir -p /root/r3dguard/models /root/r3dguard/logs

echo "[$(date -Is)] Resuming microsoft/TRELLIS.2-4B"
hf download microsoft/TRELLIS.2-4B \
  --local-dir /root/r3dguard/models/TRELLIS.2-4B
echo "[$(date -Is)] TRELLIS.2-4B complete"

echo "[$(date -Is)] Resuming Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
hf download Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
  --local-dir /root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers
echo "[$(date -Is)] SANA1.5 complete"

du -sh /root/r3dguard/models/TRELLIS.2-4B
du -sh /root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers
df -h /
