#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "[1/5] Installing system dependencies"
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl ffmpeg git git-lfs \
  libegl1 libgl1 libglib2.0-0 libjpeg-dev ninja-build sudo wget
git lfs install --system

echo "[2/5] Installing Miniforge when absent"
if [[ ! -x /opt/conda/bin/conda ]]; then
  installer=/tmp/Miniforge3-Linux-x86_64.sh
  curl -fL --retry 3 \
    https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
    -o "$installer"
  bash "$installer" -b -p /opt/conda
  rm -f "$installer"
fi
/opt/conda/bin/conda config --system --set auto_activate_base false

echo "[3/5] Creating project directories"
install -d \
  /root/r3dguard/repos \
  /root/r3dguard/models \
  /root/r3dguard/data \
  /root/r3dguard/outputs \
  /root/r3dguard/logs

echo "[4/5] Creating Python 3.10 environment"
if ! /opt/conda/bin/conda env list | awk '{print $1}' | grep -qx trellis2; then
  /opt/conda/bin/conda create -y -n trellis2 python=3.10 pip
fi

echo "[5/5] Writing activation helper"
printf '%s\n' \
  'source /opt/conda/etc/profile.d/conda.sh' \
  'conda activate trellis2' \
  'export CUDA_HOME=/usr/local/cuda-12.4' \
  'export PATH="$CUDA_HOME/bin:$PATH"' \
  'export R3DGUARD_HOME=/root/r3dguard' \
  > /root/r3dguard/activate.sh
chmod 0644 /root/r3dguard/activate.sh

echo "Bootstrap complete"
/opt/conda/bin/conda --version
/opt/conda/bin/conda run -n trellis2 python --version
