#!/usr/bin/env bash
# Activate Hunyuan3D-2.1 + scoring environment (conda env name: hunyuan21).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R3DGUARD_HOME="${R3DGUARD_HOME:-$ROOT}"
export R3DGUARD_MODELS="${R3DGUARD_MODELS:-$R3DGUARD_HOME/models}"
export HUNYUAN21_REPO="${HUNYUAN21_REPO:-$R3DGUARD_HOME/repos/Hunyuan3D-2.1}"
export HUNYUAN21_MODEL="${HUNYUAN21_MODEL:-$R3DGUARD_MODELS/Hunyuan3D-2.1}"
export HUNYUAN_DINO_MODEL="${HUNYUAN_DINO_MODEL:-$R3DGUARD_MODELS/dinov2-giant}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; add it to PATH before sourcing activate_hunyuan21.sh" >&2
  return 1 2>/dev/null || exit 1
fi
eval "$(conda shell.bash hook)"
conda activate hunyuan21

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$HUNYUAN21_REPO/hy3dshape:$HUNYUAN21_REPO/hy3dpaint:$R3DGUARD_HOME/src:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$R3DGUARD_MODELS/hf-cache}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
