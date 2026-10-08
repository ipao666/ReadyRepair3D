#!/usr/bin/env bash
# Activate SANA / feature-extraction environment (conda env name: trellis2).
# Kept for historical env name; main 3D backend is Hunyuan3D-2.1.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R3DGUARD_HOME="${R3DGUARD_HOME:-$ROOT}"
export R3DGUARD_MODELS="${R3DGUARD_MODELS:-$R3DGUARD_HOME/models}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; add it to PATH before sourcing activate_sana.sh" >&2
  return 1 2>/dev/null || exit 1
fi
eval "$(conda shell.bash hook)"
conda activate trellis2

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$R3DGUARD_HOME/src:${PYTHONPATH:-}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$R3DGUARD_MODELS/hf-cache}"
