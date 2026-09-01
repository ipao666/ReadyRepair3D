#!/usr/bin/env bash

source /opt/conda/etc/profile.d/conda.sh
conda activate trellis2

export R3DGUARD_HOME=/root/r3dguard
export R3DGUARD_MODELS=/root/r3dguard/models
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$R3DGUARD_HOME/repos/TRELLIS.2${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST=8.0

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$R3DGUARD_MODELS/hf-cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
