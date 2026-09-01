#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
LOG="$ROOT/logs/pipeline_env_install.log"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] pipeline environment installation started"
source /opt/conda/etc/profile.d/conda.sh
conda activate trellis2

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade "pip<26" wheel setuptools

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install \
  transformers==4.57.3 \
  diffusers==0.39.0 \
  accelerate \
  peft \
  sentencepiece \
  einops \
  kornia \
  timm \
  opencv-python-headless \
  scipy \
  scikit-learn \
  joblib \
  matplotlib \
  trimesh \
  pygltflib \
  pytest

python -m pip check
python - <<'PY'
import cv2
import diffusers
import joblib
import scipy
import sklearn
import torch
import transformers
from diffusers import SanaPipeline

assert torch.cuda.is_available()
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "transformers": transformers.__version__,
    "diffusers": diffusers.__version__,
    "SanaPipeline": SanaPipeline.__name__,
    "opencv": cv2.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "joblib": joblib.__version__,
})
PY

echo "[$(date -Is)] pipeline environment installation completed"
