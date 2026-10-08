#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/r3dguard
REPO="$ROOT/repos/Hunyuan3D-2.1"
LOG="$ROOT/logs/hunyuan21_install.log"
CONSTRAINTS="$ROOT/hunyuan21_constraints.txt"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] Hunyuan3D-2.1 environment installation started"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export MAX_JOBS=8
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx hunyuan21; then
  conda create -y -n hunyuan21 python=3.10 pip \
    --override-channels -c https://repo.anaconda.com/pkgs/main
fi

source "$ROOT/activate_hunyuan21.sh"
python -m pip install --upgrade "pip<26" "setuptools==80.9.0" wheel

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install cython
python -m pip install \
  -r <(sed -e '/^[[:space:]]*bpy==/d' \
           -e '/^[[:space:]]*--extra-index-url/d' \
           -e '/^[[:space:]]*tb_nightly==/d' \
           -e '/^[[:space:]]*basicsr==/d' \
           -e '/^[[:space:]]*realesrgan==/d' \
           -e 's/^ninja==1\.11\.1\.1$/ninja==1.11.1.4/' \
           "$REPO/requirements.txt") \
  -c "$CONSTRAINTS" \
  --no-build-isolation
python -m pip install future lmdb yapf filterpy
python -m pip install facexlib==0.3.0 gfpgan==1.3.8 --no-deps
python -m pip install basicsr==1.4.2 realesrgan==0.3.0 \
  --no-deps --no-build-isolation
if ! python -c 'import bpy' >/dev/null 2>&1; then
  if [[ "${SKIP_BPY:-0}" == "1" ]]; then
    echo "[$(date -Is)] Skipping bpy installation by request"
  else
    BPY_WHEEL_LOCAL="/root/r3dguard/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl"
    if [[ -f "$BPY_WHEEL_LOCAL" ]]; then
      python -m pip install "$BPY_WHEEL_LOCAL"
    else
      python -m pip install \
        https://download.blender.org/pypi/bpy/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl
    fi
  fi
fi

# Blender's official cp310 filename contains an incorrect cp39 tag in WHEEL
# metadata. The binary imports under Python 3.10, but the bad tag makes
# `pip check` report a false platform incompatibility.
BPY_WHEEL_METADATA="$CONDA_PREFIX/lib/python3.10/site-packages/bpy-4.0.0.dist-info/WHEEL"
if [[ -f "$BPY_WHEEL_METADATA" ]] && grep -q '^Tag: cp39-cp39-manylinux_2_28_x86_64$' "$BPY_WHEEL_METADATA"; then
  sed -i 's/^Tag: cp39-cp39-manylinux_2_28_x86_64$/Tag: cp310-cp310-manylinux_2_28_x86_64/' "$BPY_WHEEL_METADATA"
fi

pushd "$REPO/hy3dpaint/custom_rasterizer"
python -m pip install -e . --no-build-isolation
popd

pushd "$REPO/hy3dpaint/DifferentiableRenderer"
bash compile_mesh_painter.sh
popd

mkdir -p "$REPO/hy3dpaint/ckpt"
REALESRGAN="$REPO/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
REALESRGAN_SHA256=4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1
if ! echo "$REALESRGAN_SHA256  $REALESRGAN" | sha256sum -c -; then
  rm -f "$REALESRGAN"
  huggingface-cli download amd/realesrgan-x4plus RealESRGAN_x4plus.pth \
    --local-dir "$REPO/hy3dpaint/ckpt"
  echo "$REALESRGAN_SHA256  $REALESRGAN" | sha256sum -c -
fi

set +e
PIP_CHECK_OUTPUT="$(python -m pip check 2>&1)"
set -e
printf '%s\n' "$PIP_CHECK_OUTPUT"
UNEXPECTED_CHECK_OUTPUT="$(
  printf '%s\n' "$PIP_CHECK_OUTPUT" |
    sed \
      -e '/^basicsr 1\.4\.2 requires tb-nightly, which is not installed\.$/d' \
      -e '/^No broken requirements found\.$/d' \
      -e '/^[[:space:]]*$/d'
)"
if [[ -n "$UNEXPECTED_CHECK_OUTPUT" ]]; then
  echo "Unexpected dependency problems remain:" >&2
  printf '%s\n' "$UNEXPECTED_CHECK_OUTPUT" >&2
  exit 1
fi
echo "[$(date -Is)] Hunyuan3D-2.1 environment installation completed"
