#!/usr/bin/env bash
set -euo pipefail

ROOT="${R3DGUARD_ROOT:-/root/r3dguard}"
TARGET="${QWEN3_MODEL_DIR:-${ROOT}/models/Qwen3-8B}"
REPO="Qwen/Qwen3-8B"
MIN_FREE_KIB=$((30 * 1024 * 1024))

mkdir -p "${TARGET}"
FREE_KIB="$(df -Pk "${TARGET}" | awk 'NR==2 {print $4}')"
if [[ -z "${FREE_KIB}" || "${FREE_KIB}" -lt "${MIN_FREE_KIB}" ]]; then
  echo "ERROR: at least 30 GiB free disk space is required; available KiB=${FREE_KIB:-unknown}" >&2
  exit 2
fi

if [[ -f "${ROOT}/activate.sh" ]]; then
  source "${ROOT}/activate.sh" >/dev/null 2>&1
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: Hugging Face CLI 'hf' was not found after environment activation" >&2
  exit 3
fi

export HF_HOME="${HF_HOME:-${ROOT}/models/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "Downloading ${REPO} to ${TARGET} (resume is automatic)"
hf download "${REPO}" --local-dir "${TARGET}"
echo "QWEN3_DOWNLOAD_OK target=${TARGET}"
