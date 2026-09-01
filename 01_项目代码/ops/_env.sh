#!/usr/bin/env bash
# Shared project root resolver for ReadyRepair3D ops scripts.
# Override with: export R3DGUARD_HOME=/path/to/ReadyRepair3D

if [[ -n "${R3DGUARD_HOME:-}" && -d "${R3DGUARD_HOME}" ]]; then
  ROOT="$(cd "${R3DGUARD_HOME}" && pwd)"
else
  _OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${_OPS_DIR}/.." && pwd)"
fi

export R3DGUARD_HOME="$ROOT"
export R3DGUARD_MODELS="${R3DGUARD_MODELS:-$ROOT/models}"

resolve_calibration() {
  local preferred="$ROOT/config/quality_v2_calibration.json"
  local legacy="$ROOT/data/hunyuan_validation64/auto_quality_v2/calibration.json"
  if [[ -s "$preferred" ]]; then
    echo "$preferred"
  elif [[ -s "$legacy" ]]; then
    echo "$legacy"
  else
    echo "$preferred"
  fi
}

hunyuan_python() {
  if [[ -n "${R3D_HUNYUAN_PYTHON:-}" && -x "${R3D_HUNYUAN_PYTHON}" ]]; then
    echo "${R3D_HUNYUAN_PYTHON}"
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    echo python
  fi
}
