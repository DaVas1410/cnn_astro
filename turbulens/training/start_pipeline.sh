#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash start_pipeline.sh CONFIG.yaml [pipeline.py options...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$1"
shift

resolve_python() {
    if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
        printf '%s\n' "${PIPELINE_PYTHON}"
    elif [[ -n "${CONDA_ENV:-}" && -x "${CONDA_ENV}/bin/python" ]]; then
        printf '%s\n' "${CONDA_ENV}/bin/python"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        echo "ERROR: Python not found. Activate the environment or set PIPELINE_PYTHON/CONDA_ENV." >&2
        exit 2
    fi
}

PYTHON_EXE="$(resolve_python)"
RUN_ROOT="$("${PYTHON_EXE}" - "${SCRIPT_DIR}" "${CONFIG_PATH}" <<'PY_RUNROOT'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pipeline_lib.config import load_and_resolve_config
print(load_and_resolve_config(Path(sys.argv[2]))["paths"]["run_root"])
PY_RUNROOT
)"
mkdir -p "${RUN_ROOT}/logs"
STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
BOOT_LOG="${RUN_ROOT}/logs/nohup_bootstrap_${STAMP}.log"
nohup bash "${SCRIPT_DIR}/run_pipeline.sh" "${CONFIG_PATH}" "$@" </dev/null >"${BOOT_LOG}" 2>&1 &
PID=$!
echo "Started local pipeline in background."
echo "Launcher PID: ${PID}"
echo "Bootstrap log: ${BOOT_LOG}"
echo "Canonical log: ${RUN_ROOT}/logs/pipeline_launcher.log"
