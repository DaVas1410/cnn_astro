#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_pipeline.sh CONFIG.yaml [pipeline.py options...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$1"
shift
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: Configuration file not found: ${CONFIG_PATH}" >&2
    exit 2
fi
CONFIG_PATH="$(cd -- "$(dirname -- "${CONFIG_PATH}")" && pwd)/$(basename -- "${CONFIG_PATH}")"

resolve_python() {
    if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
        printf '%s\n' "${PIPELINE_PYTHON}"
        return
    fi
    if [[ -n "${CONDA_ENV:-}" && -x "${CONDA_ENV}/bin/python" ]]; then
        printf '%s\n' "${CONDA_ENV}/bin/python"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    echo "ERROR: Python not found. Activate the desired environment or set PIPELINE_PYTHON/CONDA_ENV." >&2
    exit 2
}

PYTHON_EXE="$(resolve_python)"
if [[ ! -x "${PYTHON_EXE}" ]]; then
    echo "ERROR: Python executable is not executable: ${PYTHON_EXE}" >&2
    exit 2
fi
export PATH="$(dirname -- "${PYTHON_EXE}"):${PATH}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED=1
umask 002

"${PYTHON_EXE}" - <<'PY_VERSION'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"ERROR: Python 3.10 or newer is required; found {sys.version.split()[0]}")
PY_VERSION

RUN_ROOT="$("${PYTHON_EXE}" - "${SCRIPT_DIR}" "${CONFIG_PATH}" <<'PY_RUNROOT'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pipeline_lib.config import load_and_resolve_config
print(load_and_resolve_config(Path(sys.argv[2]))["paths"]["run_root"])
PY_RUNROOT
)"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/pipeline_state"
LOG_FILE="${RUN_ROOT}/logs/pipeline_launcher.log"
LAUNCHER_PID_FILE="${RUN_ROOT}/pipeline_state/launcher.pid"
PYTHON_PID_FILE="${RUN_ROOT}/pipeline_state/python.pid"
printf '%s\n' "$$" > "${LAUNCHER_PID_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1
START_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
START_SECONDS="$(date +%s)"
CHILD_PID=""

cleanup_pid_files() { rm -f "${LAUNCHER_PID_FILE}" "${PYTHON_PID_FILE}" 2>/dev/null || true; }
forward_signal() {
    local signal_name="$1"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Launcher received ${signal_name}."
    if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" || true
    fi
    cleanup_pid_files
    exit 130
}
trap 'forward_signal SIGINT' INT
trap 'forward_signal SIGTERM' TERM
finish() {
    local status=$?
    local end_seconds
    end_seconds="$(date +%s)"
    cleanup_pid_files
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Pipeline launcher exit=${status} elapsed_seconds=$((end_seconds - START_SECONDS))"
    trap - EXIT
    exit "${status}"
}
trap finish EXIT

echo "============================================================"
echo "Local pipeline launcher PID : $$"
echo "Started UTC                 : ${START_UTC}"
echo "Configuration               : ${CONFIG_PATH}"
echo "Run root                    : ${RUN_ROOT}"
echo "Python                      : ${PYTHON_EXE}"
echo "PIPELINE_GPUS override      : ${PIPELINE_GPUS:-<not set; use YAML>}"
echo "Log                         : ${LOG_FILE}"
echo "============================================================"

"${PYTHON_EXE}" -u "${SCRIPT_DIR}/pipeline.py" --config "${CONFIG_PATH}" "$@" &
CHILD_PID=$!
printf '%s\n' "${CHILD_PID}" > "${PYTHON_PID_FILE}"
echo "Pipeline Python PID         : ${CHILD_PID}"
wait "${CHILD_PID}"
