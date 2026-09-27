#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash stop_pipeline.sh CONFIG.yaml" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$1"
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
        echo "ERROR: Python not found. Activate the desired environment or set PIPELINE_PYTHON/CONDA_ENV." >&2
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
PID_FILE="${RUN_ROOT}/pipeline_state/launcher.pid"
if [[ ! -f "${PID_FILE}" ]]; then
    echo "No launcher PID file found at ${PID_FILE}. The pipeline may not be running."
    exit 0
fi
PID="$(tr -d '[:space:]' < "${PID_FILE}")"
if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
    echo "Stale PID file (${PID:-empty}); removing it."
    rm -f "${PID_FILE}"
    exit 0
fi
echo "Sending SIGTERM to pipeline launcher PID ${PID}..."
kill -TERM "${PID}"
echo "Signal sent. Resumable checkpoints are retained."
