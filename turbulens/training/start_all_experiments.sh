#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/configs}"

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
SEQUENCE_LOG_DIR="${SEQUENCE_LOG_DIR:-$("${PYTHON_EXE}" - "${SCRIPT_DIR}" "${CONFIG_DIR}/config_multitask.yaml" <<'PY_LOGDIR'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pipeline_lib.config import load_and_resolve_config
cfg = load_and_resolve_config(Path(sys.argv[2]))
run_root = Path(cfg["paths"]["run_root"])
print(run_root.parents[1] / "sequence_logs" / str(cfg["experiment"]["version"]))
PY_LOGDIR
)}"
mkdir -p "${SEQUENCE_LOG_DIR}"
STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
LOG="${SEQUENCE_LOG_DIR}/nohup_sequence_${STAMP}.log"
nohup bash "${SCRIPT_DIR}/run_all_experiments.sh" </dev/null >"${LOG}" 2>&1 &
PID=$!
echo "Started sequential experiment suite in background."
echo "PID: ${PID}"
echo "Log: ${LOG}"
