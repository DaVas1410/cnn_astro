#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/configs}"
CURRENT_CHILD=""
START_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
START_SECONDS="$(date +%s)"

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
export PATH="$(dirname -- "${PYTHON_EXE}"):${PATH}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED=1

# Put orchestration logs next to scientific results, not inside the source tree.
if [[ -z "${SEQUENCE_LOG_DIR:-}" ]]; then
    SEQUENCE_LOG_DIR="$(${PYTHON_EXE} - "${SCRIPT_DIR}" "${CONFIG_DIR}/config_multitask.yaml" <<'PY_LOGDIR'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pipeline_lib.config import load_and_resolve_config
cfg = load_and_resolve_config(Path(sys.argv[2]))
run_root = Path(cfg["paths"]["run_root"])
output_root = run_root.parents[1]
print(output_root / "sequence_logs" / str(cfg["experiment"]["version"]))
PY_LOGDIR
)"
fi
mkdir -p "${SEQUENCE_LOG_DIR}"

finish_sequence() {
    local status=$?
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Sequence exit=${status} elapsed_seconds=$(( $(date +%s) - START_SECONDS ))"
    trap - EXIT
    exit "${status}"
}
trap finish_sequence EXIT

forward_signal() {
    local signal_name="$1"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Sequence received ${signal_name}." >&2
    if [[ -n "${CURRENT_CHILD}" ]] && kill -0 "${CURRENT_CHILD}" 2>/dev/null; then
        kill -TERM "${CURRENT_CHILD}" 2>/dev/null || true
        wait "${CURRENT_CHILD}" || true
    fi
    exit 130
}
trap 'forward_signal SIGINT' INT
trap 'forward_signal SIGTERM' TERM

run_experiment() {
    local label="$1"
    local config="$2"
    local log
    local status
    log="${SEQUENCE_LOG_DIR}/${label}.log"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] START ${label}: ${config}"
    bash "${SCRIPT_DIR}/run_pipeline.sh" "${config}" >"${log}" 2>&1 &
    CURRENT_CHILD=$!
    set +e
    wait "${CURRENT_CHILD}"
    status=$?
    set -e
    CURRENT_CHILD=""
    if [[ ${status} -ne 0 ]]; then
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] FAILED ${label}; exit=${status}; log=${log}" >&2
        exit "${status}"
    fi
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] END ${label}; log=${log}"
}

echo "Sequence launcher PID: $$"
echo "Started UTC: ${START_UTC}"
echo "Python: ${PYTHON_EXE}"
echo "Logs: ${SEQUENCE_LOG_DIR}"

run_experiment multitask "${CONFIG_DIR}/config_multitask.yaml"
run_experiment single_k_min "${CONFIG_DIR}/config_single_k_min.yaml"
run_experiment single_k_max "${CONFIG_DIR}/config_single_k_max.yaml"
run_experiment single_sigma "${CONFIG_DIR}/config_single_sigma.yaml"
run_experiment single_beta "${CONFIG_DIR}/config_single_beta.yaml"

COMPARISON_LOG="${SEQUENCE_LOG_DIR}/comparison.log"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] START final comparison"
"${PYTHON_EXE}" -u "${SCRIPT_DIR}/compare_multitask_single.py" \
    --config "${CONFIG_DIR}/comparison.yaml" --overwrite >"${COMPARISON_LOG}" 2>&1 &
CURRENT_CHILD=$!
set +e
wait "${CURRENT_CHILD}"
status=$?
set -e
CURRENT_CHILD=""
if [[ ${status} -ne 0 ]]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] FAILED final comparison; exit=${status}; log=${COMPARISON_LOG}" >&2
    exit "${status}"
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] END final comparison; log=${COMPARISON_LOG}"
echo "All experiments completed in $(( $(date +%s) - START_SECONDS )) seconds."
