#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash status_pipeline.sh CONFIG.yaml" >&2
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
echo "Run root: ${RUN_ROOT}"
for f in "${RUN_ROOT}/pipeline_state/launcher.pid" "${RUN_ROOT}/pipeline_state/python.pid"; do
  if [[ -f "$f" ]]; then
    pid="$(tr -d '[:space:]' < "$f")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo "$(basename "$f"): $pid (running)"; else echo "$(basename "$f"): $pid (not running/stale)"; fi
  fi
done
if [[ -f "${RUN_ROOT}/pipeline_state/pipeline.json" ]]; then
    echo "--- pipeline.json ---"
    cat "${RUN_ROOT}/pipeline_state/pipeline.json"
fi
if [[ -f "${RUN_ROOT}/logs/pipeline_launcher.log" ]]; then
    echo "--- last 20 log lines ---"
    tail -n 20 "${RUN_ROOT}/logs/pipeline_launcher.log"
fi
