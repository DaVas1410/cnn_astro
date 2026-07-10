#!/bin/bash
# run.sh — train the full ensemble + evaluate on a single Linux PC with a GPU (A100, etc.).
# Sequential (one member at a time on one GPU) — the SLURM array's job, minus SLURM.
# Usage: bash scripts/local_run/run.sh
#
# Optional env vars:
#   CONFIG=/path/to/config.yaml   override the config (default: hpc_ensemble/config.yaml)
#   CUDA_VISIBLE_DEVICES=0         pick which GPU to use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ENSEMBLE_DIR="$PROJECT_DIR/scripts/hpc_ensemble"
VENV_DIR="$PROJECT_DIR/.venv"
SRC_CONFIG="${CONFIG:-$ENSEMBLE_DIR/config.yaml}"

cd "$PROJECT_DIR"
mkdir -p logs

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR — run 'bash scripts/local_run/setup.sh' first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Generate a run-local config with project_dir pointed at THIS machine's project root,
# without touching the tracked config.yaml. (Paths in the config resolve via hpc.project_dir.)
RUN_CONFIG="$PROJECT_DIR/logs/config_local.yaml"
sed "s|^\( *project_dir:\).*|\1 \"$PROJECT_DIR\"|" "$SRC_CONFIG" > "$RUN_CONFIG"

N_MEMBERS=$(python -c "import yaml,sys; print(yaml.safe_load(open('$RUN_CONFIG'))['ensemble']['n_members'])")

echo "=================================================="
echo "Local ensemble run"
echo "Project   : $PROJECT_DIR"
echo "Config    : $SRC_CONFIG  ->  $RUN_CONFIG"
echo "Members   : $N_MEMBERS"
echo "Device    : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no CUDA!)")')"
echo "Start     : $(date)"
echo "=================================================="

for (( mid=0; mid<N_MEMBERS; mid++ )); do
    echo ""
    echo "----- Training member $mid / $((N_MEMBERS-1)) -----"
    python "$ENSEMBLE_DIR/train_member.py" \
        --config "$RUN_CONFIG" \
        --member-id "$mid" \
        2>&1 | tee "logs/member_${mid}.log"
done

echo ""
echo "----- Evaluating ensemble -----"
python "$ENSEMBLE_DIR/evaluate_ensemble.py" \
    --config "$RUN_CONFIG" \
    2>&1 | tee "logs/evaluate.log"

echo ""
echo "=================================================="
echo "Done: $(date)"
echo "Results in: outputs/hpc_ensemble/"
echo "=================================================="
