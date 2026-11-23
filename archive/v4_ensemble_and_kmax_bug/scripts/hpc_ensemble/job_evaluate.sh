#!/bin/bash
#SBATCH --job-name=fractal_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu                  # CHANGE: match hpc.partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juan.vasconez@yachaytech.edu.ec  # CHANGE: your email

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=================================================="
echo "Ensemble evaluation — Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=================================================="

cd "$PROJECT_DIR"

module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311   # CHANGE: match hpc.conda_env in config.yaml

python scripts/hpc_ensemble/evaluate_ensemble.py \
    --config scripts/hpc_ensemble/config.yaml

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
