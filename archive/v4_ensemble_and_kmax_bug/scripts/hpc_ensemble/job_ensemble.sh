#!/bin/bash
#SBATCH --job-name=fractal_ensemble
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu                  # CHANGE: set in config.yaml hpc.partition
#SBATCH --gres=gpu:1                     # CHANGE: match hpc.gpus_per_task
#SBATCH --cpus-per-task=8               # CHANGE: match hpc.cpus_per_task
#SBATCH --mem=32G                        # CHANGE: match hpc.mem_gb
#SBATCH --time=06:00:00                  # CHANGE: match hpc.time_limit
#SBATCH --output=logs/ensemble_%A_%a.out
#SBATCH --error=logs/ensemble_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=juan.vasconez@yachaytech.edu.ec  # CHANGE: your email

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=================================================="
echo "Ensemble member: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=================================================="

cd "$PROJECT_DIR"
# logs/ only holds SLURM stdout/stderr and is created in the *submit* directory
# at submission time. The repo root may be read-only on a shared HPC install,
# so don't let a failed mkdir here kill the job.
mkdir -p logs 2>/dev/null || true

# Load conda — adapt module name to your HPC
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311   # CHANGE: match hpc.conda_env in config.yaml

echo "Python: $(which python)"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

python scripts/hpc_ensemble/train_member.py \
    --config scripts/hpc_ensemble/config.yaml \
    --member-id "${SLURM_ARRAY_TASK_ID}"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
