#!/bin/bash
#SBATCH --job-name=fractal_gen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juan.vasconez@yachaytech.edu.ec

# ─── SLURM Job Array for Fractal Dataset Generation ───────────────
#
# Array tasks: 32 (one per k_min value 1..32, k_max=auto/Nyquist)
# Each task generates 1000 images × 3 dimensions (128², 256², 512²)
# then merges into a single grouped HDF5 file.
#
# Submit with:  sbatch scripts/job.sh
# Monitor with: squeue -u $USER
# Cancel with:  scancel <job_id>

echo "================================================"
echo "SLURM Job: ${SLURM_JOB_NAME}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID} of 0-31"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start: $(date)"
echo "================================================"

# ─── Project directory ─────────────────────────────────────────────
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}" || { echo "ERROR: Cannot cd to ${PROJECT_DIR}"; exit 1; }

# Create output directories
mkdir -p data/raw logs

# ─── Activate conda environment ───────────────────────────────────
module load miniconda3/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py39

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# ─── Run generation ───────────────────────────────────────────────
python scripts/run_generation.py

EXIT_CODE=$?

echo "================================================"
echo "End: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================"

exit ${EXIT_CODE}
