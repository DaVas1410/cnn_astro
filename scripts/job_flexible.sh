#!/bin/bash
#SBATCH --job-name=fractal_flexible
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juan.vasconez@yachaytech.edu.ec

# ─── SLURM Job for Flexible Fractal Dataset Generation ────────────────────────
#
# Generates a diverse dataset by sampling k_min, k_max, sigma uniformly from ranges.
#
# Feature:
#   - Each image has UNIQUE random parameters (k_min, k_max, sigma)
#   - High parameter diversity -> diverse training data
#   - Single job (no job arrays needed unless you want multiple resolution/parameter runs)
#
# Usage:
#   # With defaults:
#   sbatch scripts/job_flexible.sh
#
#   # With custom ranges:
#   sbatch --export=KMIN_LOW=1,KMIN_HIGH=64 scripts/job_flexible.sh
#
#   Submit with:   sbatch scripts/job_flexible.sh
#   Monitor with:  squeue -u $USER
#   Cancel with:   scancel <job_id>

echo "=================================================="
echo "SLURM Flexible Dataset Generation Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start: $(date)"
echo "=================================================="

# ─── Project Directory ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}" || { echo "ERROR: Cannot cd to ${PROJECT_DIR}"; exit 1; }

# Create output directories
mkdir -p data/raw logs

# ─── Activate Conda Environment ───────────────────────────────────────────
module load miniconda3/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py39

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# ─── Parameter Configuration ──────────────────────────────────────────────
# All parameters can be overridden via environment variables at submission time.
# Example: sbatch --export=KMIN_LOW=2,KMIN_HIGH=64 scripts/job_flexible.sh

# Image dimensions
export IMAGE_WIDTH=${IMAGE_WIDTH:-512}
export IMAGE_HEIGHT=${IMAGE_HEIGHT:-512}
export IMAGE_DEPTH=${IMAGE_DEPTH:-1}

# Total images and batch configuration
export NUM_IMAGES=${NUM_IMAGES:-10000}
export BATCH_SIZE=${BATCH_SIZE:-50}

# K-min range (continuous uniform sampling)
export KMIN_LOW=${KMIN_LOW:-1.0}
export KMIN_HIGH=${KMIN_HIGH:-32.0}

# K-max configuration: either 'auto' (Nyquist) or range
export KMAX_MODE=${KMAX_MODE:-auto}      # 'auto' or 'range'
export KMAX_LOW=${KMAX_LOW:-100.0}
export KMAX_HIGH=${KMAX_HIGH:-256.0}

# Sigma range (log-normal distribution parameter)
export SIGMA_LOW=${SIGMA_LOW:-2.0}
export SIGMA_HIGH=${SIGMA_HIGH:-2.5}

# Mean range (log-normal distribution mean)
export MEAN_LOW=${MEAN_LOW:-1.0}
export MEAN_HIGH=${MEAN_HIGH:-1.0}

# Power spectrum slope (fixed, typically -5/3 for Kolmogorov turbulence)
export BETA=${BETA:--1.66666667}

# Output file (if not specified, auto-generated based on parameters)
export OUTPUT_FILE=${OUTPUT_FILE:-""}

# Parallel processing (use SLURM_CPUS_PER_TASK by default)
export NUM_WORKERS=${SLURM_CPUS_PER_TASK}

# ─── Print Configuration ──────────────────────────────────────────────────
echo ""
echo "Parameter Configuration:"
echo "  Image dimensions: ${IMAGE_WIDTH} x ${IMAGE_HEIGHT} x ${IMAGE_DEPTH}"
echo "  Total images: ${NUM_IMAGES}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  K-min range: [${KMIN_LOW}, ${KMIN_HIGH}]"
echo "  K-max mode: ${KMAX_MODE}"
if [ "${KMAX_MODE}" = "range" ]; then
  echo "  K-max range: [${KMAX_LOW}, ${KMAX_HIGH}]"
fi
echo "  Sigma range: [${SIGMA_LOW}, ${SIGMA_HIGH}]"
echo "  Mean range: [${MEAN_LOW}, ${MEAN_HIGH}]"
echo "  Beta: ${BETA}"
echo "  Workers: ${NUM_WORKERS}"
echo ""

# ─── Run Generation ────────────────────────────────────────────────────────
python scripts/run_generation_flexible.py

EXIT_CODE=$?

echo "=================================================="
echo "End: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=================================================="

exit ${EXIT_CODE}
