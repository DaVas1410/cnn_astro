#!/bin/bash
# submit.sh — professor runs once to launch full ensemble pipeline.
# Usage: bash scripts/hpc_ensemble/submit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"
mkdir -p logs

echo "Submitting ensemble training (5 members as SLURM array)..."
ARRAY_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/job_ensemble.sh")
echo "  Ensemble job ID: $ARRAY_JOB_ID  (members 0–4)"

echo "Submitting evaluation (will run after all 5 members succeed)..."
EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB_ID "$SCRIPT_DIR/job_evaluate.sh")
echo "  Evaluation job ID: $EVAL_JOB_ID"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel $ARRAY_JOB_ID $EVAL_JOB_ID"
echo ""
echo "Results will be written to: outputs/hpc_ensemble/"
