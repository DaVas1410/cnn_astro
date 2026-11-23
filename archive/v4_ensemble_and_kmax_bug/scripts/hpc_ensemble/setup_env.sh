#!/bin/bash
# setup_env.sh — run once on HPC before submitting jobs.
# Creates a conda env named as specified in config.yaml.
# Usage: bash setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="py311"        # CHANGE: match hpc.conda_env in config.yaml

echo "=== Setting up environment: $ENV_NAME ==="

# Load conda (adapt module name to your HPC)
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists — updating packages"
    conda activate "$ENV_NAME"
else
    echo "Creating environment $ENV_NAME with Python 3.11"
    conda create -y -n "$ENV_NAME" python=3.11
    conda activate "$ENV_NAME"
fi

pip install -r "$SCRIPT_DIR/requirements.txt"

echo "=== Setup complete ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
