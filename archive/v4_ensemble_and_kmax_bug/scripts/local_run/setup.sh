#!/bin/bash
# setup.sh — one-time dependency install for a single Linux PC with a GPU (e.g. A100).
# No conda, no SLURM. Creates a plain python venv at the project root and installs deps.
# Usage: bash scripts/local_run/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_DIR="$PROJECT_DIR/.venv"
PY="${PYTHON:-python3}"

echo "=== Local environment setup ==="
echo "Project : $PROJECT_DIR"
echo "Python  : $($PY --version)"
echo "Venv    : $VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    "$PY" -m venv "$VENV_DIR"
else
    echo "Venv already exists — reusing it."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Reuse the same dependency list as the HPC ensemble.
# NOTE: the default PyPI torch wheels on Linux are CUDA builds and work on an A100.
# If you need a specific CUDA version, install torch first, e.g.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r "$SCRIPT_DIR/../hpc_ensemble/requirements.txt"

echo ""
echo "=== Setup complete ==="
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}' + (f' | {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''))"
echo ""
echo "Next: bash scripts/local_run/run.sh"
