#!/usr/bin/env bash
# Second independent replica of the large disjoint-sampling splits: train/val/test
# as separate HDF5 files, each with its own MASTER_SEED so parameter draws and
# pyFC seeds never overlap between splits -- nor with the first large set
# (MASTER_SEED 1000/2000/3000) or the small set (4000/5000/6000).
#
#   train 100000  MASTER_SEED=7000
#   val    20000  MASTER_SEED=8000
#   test   50000  MASTER_SEED=9000
#
# Same distribution as the chosen training config: k_min~U[1,32],
# k_max~U[34,64], sigma~U[0.01,5], beta~U[-3,-1], mean=1.0, 128x128.
set -euo pipefail
cd "$(dirname "$0")/.."

# NOTE (2026-08-07): `uv run --with ...` no longer works on this Windows box --
# the ephemeral interpreter uv materialises in its cache is blocked by an
# Application Control policy ("Failed to spawn: `python`", os error 4551). A
# venv created in the project tree is allowed, so we use a persistent one.
# Recreate it with:
#   uv venv --python 3.12 .venv-gen
#   uv pip install --python .venv-gen/Scripts/python.exe \
#       numpy h5py scipy matplotlib==3.10.8 psutil
# matplotlib is pinned: the 3.11.x wheel's _c_internal_utils .pyd is blocked by
# the same policy ("DLL load failed"); 3.10.8 loads fine.
PY="${PYTHON:-.venv-gen/Scripts/python.exe}"
RUN="$PY scripts/run_generation_balanced.py"

export SAMPLING=disjoint
export IMAGE_WIDTH=128 IMAGE_HEIGHT=128 IMAGE_DEPTH=1
export KMIN_LOW=1.0 KMIN_HIGH=32.0 KMAX_LOW=34.0 KMAX_HIGH=64.0
export SIGMA_LOW=0.01 SIGMA_HIGH=5.0
export BETA_LOW=-3.0 BETA_HIGH=-1.0
export MEAN_LOW=1.0 MEAN_HIGH=1.0
export BATCH_SIZE=200 NUM_WORKERS=15

NUM_IMAGES=100000 MASTER_SEED=7000 OUTPUT_FILE=flat_4param_128x128_100000_disjoint_train_v2.h5 $RUN
NUM_IMAGES=20000  MASTER_SEED=8000 OUTPUT_FILE=flat_4param_128x128_20000_disjoint_val_v2.h5   $RUN
NUM_IMAGES=50000  MASTER_SEED=9000 OUTPUT_FILE=flat_4param_128x128_50000_disjoint_test_v2.h5  $RUN

echo "ALL V2 SPLITS DONE"
