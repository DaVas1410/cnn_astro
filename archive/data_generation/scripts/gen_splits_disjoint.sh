#!/usr/bin/env bash
# Generate the three disjoint-sampling splits (train/val/test) as separate HDF5
# files, each with its own MASTER_SEED so parameter draws and pyFC seeds never
# overlap between splits.
#
#   train 100000  MASTER_SEED=1000
#   val    20000  MASTER_SEED=2000
#   test   50000  MASTER_SEED=3000
#
# Config matches data/raw/flat_4param_128x128_100000_disjoint.h5 (the chosen
# training distribution): k_min~U[1,32], k_max~U[34,64], sigma~U[0.01,5],
# beta~U[-3,-1], mean=1.0, 128x128.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="uv run --with numpy --with h5py --with scipy --with matplotlib --with psutil python scripts/run_generation_balanced.py"

export SAMPLING=disjoint
export IMAGE_WIDTH=128 IMAGE_HEIGHT=128 IMAGE_DEPTH=1
export KMIN_LOW=1.0 KMIN_HIGH=32.0 KMAX_LOW=34.0 KMAX_HIGH=64.0
export SIGMA_LOW=0.01 SIGMA_HIGH=5.0
export BETA_LOW=-3.0 BETA_HIGH=-1.0
export MEAN_LOW=1.0 MEAN_HIGH=1.0
export BATCH_SIZE=200 NUM_WORKERS=15

NUM_IMAGES=100000 MASTER_SEED=1000 OUTPUT_FILE=flat_4param_128x128_100000_disjoint_train.h5 $RUN
NUM_IMAGES=20000  MASTER_SEED=2000 OUTPUT_FILE=flat_4param_128x128_20000_disjoint_val.h5   $RUN
NUM_IMAGES=50000  MASTER_SEED=3000 OUTPUT_FILE=flat_4param_128x128_50000_disjoint_test.h5  $RUN

echo "ALL SPLITS DONE"
