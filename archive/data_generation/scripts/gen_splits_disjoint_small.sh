#!/usr/bin/env bash
# Smaller version of gen_splits_disjoint.sh: three disjoint-sampling splits as
# separate HDF5 files, each with its own MASTER_SEED so parameter draws and pyFC
# seeds never overlap between splits -- nor with the large 100k/20k/50k splits
# (those use MASTER_SEED 1000/2000/3000).
#
#   train 10000  MASTER_SEED=4000
#   val    2000  MASTER_SEED=5000
#   test   5000  MASTER_SEED=6000
#
# Same distribution as the chosen training config: k_min~U[1,32],
# k_max~U[34,64], sigma~U[0.01,5], beta~U[-3,-1], mean=1.0, 128x128.
set -euo pipefail
cd "$(dirname "$0")/.."

# matplotlib is pinned: on this Windows box the 3.11.x wheel's _c_internal_utils
# .pyd is blocked by an Application Control policy ("DLL load failed"); 3.10.8
# (the requirements_py311.txt pin) loads fine.
RUN="uv run --with numpy --with h5py --with scipy --with matplotlib==3.10.8 --with psutil python scripts/run_generation_balanced.py"

export SAMPLING=disjoint
export IMAGE_WIDTH=128 IMAGE_HEIGHT=128 IMAGE_DEPTH=1
export KMIN_LOW=1.0 KMIN_HIGH=32.0 KMAX_LOW=34.0 KMAX_HIGH=64.0
export SIGMA_LOW=0.01 SIGMA_HIGH=5.0
export BETA_LOW=-3.0 BETA_HIGH=-1.0
export MEAN_LOW=1.0 MEAN_HIGH=1.0
export BATCH_SIZE=200 NUM_WORKERS=15

NUM_IMAGES=10000 MASTER_SEED=4000 OUTPUT_FILE=flat_4param_128x128_10000_disjoint_train.h5 $RUN
NUM_IMAGES=2000  MASTER_SEED=5000 OUTPUT_FILE=flat_4param_128x128_2000_disjoint_val.h5    $RUN
NUM_IMAGES=5000  MASTER_SEED=6000 OUTPUT_FILE=flat_4param_128x128_5000_disjoint_test.h5   $RUN

echo "ALL SMALL SPLITS DONE"
