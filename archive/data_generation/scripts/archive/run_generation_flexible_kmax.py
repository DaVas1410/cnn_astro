#!/usr/bin/env python3
"""
Flexible Dataset Generation — varying k_min, k_max, and sigma.

Sampling strategy (k_max first, then k_min conditional on k_max):
  k_max  ~ Uniform[KMAX_LOW, KMAX_HIGH]          — uniform, independent
  k_min  ~ Uniform[KMIN_LOW, k_max - KMAX_MIN_GAP]  — conditional on k_max
  sigma  ~ Uniform[SIGMA_LOW, SIGMA_HIGH]
  mean   = 1.0 (fixed)
  beta   = -5/3 (fixed)

This ensures k_max is uniformly distributed and not correlated with k_min.
pyFC requires k_max - k_min >= KMAX_MIN_GAP (2.0) to avoid degenerate spectra.

Environment variables (all optional, shown with defaults):
  IMAGE_WIDTH=128   IMAGE_HEIGHT=128   IMAGE_DEPTH=1
  NUM_IMAGES=100000   BATCH_SIZE=50   NUM_WORKERS=1
  KMAX_LOW=5.0   KMAX_HIGH=64.0
  KMIN_LOW=1.0
  SIGMA_LOW=0.01  SIGMA_HIGH=5.0
  OUTPUT_FILE=''   (auto-generates name if empty)
"""

import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'pyFC_lib'))

import h5py
import numpy as np

from dataset_generator import _generate_image_worker


def _env_float(name, default): return float(os.environ.get(name, str(default)))
def _env_int(name, default):   return int(os.environ.get(name, str(default)))
def _env_str(name, default):   return os.environ.get(name, default)


OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'data', 'raw')
PERF_DIR     = os.path.join(OUTPUT_DIR, 'perf')

NUM_IMAGES   = _env_int('NUM_IMAGES',   100000)
IMAGE_WIDTH  = _env_int('IMAGE_WIDTH',  128)
IMAGE_HEIGHT = _env_int('IMAGE_HEIGHT', 128)
IMAGE_DEPTH  = _env_int('IMAGE_DEPTH',  1)
BATCH_SIZE   = _env_int('BATCH_SIZE',   50)
NUM_WORKERS  = _env_int('NUM_WORKERS',  1)
if NUM_WORKERS == 1:
    NUM_WORKERS = _env_int('SLURM_CPUS_PER_TASK', 1)

KMAX_LOW   = _env_float('KMAX_LOW',   5.0)
KMAX_HIGH  = _env_float('KMAX_HIGH',  64.0)
KMIN_LOW   = _env_float('KMIN_LOW',   1.0)
SIGMA_LOW  = _env_float('SIGMA_LOW',  0.01)
SIGMA_HIGH = _env_float('SIGMA_HIGH', 5.0)
MEAN_LOW   = _env_float('MEAN_LOW',   1.0)
MEAN_HIGH  = _env_float('MEAN_HIGH',  1.0)
BETA       = _env_float('BETA',       -5.0 / 3.0)
OUTPUT_FILE = _env_str('OUTPUT_FILE', '')

COMPRESSION      = 'gzip'
COMPRESSION_OPTS = 1

KMAX_MIN_GAP = 2.0  # pyFC requires k_max - k_min >= this to avoid degenerate spectra


def _create_hdf5(output_file: str, total_images: int) -> None:
    with h5py.File(output_file, 'w') as f:
        f.create_dataset(
            'images',
            shape=(total_images, IMAGE_HEIGHT, IMAGE_WIDTH),
            maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=np.float32,
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
            chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        f['images'].attrs['description'] = 'Log10-transformed fractal density slices'
        f['images'].attrs['transform']   = 'log10(density_slice)'

        pg = f.create_group('parameters')
        pg.attrs['description'] = 'Per-image generation parameters'

        for name in ['k_min', 'k_max', 'sigma', 'mean', 'beta']:
            pg.create_dataset(name, shape=(total_images,), maxshape=(None,),
                              dtype=np.float32, compression=COMPRESSION,
                              compression_opts=COMPRESSION_OPTS, chunks=True)
        for name in ['ni', 'nj', 'nk']:
            pg.create_dataset(name, shape=(total_images,), maxshape=(None,),
                              dtype=np.int32, compression=COMPRESSION,
                              compression_opts=COMPRESSION_OPTS, chunks=True)
        pg.create_dataset('seed', shape=(total_images,), maxshape=(None,),
                          dtype=np.int64, compression=COMPRESSION,
                          compression_opts=COMPRESSION_OPTS, chunks=True)

        f.attrs['method']           = 'lognormal_flexible_kmax_v2'
        f.attrs['sampling']         = 'kmax_first: kmax~U[KMAX_LOW,KMAX_HIGH], kmin~U[KMIN_LOW,kmax-gap]'
        f.attrs['image_dimensions'] = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]
        f.attrs['batch_size']       = BATCH_SIZE
        f.attrs['total_images']     = total_images
        f.attrs['creation_date']    = datetime.now().isoformat()
        f.attrs['generator']        = 'run_generation_flexible_kmax.py'
        f.attrs['num_workers']      = NUM_WORKERS
        f.attrs['kmin_low']         = KMIN_LOW
        f.attrs['kmax_range']       = [KMAX_LOW, KMAX_HIGH]
        f.attrs['sigma_range']      = [SIGMA_LOW, SIGMA_HIGH]


def main() -> None:
    if NUM_IMAGES <= 0:
        raise ValueError(f'NUM_IMAGES must be > 0, got {NUM_IMAGES}')
    if KMAX_HIGH <= KMAX_LOW:
        raise ValueError(f'KMAX_HIGH ({KMAX_HIGH}) <= KMAX_LOW ({KMAX_LOW})')
    if KMAX_LOW < KMIN_LOW + KMAX_MIN_GAP:
        raise ValueError(
            f'KMAX_LOW ({KMAX_LOW}) must be >= KMIN_LOW ({KMIN_LOW}) + '
            f'KMAX_MIN_GAP ({KMAX_MIN_GAP}) = {KMIN_LOW + KMAX_MIN_GAP:.1f} '
            'so that k_min has a valid range for the smallest k_max.')
    if SIGMA_LOW <= 0:
        raise ValueError(f'SIGMA_LOW must be > 0, got {SIGMA_LOW}')
    if SIGMA_HIGH < SIGMA_LOW:
        raise ValueError(f'SIGMA_HIGH ({SIGMA_HIGH}) < SIGMA_LOW ({SIGMA_LOW})')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERF_DIR,   exist_ok=True)

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if OUTPUT_FILE:
        output_path = (OUTPUT_FILE if os.path.isabs(OUTPUT_FILE)
                       else os.path.join(OUTPUT_DIR, OUTPUT_FILE))
    else:
        output_path = os.path.join(
            OUTPUT_DIR,
            f'flexible_kmax_{run_ts}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}.h5'
        )

    kmin_max_possible = KMAX_HIGH - KMAX_MIN_GAP

    print('=' * 70)
    print('FLEXIBLE DATASET GENERATION — k_max first, k_min conditional')
    print('=' * 70)
    print(f'Output:     {output_path}')
    print(f'Images:     {NUM_IMAGES}')
    print(f'Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_DEPTH}')
    print(f'Workers:    {NUM_WORKERS}   Batch: {BATCH_SIZE}')
    print(f'k_max:      [{KMAX_LOW}, {KMAX_HIGH}] uniform (independent)')
    print(f'k_min:      [{KMIN_LOW}, k_max-{KMAX_MIN_GAP}]  (uniform conditional on k_max)')
    print(f'            k_min effective max = {kmin_max_possible:.1f} (when k_max={KMAX_HIGH})')
    print(f'sigma:      [{SIGMA_LOW}, {SIGMA_HIGH}] uniform')
    print('=' * 70)

    _create_hdf5(output_path, NUM_IMAGES)

    batch_times  = []
    start_time   = time.time()
    global_index = 0

    with h5py.File(output_path, 'a') as h5f:
        images_ds = h5f['images']
        pg        = h5f['parameters']

        with mp.Pool(processes=NUM_WORKERS) as pool:
            num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in range(num_batches):
                t0  = time.time()
                cur = min(BATCH_SIZE, NUM_IMAGES - global_index)

                # Sample k_max uniformly, then k_min conditionally
                k_maxs = np.random.uniform(KMAX_LOW, KMAX_HIGH, size=cur)
                k_mins = np.array([
                    np.random.uniform(KMIN_LOW, float(kx) - KMAX_MIN_GAP)
                    for kx in k_maxs
                ])
                sigmas = np.random.uniform(SIGMA_LOW, SIGMA_HIGH, size=cur)
                means  = np.random.uniform(MEAN_LOW,  MEAN_HIGH,  size=cur)
                seeds  = np.random.randint(0, 2**31,  size=cur, dtype=np.int64)

                work_args = [
                    (float(k_mins[i]), float(k_maxs[i]),
                     IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH,
                     float(means[i]), float(sigmas[i]), float(BETA),
                     False, int(seeds[i]))
                    for i in range(cur)
                ]
                results = pool.starmap(_generate_image_worker, work_args)

                end_index = global_index + cur
                images_ds[global_index:end_index] = np.array(
                    [img for img, _ in results], dtype=np.float32)

                bparams = {k: [] for k in
                           ['k_min', 'k_max', 'sigma', 'mean', 'beta',
                            'ni', 'nj', 'nk', 'seed']}
                for _, p in results:
                    for key in bparams:
                        bparams[key].append(p[key])
                for key, vals in bparams.items():
                    pg[key][global_index:end_index] = vals

                global_index = end_index
                dt      = time.time() - t0
                elapsed = time.time() - start_time
                rate    = global_index / elapsed if elapsed > 0 else 0
                eta     = (NUM_IMAGES - global_index) / rate if rate > 0 else 0
                batch_times.append(dt)
                print(f'Batch {batch_idx+1:3d}/{num_batches} | '
                      f'{global_index:6d}/{NUM_IMAGES} '
                      f'({100*global_index/NUM_IMAGES:5.1f}%) | '
                      f'{rate:7.2f} img/s | ETA {eta/60:6.1f} min')

    total_time   = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)

    perf = {
        'run_type':          'flexible_kmax_first',
        'sampling':          'kmax~U[KMAX_LOW,KMAX_HIGH], kmin~U[KMIN_LOW,kmax-gap]',
        'output_file':       os.path.basename(output_path),
        'num_images':        NUM_IMAGES,
        'image_size':        [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
        'kmax_range':        [KMAX_LOW, KMAX_HIGH],
        'kmin_low':          KMIN_LOW,
        'sigma_range':       [SIGMA_LOW, SIGMA_HIGH],
        'batch_size':        BATCH_SIZE,
        'num_workers':       NUM_WORKERS,
        'time_seconds':      round(total_time, 2),
        'time_minutes':      round(total_time / 60.0, 2),
        'images_per_second': round(NUM_IMAGES / total_time, 2) if total_time > 0 else 0,
        'avg_batch_seconds': round(float(np.mean(batch_times)), 3) if batch_times else 0,
        'file_size_mb':      round(file_size_mb, 2),
        'created_at':        datetime.now().isoformat(),
    }
    perf_path = os.path.join(
        PERF_DIR,
        f'flexible_kmax_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}_{run_ts}_perf.json')
    with open(perf_path, 'w') as fp:
        json.dump(perf, fp, indent=2)

    print('=' * 70)
    print(f'Complete in {total_time / 60:.2f} min | {file_size_mb:.1f} MB')
    print(f'Output: {output_path}')
    print(f'Perf:   {perf_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
