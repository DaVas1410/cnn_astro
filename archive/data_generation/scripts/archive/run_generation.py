#!/usr/bin/env python3
"""
SLURM Job Array Wrapper for Dataset Generation

Reads SLURM_ARRAY_TASK_ID to select a (k_min, k_max) pair, generates
datasets across multiple dimensions, and merges them into a single
grouped HDF5 file.

Usage:
    Launched by job.sh via SLURM job array. Can also be run standalone:
        SLURM_ARRAY_TASK_ID=0 SLURM_CPUS_PER_TASK=4 python run_generation.py
"""

import sys
import os
import configparser
import json
import shutil
import time

# Ensure project root and src are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from scripts/
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'pyFC_lib'))

import numpy as np
from dataset_generator import DatasetGen

# ─── Parameter Space ───────────────────────────────────────────────
K_MINS = list(range(1, 33))            # k_min from 1 to 32 in steps of 1
DIMENSIONS = [(128, 128), (256, 256), (512, 512)]
NUM_IMAGES = 1000
BATCH_SIZE = 50

# Distribution (fixed across all tasks)
MEAN = 1.0
SIGMA = float(np.sqrt(5.0))
BETA = -5.0 / 3.0

# Output directory (relative to project root)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

# ─── Build task list: one task per k_min, k_max = auto (Nyquist) ──
PAIRS = [(km, None) for km in K_MINS]


def create_config(output_file, k_min, k_max, height, width, num_workers):
    """Write a temporary INI config for one (k_min, k_max, dimension) run."""
    config_path = output_file.replace('.h5', '.ini')
    config = configparser.ConfigParser()

    k_max_str = 'auto' if k_max is None else str(k_max)

    config['dataset'] = {
        'output_file': output_file,
        'batch_size': str(BATCH_SIZE),
        'num_images': str(NUM_IMAGES),
        'image_width': str(width),
        'image_height': str(height),
        'image_depth': '1',
        'k_values': str(k_min),
        'k_max_values': k_max_str,
    }
    config['distribution'] = {
        'mean': str(MEAN),
        'sigma': str(SIGMA),
        'beta': str(BETA),
    }
    config['fractal'] = {
        'ni': str(width),
        'nj': str(height),
        'nk': '1',
        'verbose': 'false',
        'random_seed': 'random',
    }
    config['hdf5'] = {
        'compression': 'gzip',
        'compression_opts': '1',
        'chunks_auto': 'true',
        'chunk_size_images': 'auto',
        'chunk_size_labels': 'auto',
    }
    config['processing'] = {
        'memory_monitoring': 'true',
        'progress_reporting': 'true',
        'progress_report_interval': '5',
        'clear_memory_after_batch': 'true',
        'force_garbage_collection': 'true',
        'num_workers': str(num_workers),
    }
    config['metadata'] = {
        'method': 'lognormal',
        'store_creation_date': 'true',
        'store_parameters': 'true',
    }
    config['validation'] = {
        'validate_after_creation': 'true',
        'check_finite_values': 'true',
        'log_statistics': 'true',
    }
    config['logging'] = {
        'enable_logging': 'true',
        'log_file': output_file.replace('.h5', '.log'),
        'log_level': 'INFO',
        'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s',
        'console_output': 'false',
        'file_output': 'true',
    }

    with open(config_path, 'w') as f:
        config.write(f)
    return config_path


def main():
    # ─── Read SLURM environment ────────────────────────────────────
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    if task_id < 0 or task_id >= len(PAIRS):
        print(f'ERROR: SLURM_ARRAY_TASK_ID={task_id} out of range [0, {len(PAIRS)-1}]')
        sys.exit(1)

    k_min, k_max = PAIRS[task_id]
    kmax_label = 'auto' if k_max is None else str(k_max)
    print(f'=== Task {task_id}/{len(PAIRS)-1}: k_min={k_min}, k_max={kmax_label}, '
          f'workers={num_workers} ===')
    print(f'Dimensions: {DIMENSIONS}')
    print(f'Images per combination: {NUM_IMAGES}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'perf'), exist_ok=True)
    t_start = time.time()

    # ─── Generate one HDF5 per dimension ───────────────────────────
    dim_files = []
    dim_labels = []
    perf_dims = []
    for h, w in DIMENSIONS:
        nyquist = min(h, w) // 2
        dim_label = f'{h}x{w}'

        # Nyquist safety: skip if k_min >= Nyquist
        if k_min >= nyquist:
            print(f'  ⚠ [{dim_label}] Skipping: k_min={k_min} >= Nyquist={nyquist}')
            continue

        # k_max=None means auto → use Nyquist; otherwise clamp
        effective_kmax = nyquist if k_max is None else min(k_max, nyquist)
        if effective_kmax <= k_min:
            print(f'  ⚠ [{dim_label}] Skipping: effective k_max={effective_kmax} <= k_min={k_min}')
            continue

        output_file = os.path.join(
            OUTPUT_DIR, f'task{task_id}_kmin{k_min}_kmax-auto_{dim_label}.h5'
        )
        print(f'\n--- [{dim_label}] k_min={k_min}, k_max={effective_kmax} '
              f'(Nyquist={nyquist}) ---')

        t_dim_start = time.time()
        cfg = create_config(output_file, k_min, effective_kmax, h, w, num_workers)
        gen = DatasetGen(config_file=cfg)
        gen.populate_dataset()
        t_dim_end = time.time()

        dim_elapsed = t_dim_end - t_dim_start
        file_size_mb = os.path.getsize(output_file) / 1024**2
        perf_dims.append({
            'dimension': dim_label,
            'effective_k_max': effective_kmax,
            'num_images': NUM_IMAGES,
            'time_seconds': round(dim_elapsed, 2),
            'images_per_second': round(NUM_IMAGES / dim_elapsed, 2) if dim_elapsed > 0 else 0,
            'file_size_mb': round(file_size_mb, 2),
        })

        dim_files.append(output_file)
        dim_labels.append(dim_label)
        print(f'  -> {output_file} ({dim_elapsed:.1f}s, {NUM_IMAGES/dim_elapsed:.1f} img/s)')

    if not dim_files:
        print('WARNING: No valid dimension/k combinations for this task. Nothing generated.')
        sys.exit(0)

    # ─── Merge per-dimension files into grouped HDF5 ───────────────
    dims_str = '-'.join(str(int(dl.split('x')[0])) for dl in sorted(dim_labels, key=lambda d: int(d.split('x')[0])))
    merged_file = os.path.join(
        OUTPUT_DIR, f'task{task_id}_kmin{k_min}_kmax-auto_{dims_str}_merged.h5'
    )
    print(f'\nMerging {len(dim_files)} files -> {merged_file}')
    t_merge_start = time.time()
    DatasetGen.merge_datasets(dim_files, merged_file, dup_check=True)
    t_merge_end = time.time()
    DatasetGen.print_dataset_summary(merged_file)

    # ─── Cleanup individual dimension files ────────────────────────
    for fp in dim_files:
        os.remove(fp)
        ini = fp.replace('.h5', '.ini')
        if os.path.exists(ini):
            os.remove(ini)
        log = fp.replace('.h5', '.log')
        if os.path.exists(log):
            os.remove(log)

    elapsed = time.time() - t_start
    merged_size_mb = os.path.getsize(merged_file) / 1024**2
    total_images = sum(p['num_images'] for p in perf_dims)

    # ─── Save performance metrics as JSON ──────────────────────────
    perf_data = {
        'task_id': task_id,
        'k_min': k_min,
        'k_max': 'auto',
        'num_workers': num_workers,
        'dimensions_generated': dim_labels,
        'num_images_per_combo': NUM_IMAGES,
        'total_images': total_images,
        'per_dimension': perf_dims,
        'merge_time_seconds': round(t_merge_end - t_merge_start, 2),
        'total_time_seconds': round(elapsed, 2),
        'total_time_minutes': round(elapsed / 60, 2),
        'overall_images_per_second': round(total_images / elapsed, 2) if elapsed > 0 else 0,
        'merged_file_size_mb': round(merged_size_mb, 2),
        'output_file': os.path.basename(merged_file),
    }
    perf_file = os.path.join(OUTPUT_DIR, 'perf', f'task{task_id}_perf.json')
    with open(perf_file, 'w') as f:
        json.dump(perf_data, f, indent=2)

    print(f'\n=== Task {task_id} completed in {elapsed/60:.1f} min ===')
    print(f'Output: {merged_file} ({merged_size_mb:.1f} MB)')
    print(f'Performance: {perf_file}')


if __name__ == '__main__':
    main()
