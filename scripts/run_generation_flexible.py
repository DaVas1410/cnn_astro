#!/usr/bin/env python3
"""
Flexible Dataset Generation with Continuous Random Parameter Sampling

Generates a dataset of fractal cube slices with per-image random sampling of:
- k_min: Minimum wavenumber (uniform from range)
- k_max: Maximum wavenumber (uniform from range or auto/Nyquist)
- sigma: Log-normal distribution parameter (uniform from range)
- mean: Log-normal distribution mean (uniform from range)

Each image has unique parameters, creating a highly diverse dataset.

Configuration via environment variables:
  IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH: Image dimensions
  NUM_IMAGES: Total number of images to generate
  BATCH_SIZE: Images per batch
  NUM_WORKERS: Parallel workers (default: 1)
  KMIN_LOW, KMIN_HIGH: k_min sampling range
  KMAX_LOW, KMAX_HIGH: k_max sampling range (or 'auto' for Nyquist)
  SIGMA_LOW, SIGMA_HIGH: sigma sampling range
  MEAN_LOW, MEAN_HIGH: mean sampling range
  OUTPUT_FILE: Output HDF5 file path

Usage:
  # Local execution with environment variables
  NUM_IMAGES=1000 IMAGE_WIDTH=512 IMAGE_HEIGHT=512 KMIN_LOW=1 KMIN_HIGH=32 \\
    KMAX_LOW=100 KMAX_HIGH=256 SIGMA_LOW=2.0 SIGMA_HIGH=2.5 python run_generation_flexible.py
  
  # In SLURM job script
  python scripts/run_generation_flexible.py

Author: Copilot
Date: 2026
"""

import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

# Setup paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'pyFC_lib'))

import h5py
import numpy as np

from dataset_generator import _generate_image_worker
from param_sampler import ParameterSampler


# ─── Configuration from Environment Variables ───────────────────────────────

def get_env_float(name: str, default: float) -> float:
    """Get float environment variable with default."""
    return float(os.environ.get(name, str(default)))


def get_env_int(name: str, default: int) -> int:
    """Get int environment variable with default."""
    return int(os.environ.get(name, str(default)))


def get_env_str(name: str, default: str) -> str:
    """Get string environment variable with default."""
    return os.environ.get(name, default)


# Output and dimensions
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PERF_DIR = os.path.join(OUTPUT_DIR, "perf")

OUTPUT_FILE = get_env_str("OUTPUT_FILE", "")
NUM_IMAGES = get_env_int("NUM_IMAGES", 1000)
IMAGE_WIDTH = get_env_int("IMAGE_WIDTH", 512)
IMAGE_HEIGHT = get_env_int("IMAGE_HEIGHT", 512)
IMAGE_DEPTH = get_env_int("IMAGE_DEPTH", 1)

# Batch processing
BATCH_SIZE = get_env_int("BATCH_SIZE", 50)
NUM_WORKERS = get_env_int("NUM_WORKERS", 1)
# Support SLURM_CPUS_PER_TASK as fallback for HPC environments
if NUM_WORKERS == 1:
    NUM_WORKERS = get_env_int("SLURM_CPUS_PER_TASK", 1)

# K-min range
KMIN_LOW = get_env_float("KMIN_LOW", 1.0)
KMIN_HIGH = get_env_float("KMIN_HIGH", 32.0)

# K-max range (or 'auto' for Nyquist)
KMAX_MODE = get_env_str("KMAX_MODE", "range").lower()
KMAX_LOW = get_env_float("KMAX_LOW", 100.0)
KMAX_HIGH = get_env_float("KMAX_HIGH", 256.0)

# Distribution parameters
SIGMA_LOW = get_env_float("SIGMA_LOW", 2.0)
SIGMA_HIGH = get_env_float("SIGMA_HIGH", 2.5)
MEAN_LOW = get_env_float("MEAN_LOW", 1.0)
MEAN_HIGH = get_env_float("MEAN_HIGH", 1.0)

# Beta (power spectrum slope) - typically fixed
BETA = get_env_float("BETA", -5.0 / 3.0)

# HDF5 options
COMPRESSION = "gzip"
COMPRESSION_OPTS = 1


# ─── HDF5 File Creation ─────────────────────────────────────────────────────

def create_hdf5_layout(output_file: str, total_images: int) -> None:
    """
    Create HDF5 file structure with flat layout for parameter tracking.
    
    Creates:
    - images: (total_images, IMAGE_HEIGHT, IMAGE_WIDTH) float32 array
    - parameters/: Group containing per-image parameter arrays
    """
    with h5py.File(output_file, "w") as f:
        # Main images dataset
        f.create_dataset(
            "images",
            shape=(total_images, IMAGE_HEIGHT, IMAGE_WIDTH),
            maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=np.float32,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
            chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        f["images"].attrs["description"] = "Log10-transformed fractal density slices"
        f["images"].attrs["transform"] = "log10(density_slice)"

        # Parameters group
        params_group = f.create_group("parameters")
        params_group.attrs["description"] = "Per-image generation parameters"

        # Floating-point parameters
        for name in ["k_min", "k_max", "sigma", "mean", "beta"]:
            params_group.create_dataset(
                name,
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.float32,
                compression=COMPRESSION,
                compression_opts=COMPRESSION_OPTS,
                chunks=True,
            )

        # Integer parameters
        for name in ["ni", "nj", "nk"]:
            params_group.create_dataset(
                name,
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.int32,
                compression=COMPRESSION,
                compression_opts=COMPRESSION_OPTS,
                chunks=True,
            )

        # Seed (int64)
        params_group.create_dataset(
            "seed",
            shape=(total_images,),
            maxshape=(None,),
            dtype=np.int64,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
            chunks=True,
        )

        # File-level metadata
        f.attrs["method"] = "lognormal_flexible"
        f.attrs["image_dimensions"] = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]
        f.attrs["batch_size"] = BATCH_SIZE
        f.attrs["total_images"] = total_images
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["generator"] = "run_generation_flexible.py"
        f.attrs["num_workers"] = NUM_WORKERS


# ─── Main Generation ───────────────────────────────────────────────────────

def main() -> None:
    """Main generation workflow."""
    
    # Validate input parameters
    if NUM_IMAGES <= 0:
        raise ValueError(f"NUM_IMAGES must be > 0, got {NUM_IMAGES}")
    if KMIN_HIGH < KMIN_LOW:
        raise ValueError(f"KMIN_HIGH ({KMIN_HIGH}) < KMIN_LOW ({KMIN_LOW})")
    if SIGMA_LOW <= 0 or SIGMA_HIGH <= 0:
        raise ValueError(f"SIGMA range must be > 0, got [{SIGMA_LOW}, {SIGMA_HIGH}]")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERF_DIR, exist_ok=True)
    
    # Determine output filename if not specified
    if not OUTPUT_FILE:
        auto_kmax_str = "auto" if KMAX_MODE == "auto" else f"{KMAX_LOW}-{KMAX_HIGH}"
        output_name = (
            f"flexible_{KMIN_LOW}-{KMIN_HIGH}_"
            f"sigma{SIGMA_LOW}-{SIGMA_HIGH}_"
            f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}.h5"
        )
        output_path = os.path.join(OUTPUT_DIR, output_name)
    else:
        # If OUTPUT_FILE is absolute, use it; otherwise join with OUTPUT_DIR
        if os.path.isabs(OUTPUT_FILE):
            output_path = OUTPUT_FILE
        else:
            output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Initialize parameter sampler
    auto_kmax = KMAX_MODE == "auto"
    kmax_range = None if auto_kmax else (KMAX_LOW, KMAX_HIGH)
    
    sampler = ParameterSampler(
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_depth=IMAGE_DEPTH,
        kmin_range=(KMIN_LOW, KMIN_HIGH),
        kmax_range=kmax_range,
        sigma_range=(SIGMA_LOW, SIGMA_HIGH),
        mean_range=(MEAN_LOW, MEAN_HIGH),
        auto_kmax=auto_kmax,
    )
    
    print("=" * 70)
    print("FLEXIBLE DATASET GENERATION WITH RANDOM PARAMETER SAMPLING")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Images: {NUM_IMAGES}")
    print(f"Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_DEPTH}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(sampler)
    print("=" * 70)
    
    # Create HDF5 file structure
    create_hdf5_layout(output_path, NUM_IMAGES)
    
    # Generation loop
    batch_times = []
    start_time = time.time()
    global_index = 0
    
    with h5py.File(output_path, "a") as h5f:
        images_ds = h5f["images"]
        params_group = h5f["parameters"]
        
        with mp.Pool(processes=NUM_WORKERS) as pool:
            num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                t0 = time.time()
                
                # Determine batch size for this batch (may be smaller on last batch)
                current_batch_size = min(BATCH_SIZE, NUM_IMAGES - global_index)
                
                # Sample parameters for this batch
                sampled_params = sampler.sample_batch(current_batch_size)
                seeds = np.random.randint(0, 2**31, size=current_batch_size, dtype=np.int64)
                
                # Prepare work arguments for workers
                work_args = [
                    (
                        float(sampled_params['k_min'][i]),
                        float(sampled_params['k_max'][i]),
                        IMAGE_WIDTH,
                        IMAGE_HEIGHT,
                        IMAGE_DEPTH,
                        float(sampled_params['mean'][i]),
                        float(sampled_params['sigma'][i]),
                        float(BETA),
                        False,  # verbose
                        int(seeds[i]),
                    )
                    for i in range(current_batch_size)
                ]
                
                # Generate images in parallel
                results = pool.starmap(_generate_image_worker, work_args)
                
                # Store images
                batch_images = np.array([img for img, _ in results], dtype=np.float32)
                end_index = global_index + current_batch_size
                images_ds[global_index:end_index] = batch_images
                
                # Collect parameters from results
                batch_params = {
                    "k_min": [],
                    "k_max": [],
                    "sigma": [],
                    "mean": [],
                    "beta": [],
                    "ni": [],
                    "nj": [],
                    "nk": [],
                    "seed": [],
                }
                
                for _, params_dict in results:
                    for key in batch_params:
                        batch_params[key].append(params_dict[key])
                
                # Store parameters
                for key, values in batch_params.items():
                    params_group[key][global_index:end_index] = values
                
                # Update progress
                global_index = end_index
                elapsed = time.time() - start_time
                rate = global_index / elapsed if elapsed > 0 else 0.0
                eta = (NUM_IMAGES - global_index) / rate if rate > 0 else 0.0
                
                dt = time.time() - t0
                batch_times.append(dt)
                
                pct = 100.0 * global_index / NUM_IMAGES
                print(
                    f"Batch {batch_idx + 1:3d}/{num_batches} | "
                    f"{global_index:5d}/{NUM_IMAGES} ({pct:5.1f}%) | "
                    f"{rate:7.2f} img/s | ETA {eta/60:6.1f} min"
                )
        
        # Store sampler configuration in metadata
        sampler_config = sampler.get_config()
        h5f.attrs["sampler_config"] = json.dumps(sampler_config)
    
    # Performance summary
    total_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    
    perf_summary = {
        "run_type": "flexible_random_parameters",
        "output_file": os.path.basename(output_path),
        "num_images": NUM_IMAGES,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
        "sampling_config": {
            "k_min_range": [KMIN_LOW, KMIN_HIGH],
            "k_max_range": "auto" if auto_kmax else [KMAX_LOW, KMAX_HIGH],
            "sigma_range": [SIGMA_LOW, SIGMA_HIGH],
            "mean_range": [MEAN_LOW, MEAN_HIGH],
            "beta": BETA,
        },
        "distribution": {
            "type": "lognormal_flexible",
            "per_image_sampling": True,
        },
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "time_seconds": round(total_time, 2),
        "time_minutes": round(total_time / 60.0, 2),
        "images_per_second": round(NUM_IMAGES / total_time, 2) if total_time > 0 else 0,
        "avg_batch_seconds": round(float(np.mean(batch_times)), 3) if batch_times else 0,
        "file_size_mb": round(file_size_mb, 2),
        "created_at": datetime.now().isoformat(),
    }
    
    # Save performance summary
    perf_file = os.path.join(
        PERF_DIR,
        f"flexible_{KMIN_LOW}-{KMIN_HIGH}_sigma{SIGMA_LOW}-{SIGMA_HIGH}_"
        f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}_perf.json"
    )
    with open(perf_file, "w", encoding="utf-8") as f:
        json.dump(perf_summary, f, indent=2)
    
    # Print summary
    print("=" * 70)
    print(f"✓ Generation completed in {total_time / 60.0:.2f} minutes")
    print(f"✓ Output file: {output_path} ({file_size_mb:.1f} MB)")
    print(f"✓ Perf JSON:   {perf_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
