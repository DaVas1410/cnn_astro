#!/usr/bin/env python3
"""
Single-run dataset generation with per-image random k_max.

Generates a flat HDF5 dataset (compatible with DatasetGen layout):
- images: (N, H, W)
- parameters/{k_min,k_max,beta,mean,sigma,ni,nj,nk,seed}: (N,)

Defaults target:
- N = 32000 images
- resolution = 128x128
- k_min = fixed (default 1)
- k_max sampled uniformly from continuous range [2, 64]
"""

import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

# Setup paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from scripts/
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'pyFC_lib'))

import h5py
import numpy as np

from dataset_generator import _generate_image_worker


# ------------------------- Configuration -------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PERF_DIR = os.path.join(OUTPUT_DIR, "perf")

NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "32000"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "128"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "128"))
IMAGE_DEPTH = 1

KMIN_FIXED = int(os.environ.get("KMIN_FIXED", "1"))  # Fixed k_min value
KMAX_LOW = int(os.environ.get("KMAX_LOW", "2"))
KMAX_HIGH = int(os.environ.get("KMAX_HIGH", "64"))  # inclusive

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("NUM_WORKERS", "1")))

MEAN = float(os.environ.get("MEAN", "1.0"))
SIGMA = float(np.sqrt(5.0))
BETA = float(-5.0 / 3.0)

COMPRESSION = "gzip"
COMPRESSION_OPTS = 1


def create_hdf5_layout(output_file: str, total_images: int, image_height: int, image_width: int) -> None:
    with h5py.File(output_file, "w") as f:
        f.create_dataset(
            "images",
            shape=(total_images, image_height, image_width),
            maxshape=(None, image_height, image_width),
            dtype=np.float32,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
            chunks=(1, image_height, image_width),
        )

        params_group = f.create_group("parameters")

        for name in ["k_min", "k_max", "beta", "mean", "sigma"]:
            params_group.create_dataset(
                name,
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.float32,
                compression=COMPRESSION,
                compression_opts=COMPRESSION_OPTS,
                chunks=True,
            )

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

        params_group.create_dataset(
            "seed",
            shape=(total_images,),
            maxshape=(None,),
            dtype=np.int64,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
            chunks=True,
        )

        f.attrs["method"] = "lognormal"
        f.attrs["image_dimensions"] = (image_width, image_height, IMAGE_DEPTH)
        f.attrs["batch_size"] = BATCH_SIZE
        f.attrs["total_images"] = total_images
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["generator"] = "run_generation_random_kmax.py"

        params_group.attrs["description"] = "Per-image generation parameters"
        f["images"].attrs["description"] = "Log10-transformed fractal density slices"
        f["images"].attrs["transform"] = "log10(density_slice)"


def main() -> None:
    if NUM_IMAGES <= 0:
        raise ValueError(f"NUM_IMAGES must be > 0, got {NUM_IMAGES}")
    if KMAX_LOW < 1 or KMAX_HIGH < KMAX_LOW:
        raise ValueError(f"Invalid k_max range: [{KMAX_LOW}, {KMAX_HIGH}]")
    if KMIN_FIXED < 1:
        raise ValueError(f"Invalid k_min: {KMIN_FIXED}")

    nyquist = min(IMAGE_HEIGHT, IMAGE_WIDTH) // 2
    if KMAX_HIGH > nyquist:
        print(
            f"WARNING: KMAX_HIGH={KMAX_HIGH} > Nyquist={nyquist}. "
            "Some samples may fail if k_max>Nyquist for chosen geometry."
        )
    if KMIN_FIXED >= KMAX_LOW:
        raise ValueError(
            f"Invalid configuration: k_min={KMIN_FIXED} must be < k_max_low={KMAX_LOW}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERF_DIR, exist_ok=True)

    output_name = f"randomkmax_kmin{KMIN_FIXED}_{KMAX_LOW}-{KMAX_HIGH}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}.h5"
    output_file = os.path.join(OUTPUT_DIR, output_name)

    print("=" * 64)
    print("Random-k_max dataset generation")
    print(f"Output: {output_file}")
    print(f"Images: {NUM_IMAGES}")
    print(f"Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    print(f"k_min: {KMIN_FIXED} (fixed)")
    print(f"k_max range: [{KMAX_LOW}, {KMAX_HIGH}] (uniform continuous)")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 64)

    create_hdf5_layout(output_file, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)

    batch_times = []
    start_time = time.time()
    global_index = 0

    with h5py.File(output_file, "a") as h5f:
        images_ds = h5f["images"]
        params_group = h5f["parameters"]

        with mp.Pool(processes=NUM_WORKERS) as pool:
            num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in range(num_batches):
                t0 = time.time()

                current_batch_size = min(BATCH_SIZE, NUM_IMAGES - global_index)
                kmaxs = np.random.uniform(
                    float(KMAX_LOW),
                    float(KMAX_HIGH),
                    size=current_batch_size,
                ).astype(np.float32)
                seeds = np.random.randint(0, 2**31, size=current_batch_size, dtype=np.int64)

                work_args = [
                    (
                        float(KMIN_FIXED),
                        float(kmaxs[i]),
                        IMAGE_WIDTH,
                        IMAGE_HEIGHT,
                        IMAGE_DEPTH,
                        MEAN,
                        SIGMA,
                        BETA,
                        False,
                        int(seeds[i]),
                    )
                    for i in range(current_batch_size)
                ]

                results = pool.starmap(_generate_image_worker, work_args)

                batch_images = np.array([img for img, _ in results], dtype=np.float32)
                end_index = global_index + current_batch_size
                images_ds[global_index:end_index] = batch_images

                batch_params = {
                    "k_min": [],
                    "k_max": [],
                    "beta": [],
                    "mean": [],
                    "sigma": [],
                    "ni": [],
                    "nj": [],
                    "nk": [],
                    "seed": [],
                }

                for _, params in results:
                    for key in batch_params:
                        batch_params[key].append(params[key])

                for key, values in batch_params.items():
                    params_group[key][global_index:end_index] = values

                global_index = end_index
                elapsed = time.time() - start_time
                rate = global_index / elapsed if elapsed > 0 else 0.0
                eta = (NUM_IMAGES - global_index) / rate if rate > 0 else 0.0

                dt = time.time() - t0
                batch_times.append(dt)
                print(
                    f"Batch {batch_idx + 1}/{num_batches} | "
                    f"{global_index}/{NUM_IMAGES} ({100.0 * global_index / NUM_IMAGES:.1f}%) | "
                    f"{rate:.2f} img/s | ETA {eta/60:.1f} min"
                )

    total_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / 1024**2

    perf = {
        "run_type": "random_kmax_single_run",
        "output_file": os.path.basename(output_file),
        "num_images": NUM_IMAGES,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "k_min": KMIN_FIXED,
        "k_max_range": [KMAX_LOW, KMAX_HIGH],
        "k_max_sampling": "uniform_continuous",
        "distribution": {"mean": MEAN, "sigma": SIGMA, "beta": BETA},
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "time_seconds": round(total_time, 2),
        "time_minutes": round(total_time / 60.0, 2),
        "images_per_second": round(NUM_IMAGES / total_time, 2) if total_time > 0 else 0,
        "avg_batch_seconds": round(float(np.mean(batch_times)), 3) if batch_times else 0,
        "file_size_mb": round(file_size_mb, 2),
        "created_at": datetime.now().isoformat(),
    }

    perf_file = os.path.join(PERF_DIR, f"randomkmax_kmin{KMIN_FIXED}_{KMAX_LOW}-{KMAX_HIGH}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}_perf.json")
    with open(perf_file, "w", encoding="utf-8") as f:
        json.dump(perf, f, indent=2)

    print("=" * 64)
    print(f"Completed in {total_time / 60.0:.1f} min")
    print(f"Output file: {output_file} ({file_size_mb:.1f} MB)")
    print(f"Perf JSON:  {perf_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()
