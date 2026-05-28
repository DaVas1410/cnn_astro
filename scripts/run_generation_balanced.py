#!/usr/bin/env python3
"""
Balanced 4-Parameter Dataset Generation

Generates a dataset with joint-uniform sampling over (k_min, k_max, sigma, beta)
using rejection sampling to enforce k_max > k_min + GAP. All four parameters
are continuous floats (no integer casting).

Resulting marginal distributions are close to uniform on:
  k_min  ~ U[1.0,  62.0]
  k_max  ~ U[3.0,  64.0]    (with k_max > k_min + GAP)
  sigma  ~ U[0.01, 5.0]
  beta   ~ U[-3.0, -1.0]    (covers Kolmogorov, Burgers, Kraichnan regimes)

Configuration via environment variables (defaults are the recommended values):
  NUM_IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH,
  KMIN_LOW, KMIN_HIGH, KMAX_LOW, KMAX_HIGH, KMIN_KMAX_GAP,
  SIGMA_LOW, SIGMA_HIGH, BETA_LOW, BETA_HIGH,
  MEAN_LOW, MEAN_HIGH, BATCH_SIZE, NUM_WORKERS, OUTPUT_FILE
"""

import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'pyFC_lib'))

import h5py
import numpy as np

from dataset_generator import _generate_image_worker


def _f(name, default): return float(os.environ.get(name, default))
def _i(name, default): return int(os.environ.get(name, default))
def _s(name, default): return os.environ.get(name, default)


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PERF_DIR = os.path.join(OUTPUT_DIR, "perf")

NUM_IMAGES   = _i("NUM_IMAGES", 100000)
IMAGE_WIDTH  = _i("IMAGE_WIDTH", 128)
IMAGE_HEIGHT = _i("IMAGE_HEIGHT", 128)
IMAGE_DEPTH  = _i("IMAGE_DEPTH", 1)

BATCH_SIZE  = _i("BATCH_SIZE", 200)
NUM_WORKERS = _i("NUM_WORKERS", 1)
if NUM_WORKERS == 1:
    NUM_WORKERS = _i("SLURM_CPUS_PER_TASK", max(1, mp.cpu_count() - 1))

KMIN_LOW  = _f("KMIN_LOW", 1.0)
KMIN_HIGH = _f("KMIN_HIGH", 62.0)
KMAX_LOW  = _f("KMAX_LOW", 3.0)
KMAX_HIGH = _f("KMAX_HIGH", 64.0)
KMIN_KMAX_GAP = _f("KMIN_KMAX_GAP", 2.0)

SIGMA_LOW  = _f("SIGMA_LOW", 0.01)
SIGMA_HIGH = _f("SIGMA_HIGH", 5.0)
BETA_LOW   = _f("BETA_LOW", -3.0)
BETA_HIGH  = _f("BETA_HIGH", -1.0)
MEAN_LOW   = _f("MEAN_LOW", 1.0)
MEAN_HIGH  = _f("MEAN_HIGH", 1.0)

OUTPUT_FILE = _s("OUTPUT_FILE", "balanced_4param_128x128_100000.h5")

COMPRESSION = "gzip"
COMPRESSION_OPTS = 1


def sample_joint_uniform(n, rng):
    """Joint-uniform sampling over (k_min, k_max) with rejection on k_max > k_min + GAP.

    Returns float arrays of length n. Oversamples by 2.2x to amortize ~50% rejection.
    """
    out_kmin = np.empty(n, dtype=np.float64)
    out_kmax = np.empty(n, dtype=np.float64)
    filled = 0
    while filled < n:
        need = n - filled
        draw = int(need * 2.2) + 16
        a = rng.uniform(KMIN_LOW, KMIN_HIGH, size=draw)
        b = rng.uniform(KMAX_LOW, KMAX_HIGH, size=draw)
        ok = b > (a + KMIN_KMAX_GAP)
        a, b = a[ok], b[ok]
        take = min(len(a), need)
        out_kmin[filled:filled + take] = a[:take]
        out_kmax[filled:filled + take] = b[:take]
        filled += take
    return out_kmin, out_kmax


def create_hdf5_layout(path, n):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "images",
            shape=(n, IMAGE_HEIGHT, IMAGE_WIDTH),
            maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=np.float32,
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
            chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        f["images"].attrs["description"] = "Log10-transformed fractal density slices"
        f["images"].attrs["transform"] = "log10(density_slice)"

        g = f.create_group("parameters")
        g.attrs["description"] = "Per-image generation parameters (continuous floats)"
        for name in ["k_min", "k_max", "sigma", "mean", "beta"]:
            g.create_dataset(name, shape=(n,), maxshape=(None,),
                             dtype=np.float32, compression=COMPRESSION,
                             compression_opts=COMPRESSION_OPTS, chunks=True)
        for name in ["ni", "nj", "nk"]:
            g.create_dataset(name, shape=(n,), maxshape=(None,),
                             dtype=np.int32, compression=COMPRESSION,
                             compression_opts=COMPRESSION_OPTS, chunks=True)
        g.create_dataset("seed", shape=(n,), maxshape=(None,),
                         dtype=np.int64, compression=COMPRESSION,
                         compression_opts=COMPRESSION_OPTS, chunks=True)

        f.attrs["method"] = "lognormal_balanced_4param"
        f.attrs["image_dimensions"] = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]
        f.attrs["batch_size"] = BATCH_SIZE
        f.attrs["total_images"] = n
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["generator"] = "run_generation_balanced.py"
        f.attrs["num_workers"] = NUM_WORKERS
        f.attrs["sampling"] = json.dumps({
            "scheme": "joint_uniform_rejection",
            "k_min_range": [KMIN_LOW, KMIN_HIGH],
            "k_max_range": [KMAX_LOW, KMAX_HIGH],
            "k_min_k_max_gap": KMIN_KMAX_GAP,
            "sigma_range": [SIGMA_LOW, SIGMA_HIGH],
            "beta_range": [BETA_LOW, BETA_HIGH],
            "mean_range": [MEAN_LOW, MEAN_HIGH],
            "continuous_floats": True,
        })


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERF_DIR, exist_ok=True)

    output_path = OUTPUT_FILE if os.path.isabs(OUTPUT_FILE) else os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    nyquist = np.floor(max(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH) / 2.0)
    assert KMAX_HIGH <= nyquist, f"KMAX_HIGH={KMAX_HIGH} > Nyquist={nyquist}"
    assert KMIN_LOW >= 1.0, "k_min must be >= 1"
    assert BETA_HIGH < 0, "beta is the power-spectrum exponent (expected negative)"

    rng = np.random.default_rng()

    print("=" * 70)
    print("BALANCED 4-PARAMETER DATASET GENERATION (k_min, k_max, sigma, beta)")
    print("=" * 70)
    print(f"Output:     {output_path}")
    print(f"Images:     {NUM_IMAGES}")
    print(f"Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_DEPTH} (Nyquist={nyquist})")
    print(f"Workers:    {NUM_WORKERS}    Batch: {BATCH_SIZE}")
    print(f"k_min:      U[{KMIN_LOW}, {KMIN_HIGH}]   (float)")
    print(f"k_max:      U[{KMAX_LOW}, {KMAX_HIGH}]   (float, gap={KMIN_KMAX_GAP})")
    print(f"sigma:      U[{SIGMA_LOW}, {SIGMA_HIGH}]")
    print(f"beta:       U[{BETA_LOW}, {BETA_HIGH}]")
    print("=" * 70)

    create_hdf5_layout(output_path, NUM_IMAGES)

    batch_times = []
    start = time.time()
    idx = 0

    with h5py.File(output_path, "a") as h5f:
        images_ds = h5f["images"]
        pg = h5f["parameters"]

        with mp.Pool(processes=NUM_WORKERS) as pool:
            num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE

            for b in range(num_batches):
                t0 = time.time()
                cur = min(BATCH_SIZE, NUM_IMAGES - idx)

                kmin, kmax = sample_joint_uniform(cur, rng)
                sigma = rng.uniform(SIGMA_LOW, SIGMA_HIGH, size=cur)
                beta  = rng.uniform(BETA_LOW,  BETA_HIGH,  size=cur)
                mean  = rng.uniform(MEAN_LOW,  MEAN_HIGH,  size=cur)
                seeds = rng.integers(0, 2**31, size=cur, dtype=np.int64)

                work = [
                    (float(kmin[i]), float(kmax[i]),
                     IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH,
                     float(mean[i]), float(sigma[i]), float(beta[i]),
                     False, int(seeds[i]))
                    for i in range(cur)
                ]
                results = pool.starmap(_generate_image_worker, work)

                batch_imgs = np.array([img for img, _ in results], dtype=np.float32)
                end = idx + cur
                images_ds[idx:end] = batch_imgs

                cols = {k: [] for k in ["k_min", "k_max", "sigma", "mean", "beta",
                                        "ni", "nj", "nk", "seed"]}
                for _, p in results:
                    for k in cols:
                        cols[k].append(p[k])
                for k, v in cols.items():
                    pg[k][idx:end] = v

                idx = end
                dt = time.time() - t0
                batch_times.append(dt)
                elapsed = time.time() - start
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (NUM_IMAGES - idx) / rate if rate > 0 else 0
                print(f"Batch {b+1:4d}/{num_batches} | {idx:6d}/{NUM_IMAGES} "
                      f"({100*idx/NUM_IMAGES:5.1f}%) | {rate:7.1f} img/s | "
                      f"ETA {eta/60:6.1f} min")

    total = time.time() - start
    size_mb = os.path.getsize(output_path) / (1024**2)

    perf = {
        "run_type": "balanced_4param",
        "output_file": os.path.basename(output_path),
        "num_images": NUM_IMAGES,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
        "sampling": {
            "scheme": "joint_uniform_rejection",
            "k_min_range": [KMIN_LOW, KMIN_HIGH],
            "k_max_range": [KMAX_LOW, KMAX_HIGH],
            "k_min_k_max_gap": KMIN_KMAX_GAP,
            "sigma_range": [SIGMA_LOW, SIGMA_HIGH],
            "beta_range": [BETA_LOW, BETA_HIGH],
            "mean_range": [MEAN_LOW, MEAN_HIGH],
        },
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "time_seconds": round(total, 2),
        "time_minutes": round(total / 60, 2),
        "images_per_second": round(NUM_IMAGES / total, 2) if total > 0 else 0,
        "avg_batch_seconds": round(float(np.mean(batch_times)), 3) if batch_times else 0,
        "file_size_mb": round(size_mb, 2),
        "created_at": datetime.now().isoformat(),
    }
    perf_path = os.path.join(PERF_DIR, f"balanced_4param_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}_perf.json")
    with open(perf_path, "w") as f:
        json.dump(perf, f, indent=2)

    print("=" * 70)
    print(f"OK  {total/60:.1f} min   {size_mb:.0f} MB   {NUM_IMAGES/total:.1f} img/s")
    print(f"    {output_path}")
    print(f"    {perf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
