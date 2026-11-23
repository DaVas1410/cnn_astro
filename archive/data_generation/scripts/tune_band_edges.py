#!/usr/bin/env python3
"""Tune `turbulens.spectral.estimate_band_edges` against labeled synthetic data.

pyFC-generated images carry ground-truth k_min/k_max labels
(``data/raw/*.h5``, gitignored -- run ``dvc pull`` or regenerate via
``src/param_sampler.py`` if missing). This sweeps the estimator's
``lo_frac``/``noise_multiplier`` over a grid, scores each combination by
R² and MAE against those labels, and prints a ranked table -- the
inputs for choosing (or re-choosing) the function's default parameters.

Run:
    python scripts/tune_band_edges.py [--file PATH] [--n-sample N]
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turbulens.spectral import estimate_band_edges

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "flat_4param_128x128_2000_disjoint_val.h5")

LO_FRAC_GRID = [0.1, 0.2, 0.3]
NOISE_PERCENTILE_GRID = [1.0, 2.0, 3.0, 5.0]
NOISE_MULTIPLIER_GRID = [1.0, 1.5, 2.0, 3.0]


def _r2(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def load_sample(path: str, n_sample: int, seed: int = 0):
    with h5py.File(path, "r") as f:
        n = f["images"].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
        images = f["images"][idx]
        k_min = f["parameters/k_min"][idx].astype(float)
        k_max = f["parameters/k_max"][idx].astype(float)
    return images, k_min, k_max


def score(images, k_min_true, k_max_true, lo_frac, noise_percentile, noise_multiplier):
    k_min_pred, k_max_pred = [], []
    n_nan = 0
    for image in images:
        edges = estimate_band_edges(
            image, lo_frac=lo_frac, noise_percentile=noise_percentile, noise_multiplier=noise_multiplier,
        )
        if not (np.isfinite(edges["k_min"]) and np.isfinite(edges["k_max"])):
            n_nan += 1
            k_min_pred.append(np.nan)
            k_max_pred.append(np.nan)
        else:
            k_min_pred.append(edges["k_min"])
            k_max_pred.append(edges["k_max"])
    k_min_pred, k_max_pred = np.array(k_min_pred), np.array(k_max_pred)
    valid = np.isfinite(k_min_pred) & np.isfinite(k_max_pred)
    return {
        "lo_frac": lo_frac,
        "noise_percentile": noise_percentile,
        "noise_multiplier": noise_multiplier,
        "n_nan": n_nan,
        "kmin_r2": _r2(k_min_true[valid], k_min_pred[valid]),
        "kmax_r2": _r2(k_max_true[valid], k_max_pred[valid]),
        "kmin_mae": float(np.mean(np.abs(k_min_true[valid] - k_min_pred[valid]))),
        "kmax_mae": float(np.mean(np.abs(k_max_true[valid] - k_max_pred[valid]))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=DEFAULT_FILE, help="Labeled HDF5 dataset with images + k_min/k_max.")
    parser.add_argument("--n-sample", type=int, default=300, help="Number of images to sample.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Dataset not found: {args.file}\nThese .h5 files are gitignored; regenerate or fetch them first.")
        sys.exit(1)

    images, k_min_true, k_max_true = load_sample(args.file, args.n_sample)
    print(f"Loaded {len(images)} images from {os.path.basename(args.file)}")
    print(f"k_min true range: [{k_min_true.min():.1f}, {k_min_true.max():.1f}]")
    print(f"k_max true range: [{k_max_true.min():.1f}, {k_max_true.max():.1f}]\n")

    results = []
    for lo_frac in LO_FRAC_GRID:
        for noise_percentile in NOISE_PERCENTILE_GRID:
            for noise_multiplier in NOISE_MULTIPLIER_GRID:
                results.append(score(images, k_min_true, k_max_true, lo_frac, noise_percentile, noise_multiplier))

    results.sort(key=lambda r: -(r["kmin_r2"] + r["kmax_r2"]))

    header = (
        f"{'lo_frac':>8} {'noise_pct':>9} {'noise_mult':>10} {'n_nan':>6} "
        f"{'kmin_R2':>8} {'kmax_R2':>8} {'kmin_MAE':>9} {'kmax_MAE':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results[:20]:
        print(
            f"{r['lo_frac']:>8.2f} {r['noise_percentile']:>9.1f} {r['noise_multiplier']:>10.1f} {r['n_nan']:>6} "
            f"{r['kmin_r2']:>8.3f} {r['kmax_r2']:>8.3f} {r['kmin_mae']:>9.2f} {r['kmax_mae']:>9.2f}"
        )

    best = results[0]
    print(
        f"\nBest: lo_frac={best['lo_frac']}, noise_percentile={best['noise_percentile']}, "
        f"noise_multiplier={best['noise_multiplier']} "
        f"(kmin_R2={best['kmin_r2']:.3f}, kmax_R2={best['kmax_r2']:.3f})"
    )


if __name__ == "__main__":
    main()
