#!/usr/bin/env python3
"""Verify k_min/k_max labels against the azimuthal Fourier spectrum band edges.

Reuses the band_edge_params estimator validated in synthetic_validation. pyFC
fractal images are band-limited: power lives within [k_min, k_max]. This script
measures those band edges from each image's 2D FFT and compares them to the
stored labels, and confirms the pyFC kmax band-limit fix by checking that DB1's
measured cutoff clusters near 128 and DB2's near 96.

Run:
    .venv_py311/Scripts/python.exe scripts/verify_spectra.py
"""
import json
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "spectra_check")


def _radial_profile(img2d):
    """Azimuthally averaged power spectrum P(k) of a 2D image."""
    F = fftshift(fft2(img2d.astype(np.float64)))
    P = np.abs(F) ** 2
    ny, nx = P.shape
    cy, cx = ny // 2, nx // 2
    y_i, x_i = np.mgrid[:ny, :nx]
    r_i = np.sqrt((x_i - cx) ** 2 + (y_i - cy) ** 2).astype(int)
    km = min(cx, cy)
    kr = np.arange(1, km)
    Pr = np.array([P[r_i == k].mean() if (r_i == k).any() else 0.0 for k in kr])
    return kr, Pr


def band_edge_params(img2d, lo_frac=0.1, hi_floor=1e-12):
    """k_min = lowest k with power >= lo_frac*peak; k_max = highest k > hi_floor*peak."""
    kr, Pr = _radial_profile(img2d)
    pk = Pr.max()
    if pk <= 0:
        return {"k_min": np.nan, "k_max": np.nan}
    Pn = Pr / pk
    lo = np.where(Pn >= lo_frac)[0]
    hi = np.where(Pn > hi_floor)[0]
    if len(lo) == 0 or len(hi) == 0:
        return {"k_min": np.nan, "k_max": np.nan}
    return {"k_min": float(kr[lo[0]]), "k_max": float(kr[hi[-1]])}


def _r2(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def verify_database(h5_path, n_sample=200, target_kmax=None, seed=0):
    """Measure band edges on a random sample and compare to stored labels."""
    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
        imgs = f["images"][idx]
        lab_kmin = f["parameters/k_min"][idx].astype(float)
        lab_kmax = f["parameters/k_max"][idx].astype(float)

    meas_kmin, meas_kmax = [], []
    for img in imgs:
        fp = band_edge_params(img)
        meas_kmin.append(fp["k_min"])
        meas_kmax.append(fp["k_max"])
    meas_kmin = np.array(meas_kmin, float)
    meas_kmax = np.array(meas_kmax, float)
    valid = np.isfinite(meas_kmin) & np.isfinite(meas_kmax)

    mk, mx = meas_kmin[valid], meas_kmax[valid]
    lk, lx = lab_kmin[valid], lab_kmax[valid]
    kmax_median = float(np.median(mx))
    sanity_pass = bool(target_kmax is None or abs(kmax_median - target_kmax) <= 3)
    return {
        "file": os.path.basename(h5_path),
        "n_sampled": int(valid.sum()),
        "target_kmax": target_kmax,
        "kmin_mae": float(np.mean(np.abs(mk - lk))),
        "kmax_mae": float(np.mean(np.abs(mx - lx))),
        "kmin_r2": _r2(lk, mk),
        "kmax_r2": _r2(lx, mx),
        "kmin_within2_frac": float(np.mean(np.abs(mk - lk) <= 2)),
        "kmax_within2_frac": float(np.mean(np.abs(mx - lx) <= 2)),
        "kmax_median": kmax_median,
        "sanity_pass": sanity_pass,
        "_idx": idx.tolist(),
        "_meas_kmin": mk.tolist(),
        "_meas_kmax": mx.tolist(),
        "_lab_kmin": lk.tolist(),
        "_lab_kmax": lx.tolist(),
    }


def _plot_examples(h5_path, target_kmax, stem, n_examples=3, seed=1):
    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=n_examples, replace=False))
        imgs = f["images"][idx]
        kmins = f["parameters/k_min"][idx].astype(float)
    fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]
    for ax, img, kmin in zip(axes, imgs, kmins):
        kr, Pr = _radial_profile(img)
        ax.loglog(kr, Pr / Pr.max(), color="#1565C0")
        ax.axvline(kmin, color="green", ls="--", label=f"k_min={kmin:.0f}")
        ax.axvline(target_kmax, color="red", ls="--", label=f"k_max={target_kmax}")
        ax.set_xlabel("k")
        ax.set_ylabel("P(k)/peak")
        ax.legend(fontsize=8)
    fig.suptitle(f"{stem}: azimuthal power spectra")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{stem}_spectra_examples.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = os.path.join(PROJECT_ROOT, "data", "raw")
    dbs = [
        ("db1_kmax128_256x256_1000.h5", 128),
        ("db2_kmax96_256x256_1000.h5", 96),
    ]
    all_ok = True
    for fname, target in dbs:
        path = os.path.join(raw, fname)
        stem = fname.replace(".h5", "")
        summary = verify_database(path, target_kmax=target)
        _plot_examples(path, target, stem)
        out_json = os.path.join(OUT_DIR, f"{stem}_verify.json")
        with open(out_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        all_ok = all_ok and summary["sanity_pass"]
        print(
            f"{stem}: kmin_R2={summary['kmin_r2']:.3f} kmax_R2={summary['kmax_r2']:.3f} "
            f"kmin_MAE={summary['kmin_mae']:.2f} kmax_MAE={summary['kmax_mae']:.2f} "
            f"kmax_median={summary['kmax_median']:.1f} (target {target}) "
            f"sanity={'PASS' if summary['sanity_pass'] else 'FAIL'}"
        )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
