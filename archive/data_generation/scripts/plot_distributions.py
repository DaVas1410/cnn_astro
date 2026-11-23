#!/usr/bin/env python3
"""Plot parameter distributions for the two databases and measured-vs-label scatter.

Reads the two databases' /parameters and the *_verify.json summaries written by
verify_spectra.py. Produces:
  outputs/spectra_check/param_distributions.png  (k_min / k_max / sigma histograms)
  outputs/spectra_check/measured_vs_label.png    (measured vs label scatter)

Run (after verify_spectra.py):
    .venv_py311/Scripts/python.exe scripts/plot_distributions.py
"""
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "spectra_check")

DBS = [
    ("db1_kmax128_256x256_1000.h5", "DB1 (kmax=128)", "#1565C0"),
    ("db2_kmax96_256x256_1000.h5", "DB2 (kmax=96)", "#C62828"),
]


def _load_params(fname):
    with h5py.File(os.path.join(RAW, fname), "r") as f:
        return {
            "k_min": f["parameters/k_min"][:].astype(float),
            "k_max": f["parameters/k_max"][:].astype(float),
            "sigma": f["parameters/sigma"][:].astype(float),
        }


def plot_param_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for fname, label, color in DBS:
        p = _load_params(fname)
        axes[0].hist(p["k_min"], bins=32, range=(1, 33), alpha=0.5, label=label, color=color)
        axes[1].hist(p["k_max"], bins=60, range=(80, 140), alpha=0.6, label=label, color=color)
        axes[2].hist(p["sigma"], bins=20, range=(1.5, 2.5), alpha=0.5, label=label, color=color)
    axes[0].set_title("k_min  (U(1,32) per image)")
    axes[1].set_title("k_max  (fixed per DB)")
    axes[2].set_title("sigma  (fixed = 2.0)")
    for ax in axes:
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("Parameter distributions: DB1 vs DB2")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "param_distributions.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_measured_vs_label():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for fname, label, color in DBS:
        stem = fname.replace(".h5", "")
        with open(os.path.join(OUT_DIR, f"{stem}_verify.json")) as fh:
            s = json.load(fh)
        axes[0].scatter(s["_lab_kmin"], s["_meas_kmin"], s=8, alpha=0.5, color=color, label=label)
        axes[1].scatter(s["_lab_kmax"], s["_meas_kmax"], s=8, alpha=0.5, color=color, label=label)
    for ax, name, lim in zip(axes, ["k_min", "k_max"], [(0, 35), (60, 140)]):
        ax.plot(lim, lim, "k--", lw=1, label="ideal")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"label {name}")
        ax.set_ylabel(f"measured {name}")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle("Measured spectral band edge vs stored label")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "measured_vs_label.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("wrote", plot_param_distributions())
    print("wrote", plot_measured_vs_label())


if __name__ == "__main__":
    main()
