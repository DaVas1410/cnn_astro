"""Accuracy regression tests for `turbulens.spectral` against labeled synthetic data.

Unlike `test_spectral.py`'s hand-synthesized images, these tests measure
`estimate_band_edges` against real pyFC-generated images with known
ground-truth k_min/k_max labels (``data/raw/*.h5``), the same data the
CNN ensembles are trained on. `data/raw/*.h5` is gitignored (large,
locally generated), so these tests skip cleanly when it is absent
rather than failing CI or a fresh checkout.

To re-tune the estimator's parameters against this or other labeled
data, see ``archive/scripts/tune_band_edges.py``.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from turbulens.spectral import estimate_band_edges

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELED_DATASET = os.path.join(PROJECT_ROOT, "data", "raw", "flat_4param_128x128_2000_disjoint_val.h5")

pytestmark = pytest.mark.skipif(
    not os.path.exists(LABELED_DATASET), reason=f"labeled dataset not present: {LABELED_DATASET}",
)


def _r2(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _load_sample(n_sample=300, seed=0):
    with h5py.File(LABELED_DATASET, "r") as f:
        n = f["images"].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
        images = f["images"][idx]
        k_min = f["parameters/k_min"][idx].astype(float)
        k_max = f["parameters/k_max"][idx].astype(float)
    return images, k_min, k_max


def test_estimate_band_edges_recovers_kmin_with_high_r2():
    images, k_min_true, _ = _load_sample()

    k_min_pred = np.array([estimate_band_edges(image)["k_min"] for image in images])

    assert _r2(k_min_true, k_min_pred) > 0.99


def test_estimate_band_edges_recovers_kmax_with_high_r2():
    images, _, k_max_true = _load_sample()

    k_max_pred = np.array([estimate_band_edges(image)["k_max"] for image in images])

    assert _r2(k_max_true, k_max_pred) > 0.85


def test_estimate_band_edges_kmin_mae_within_one_pixel():
    images, k_min_true, _ = _load_sample()

    k_min_pred = np.array([estimate_band_edges(image)["k_min"] for image in images])

    assert np.mean(np.abs(k_min_true - k_min_pred)) < 1.0


def test_estimate_band_edges_kmax_mae_within_two_pixels():
    images, _, k_max_true = _load_sample()

    k_max_pred = np.array([estimate_band_edges(image)["k_max"] for image in images])

    assert np.mean(np.abs(k_max_true - k_max_pred)) < 2.0


def test_estimate_band_edges_never_returns_nan_on_real_labeled_images():
    # Every image in this dataset is genuinely band-limited by construction (pyFC), so
    # the estimator -- built for exactly this kind of image -- should always find an edge.
    images, _, _ = _load_sample(n_sample=300)

    n_nan = sum(1 for image in images if np.isnan(estimate_band_edges(image)["k_min"]))

    assert n_nan == 0
