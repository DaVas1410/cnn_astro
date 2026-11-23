#!/usr/bin/env python3
"""Compare classical (fft-band, fft-powerlaw) and CNN k_min/k_max on real GASS clouds.

Runs `turbulens.spectral`'s two classical estimators and the archived
ResNet50 RMSE regressors (the same models used in
``observational_images/io-fits.ipynb``) against each of the four real
GASS HI cubes in ``observational_images/``, on the velocity-integrated
("full integration") 2D image. There is no ground truth for real
clouds -- this is a side-by-side comparison, not an accuracy check.

`turbulens` itself ships no bundled trained ensemble, so the CNN here
is the same standalone ResNet50Regressor from the notebook, not
`turbulens.inference.Inferencer` (which needs >=2 ensemble members in
turbulens' own checkpoint format; none exist locally).

Run:
    python scripts/compare_real_clouds.py
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.wcs import FITSFixedWarning
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turbulens.io.image_loaders import load_input
from turbulens.spectral import estimate_band_edges, fit_inertial_range

warnings.filterwarnings("ignore", category=FITSFixedWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUD_FILES = sorted(glob.glob(os.path.join(PROJECT_ROOT, "observational_images", "*.fits.gz")))

MODEL_BASE = os.path.join(PROJECT_ROOT, "outputs", "archive", "comparison", "loss_benchmark")
KMIN_MODEL_PATH = os.path.join(MODEL_BASE, "k_min", "rmse_mae", "best_model.pt")
KMAX_MODEL_PATH = os.path.join(MODEL_BASE, "k_max", "rmse", "best_model.pt")

# Training normalization from io-fits.ipynb: per-image min-max -> [0, 1], resize to 128x128.
KMIN_MIN, KMIN_MAX = 1.0, 62.0
KMAX_MIN, KMAX_MAX = 5.0, 64.0
TRAIN_SIZE = 128
DEVICE = torch.device("cpu")


class ResNet50Regressor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


def _preprocess(image2d: np.ndarray) -> torch.Tensor:
    img = image2d.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    if t.shape[-2:] != (TRAIN_SIZE, TRAIN_SIZE):
        t = F.interpolate(t, size=(TRAIN_SIZE, TRAIN_SIZE), mode="bilinear", align_corners=False)
    return t.to(DEVICE)


@torch.no_grad()
def _cnn_predict(model_kmin, model_kmax, image2d: np.ndarray) -> dict:
    t = _preprocess(image2d)
    kmin_norm = model_kmin(t).item()
    kmax_norm = model_kmax(t).item()
    return {
        "k_min": kmin_norm * (KMIN_MAX - KMIN_MIN) + KMIN_MIN,
        "k_max": kmax_norm * (KMAX_MAX - KMAX_MIN) + KMAX_MIN,
    }


def main():
    if not (os.path.exists(KMIN_MODEL_PATH) and os.path.exists(KMAX_MODEL_PATH)):
        print(f"Missing CNN checkpoints; expected:\n  {KMIN_MODEL_PATH}\n  {KMAX_MODEL_PATH}")
        sys.exit(1)

    model_kmin = ResNet50Regressor(in_channels=1).to(DEVICE)
    model_kmax = ResNet50Regressor(in_channels=1).to(DEVICE)
    model_kmin.load_state_dict(torch.load(KMIN_MODEL_PATH, map_location=DEVICE))
    model_kmax.load_state_dict(torch.load(KMAX_MODEL_PATH, map_location=DEVICE))
    model_kmin.eval()
    model_kmax.eval()

    header = (
        f"{'Cloud':<28} {'method':<12} {'k_min':>8} {'k_max':>8}   {'extra':<28}"
    )
    print(header)
    print("-" * len(header))

    for path in CLOUD_FILES:
        name = os.path.basename(path).replace(".fits.gz", "")
        cube = load_input(path)
        projected = cube.project(0, cube.n_channels)
        image = projected.data[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            band = estimate_band_edges(image)
            band_warned = len(caught) > 0

        powerlaw = fit_inertial_range(image)
        cnn = _cnn_predict(model_kmin, model_kmax, image)

        print(f"{name:<28} {'fft-band':<12} {band['k_min']:>8.1f} {band['k_max']:>8.1f}   "
              f"{'(NOT reliable: no real edge found)' if band_warned else ''}")
        print(f"{'':<28} {'fft-powerlaw':<12} {powerlaw['k_min']:>8.1f} {powerlaw['k_max']:>8.1f}   "
              f"alpha={powerlaw['alpha']:.2f}, R2={powerlaw['r_squared']:.3f}")
        print(f"{'':<28} {'cnn (rmse)':<12} {cnn['k_min']:>8.1f} {cnn['k_max']:>8.1f}")
        print()


if __name__ == "__main__":
    main()
