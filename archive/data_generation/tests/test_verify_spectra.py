"""Test the azimuthal band-edge estimator used to verify k_min/k_max labels."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from verify_spectra import band_edge_params  # noqa: E402


def _band_limited_image(n=256, kmin=10, kmax=60, seed=0):
    """Synthesize a real image whose power lives only in [kmin, kmax]."""
    rng = np.random.default_rng(seed)
    kax = np.fft.fftshift(np.fft.fftfreq(n)) * n
    kxg, kyg = np.meshgrid(kax, kax)
    kr = np.sqrt(kxg**2 + kyg**2)
    amp = np.where((kr >= kmin) & (kr <= kmax), np.clip(kr, 1e-6, None) ** (-1.6667 - 2.0), 0.0)
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    F = np.fft.ifftshift(amp * phase)
    img = np.fft.ifft2(F).real
    return img


def test_band_edges_recover_kmin_kmax():
    img = _band_limited_image(kmin=10, kmax=60)
    fp = band_edge_params(img)
    assert abs(fp["k_min"] - 10) <= 3, fp
    assert abs(fp["k_max"] - 60) <= 3, fp
