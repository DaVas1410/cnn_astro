"""Regression test for the pyFC kmax band-limiting bug.

See docs/bugs/pyfc-kmax-not-enforced.md. LogNormalFractalCube.func_target_spec
must enforce BOTH the lower (kmin) and upper (kmax) wavenumber cutoffs, so the
target power spectrum is zero outside [kmin, kmax]. The bug left only kmin
enforced, making kmax a phantom label.

Run:
    .venv_py311/Scripts/python.exe -m pytest tests/test_pyfc_band_limit.py -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pyFC_lib"))
import pyFC  # noqa: E402


class TestPyFCBandLimit(unittest.TestCase):
    def _cube(self, kmin, kmax):
        return pyFC.LogNormalFractalCube(
            ni=64, nj=64, nk=64, kmin=kmin, kmax=kmax,
            mean=1.0, sigma=1.0, beta=-1.6667,
        )

    def test_target_spec_zero_above_kmax(self):
        """Power must be exactly zero for |k| > kmax."""
        fc = self._cube(kmin=10, kmax=20)
        k = np.arange(1, 33, dtype=float)
        spec = fc.func_target_spec(k)
        above = k > 20
        self.assertTrue(
            np.all(spec[above] == 0),
            msg=f"Nonzero power above kmax=20: {spec[above][spec[above] > 0][:5]}",
        )

    def test_target_spec_zero_below_kmin(self):
        """Power must be exactly zero for |k| < kmin (unchanged behaviour)."""
        fc = self._cube(kmin=10, kmax=20)
        k = np.arange(1, 33, dtype=float)
        spec = fc.func_target_spec(k)
        self.assertTrue(np.all(spec[k < 10] == 0))

    def test_target_spec_nonzero_inside_band(self):
        """Power must be nonzero inside [kmin, kmax]."""
        fc = self._cube(kmin=10, kmax=20)
        k = np.arange(1, 33, dtype=float)
        spec = fc.func_target_spec(k)
        in_band = (k >= 10) & (k <= 20)
        self.assertTrue(np.all(spec[in_band] > 0))


if __name__ == "__main__":
    unittest.main()
