# Runbook — retrain on the k_max-fixed dataset (ready 2026-06-25, for next session)

Everything below is already prepared. This is the step-by-step for the retraining
session.

## What was done (this session)

- **Found & fixed** the pyFC `k_max` bug (`docs/bugs/pyfc-kmax-not-enforced.md`):
  `LogNormalFractalCube.func_target_spec` now band-limits the spectrum to
  `[k_min, k_max]`. Regression test: `tests/test_pyfc_band_limit.py` (3 pass).
- **Regenerated** the full dataset with the fix:
  `data/raw/balanced_4param_128x128_100000_kmaxfix.h5` (same sampling/config as
  the original; the old broken file is kept for comparison).
- **Prepared** `notebooks/comparison/joint_regression_v3.ipynb` — a copy of v2
  wired to the new dataset and `outputs/comparison/joint_regression_v3/`.
- **Fixed** the classical estimator in `notebooks/comparison/synthetic_validation.ipynb`
  to band-edge detection (classical k_min R²≈1.0; k_max recovers on fixed data).
- **Documented** `observational_images/io-fits.ipynb` (bug status + which method
  to use for real clouds).

## Steps for tomorrow

### 1. Confirm the dataset is present
```bash
.venv_py311/Scripts/python.exe -c "import h5py; f=h5py.File('data/raw/balanced_4param_128x128_100000_kmaxfix.h5'); print(f['images'].shape, list(f['parameters']))"
```
Expect `(100000, 128, 128)` and the four parameters.

### 2. Train the joint model (v3)
Open and run `notebooks/comparison/joint_regression_v3.ipynb` top-to-bottom with
the py311 kernel. It writes `outputs/comparison/joint_regression_v3/best_model.pt`
and `results.json`. (If a `results.json` is already there, the training cell
skips — delete it to force a fresh run.)

Expected: **k_max metrics improve markedly** vs v2 (v2 broken-data baseline was
k_min R²≈0.998, k_max R²≈0.695, sigma R²≈0.989).

### 3. Re-run synthetic validation against v3
In `notebooks/comparison/synthetic_validation.ipynb`, point the setup cell at the
new data and v3 checkpoint (edit these three constants):
```python
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'balanced_4param_128x128_100000_kmaxfix.h5'
CKPT      = PROJECT_ROOT / 'outputs' / 'comparison' / 'joint_regression_v3' / 'best_model.pt'
RESULTS   = PROJECT_ROOT / 'outputs' / 'comparison' / 'joint_regression_v3' / 'results.json'
```
Then run it. Expect **classical k_max R²≈0.98** (was negative on the broken data)
and improved CNN k_max.

### 4. (Optional) Real-cloud inference
In `observational_images/io-fits.ipynb`, load the joint v3 model
(`ResNet50Joint4`) from the v3 checkpoint for k_min/k_max/sigma/beta. Keep the
inertial-range `fourier_params` as the classical estimator for **real** clouds
(do NOT use band-edge on real data — see the note cell at the top of that
notebook).

## Notes
- Interpreter for everything: `.venv_py311/Scripts/python.exe` (torch 2.6 + CUDA).
  Run notebooks headless with
  `.venv_py311/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace <nb>`.
- The 3000-image pilot used for verification is at
  `data/raw/pilot_kmax_fixed_128x128_3000.h5` (safe to delete once the full run is
  trusted).
