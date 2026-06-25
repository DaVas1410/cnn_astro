# BUG: pyFC `LogNormalFractalCube` ignores `kmax` (no upper spectral cutoff)

**Status:** Open — fix identified & verified, not yet applied
**Severity:** High — corrupts the `k_max` label of every LogNormal dataset
**Found:** 2026-06-25, during synthetic validation of the joint-regression CNN

## Summary

`LogNormalFractalCube.func_target_spec` enforces only the **lower** wavenumber
cutoff `kmin`. The upper cutoff `kmax` is commented out (`# Doesn't work yet`),
so the generated power spectrum is `k^(beta-2)` for **all** `k >= kmin` with no
band limit. As a result `k_max` has essentially **no causal effect** on the
generated images, even though it is sampled and stored as a label.

## Location

`src/pyFC_lib/pyFC/clouds.py`, `LogNormalFractalCube.func_target_spec`
(lines ~1217–1238):

```python
# Doesn't work yet
# return np.where(np.logical_and(np.greater_equal(np.abs(k), kmin),
#                                np.less_equal(np.abs(k), kmax)),
#                 np.abs(k) ** (beta - 2.), 0)

return np.where(np.greater_equal(np.abs(k), kmin), np.abs(k) ** (beta - 2.), 0)
```

The sibling class `GaussianFractalCube.func_target_spec` (line ~1097) already
has the **correct** band-limited version active — proof the fix is valid:

```python
return np.where(np.logical_and(np.greater_equal(np.abs(k), kmin),
                             np.less_equal(np.abs(k), kmax)),
              np.abs(k) ** (beta - 2.), 0)
```

The broken code predates all data: `clouds.py` has a single commit
(Initial commit, 2026-05-06), well before `balanced_4param_128x128_100000.h5`
was generated (2026-05-26). The dataset generator
(`src/dataset_generator.py:86`) uses `LogNormalFractalCube`, so it hit the bug.

## Evidence

Measured with the imported pyFC and the dataset
`data/raw/balanced_4param_128x128_100000.h5`:

- **Target spectrum ignores kmax:** for `kmin=10, kmax=20`, `func_target_spec`
  returns nonzero power above kmax (`spec[k=20]=1.7e-5`, `spec[k=30]=3.8e-6`) —
  pure power law, no cutoff.
- **Classical radial power spectrum is blind to kmax:** with `kmin` fixed in
  [4,6] (residual kmin–kmax corr ≈ 0.07), spectral upper-edge features correlate
  only ≈ 0.12–0.25 with true `kmax`.
- **CNN recovers kmax only weakly** (the incidental imprint): at fixed kmin the
  joint CNN's `pred_kmax` vs true `kmax` is R² ≈ 0.47–0.74 (overall R² = 0.695),
  vs k_min R² ≈ 0.998 and sigma R² ≈ 0.989. The k_max ceiling is the data, not
  the model.
- **The fix works:** monkey-patching the band-limited target spec and
  regenerating a cube yields a **sharp** 2D-slice cutoff at kmax (e.g. kmin=15,
  kmax=35: P(k=34)=1.8e-2 → P(k=35)=8.7e-5 → P(k≥36)≈1e-33).

## Fix

In `LogNormalFractalCube.func_target_spec`, replace the kmin-only return with the
band-limited version (mirroring `GaussianFractalCube`):

```python
return np.where(np.logical_and(np.greater_equal(np.abs(k), kmin),
                               np.less_equal(np.abs(k), kmax)),
                np.abs(k) ** (beta - 2.), 0)
```

## Required follow-up after the fix

1. Regenerate the training dataset with the corrected pyFC (k_max becomes a real,
   band-limiting parameter).
2. Retrain the joint regression model; k_max should now be strongly learnable.
3. Re-run `notebooks/comparison/synthetic_validation.ipynb`. With a real kmax
   band cutoff, the classical radial-spectrum estimator should also recover kmax
   (band-edge detection), so the "classical = k_min only" decision below may be
   revisited for the regenerated data.

## Impact on current deliverables

- `outputs/comparison/joint_regression_v2/` (trained CNN) — k_max metrics reflect
  the data defect, not model capacity.
- `notebooks/comparison/synthetic_validation.ipynb` — its classical estimator is
  being fixed to band-edge detection (k_min: excellent; k_max/sigma: CNN-only)
  **for the current dataset**. This pairing is correct only while the dataset has
  the kmax defect.
