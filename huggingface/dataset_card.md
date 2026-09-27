---
license: mit
tags:
  - astronomy
  - turbulence
  - synthetic
  - regression
size_categories:
  - 100K<n<1M
---

# turbulens synthetic fractal-turbulence dataset

170,000 synthetic 128x128 log-normal fractal density images (100,000 train / 20,000 validation /
50,000 test), generated with [pyFC](https://bitbucket.org/pandante/pyfc), labeled with the four
generating parameters `k_min`, `k_max`, `sigma`, `beta`. This is the training data for the
[`turbulens`](<MODEL_REPO_ID>) deep-ensemble CNN parameter-estimation models.

## Files

```
train/flat_4param_128x128_100000_disjoint_train.h5   (100,000 images)
val/flat_4param_128x128_20000_disjoint_val.h5         (20,000 images)
test/flat_4param_128x128_50000_disjoint_test.h5       (50,000 images)
```

Each `.h5` file has:

- `images`: `(N, 128, 128)` float32, log10-transformed fractal density slices
  (`log10(density_slice)`).
- `parameters/k_min`, `k_max`, `sigma`, `mean`, `beta`: `(N,)` float32, the per-image generating
  parameters.
- `parameters/ni`, `nj`, `nk`: `(N,)` int32, the pyFC cube dimensions used.
- `parameters/seed`: `(N,)` int64, the per-image pyFC generation seed.
- File-level attrs: `method`, `image_dimensions`, `batch_size`, `total_images`, `num_workers`,
  `sampling` (JSON: scheme + all sampling ranges), `master_seed`.

```python
import h5py
with h5py.File("train/flat_4param_128x128_100000_disjoint_train.h5") as f:
    images = f["images"][:]          # (100000, 128, 128) float32
    k_min = f["parameters/k_min"][:]  # (100000,) float32
```

## Generation

Parameters are drawn independently and uniformly per image (no two images share a parameter vector):

- `k_min ~ U[1.0, 32.0]`
- `k_max ~ U[34.0, 64.0]` (disjoint from `k_min`'s range with a 2.0 gap, so `k_max > k_min` always
  holds and both marginals stay exactly flat — no rejection sampling, no order-statistic skew)
- `sigma ~ U[0.01, 5.0]`
- `beta ~ U[-3.0, -1.0]` (the power-spectrum exponent, `D(k) ~ k^beta`; covers Kolmogorov-like,
  Burgers-like, and Kraichnan-like cascades)
- `mean = 1.0` (fixed)

`k_max`'s upper bound (64) is the Nyquist frequency of the 128-pixel grid. Each split used its own
master seed so parameter draws and pyFC seeds never overlap between splits: 1000 (train), 2000
(validation), 3000 (test). Reproduce with `archive/scripts/gen_splits_disjoint.sh` in the source repository.

**Provenance note:** an earlier version of this dataset's own generation had a `pyFC` bug where the
upper spectral cutoff was not enforced (`k_max` was label-only, not an actual image feature) — see
`archive/docs/bugs/pyfc-kmax-not-enforced.md` in the source repo. This release was generated after that fix
(`archive/src/pyFC_lib/pyFC/clouds.py`'s `func_target_spec` band-limits to `[k_min, k_max]`), matching the data
the published `turbulens` model ensembles were trained on.

## Splits

Train/validation/test are separate files with disjoint master seeds, not a single file split after
the fact — there is no leakage between them by construction.

## License

MIT.
