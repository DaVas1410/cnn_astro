---
license: mit
tags:
  - astronomy
  - turbulence
  - pytorch
  - resnet
  - regression
  - uncertainty-quantification
library_name: pytorch
pipeline_tag: image-to-text
---

# turbulens: fractal-turbulence parameter ensembles

Five deep-ensemble PyTorch models that predict fractal-turbulence parameters — `k_min`, `k_max`,
`sigma`, `beta` — from a 2D radio-astronomy image (a projected HI cube, or a synthetic log-normal
fractal density slice). Use them with the [`turbulens`](https://github.com/wbandabarragan/cnn-turbulence)
Python package's `turbulens infer` CLI or `turbulens.inference.Inferencer` API — this repo is weights
only, not a standalone inference script.

## What's here

One **multitask** ensemble (one shared ResNet backbone + trunk, predicting all four targets jointly)
and four **single-target** ensembles (`k_min`, `k_max`, `sigma`, `beta`), each independently trained.
Every ensemble has 5 members (distinct seeds, shared normalization) — a deep ensemble, not a single
model: `Inferencer` requires at least 2 members and reports member-to-member spread as an epistemic
uncertainty estimate alongside the point prediction.

```
multitask/local_v2/ensemble/members/member_0{1..5}_seed{42,123,456,789,2026}/
single_k_min/local_v2/ensemble/members/member_0{1..5}_seed{...}/
single_k_max/local_v2/ensemble/members/member_0{1..5}_seed{...}/
single_sigma/local_v2/ensemble/members/member_0{1..5}_seed{...}/
single_beta/local_v2/ensemble/members/member_0{1..5}_seed{...}/
```

This mirrors the layout `turbulens.models.registry.EnsembleRegistry` expects, so a plain download of
this repo can be used directly as `--output-root`.

## Use

```bash
pip install "turbulens[image] @ git+https://github.com/wbandabarragan/cnn-turbulence.git#subdirectory=."
huggingface-cli download <REPO_ID> --local-dir ./turbulens-ensembles

# Synthetic image (already on the training log-density scale)
turbulens infer --input slice.npy --method cnn --model auto \
    --output-root ./turbulens-ensembles --out predictions.ecsv

# Real telescope observation -- --real-data is required, see turbulens --help
turbulens infer --input observation.fits --integrate 0:100 --real-data \
    --method cnn --model auto --output-root ./turbulens-ensembles --target k_max
```

Or as a library:

```python
from turbulens.inference import Inferencer
table = Inferencer("./turbulens-ensembles/multitask/local_v2/ensemble/members").predict(image, real_data=True)
```

## Training data

Trained on synthetic 128x128 log-normal fractal density slices generated with
[pyFC](https://bitbucket.org/pandante/pyfc): 100,000 train / 20,000 validation / 50,000 test images,
parameters drawn independently per image, `k_min ~ U[1, 32]`, `k_max ~ U[34, 64]` (disjoint from
`k_min`'s range, so `k_max > k_min` always holds), `sigma ~ U[0.01, 5.0]`, `beta ~ U[-3.0, -1.0]`. See
the companion dataset repo `<DATASET_REPO_ID>` for the exact data and generation config.

## Test-set performance (multitask ensemble, point estimates)

| target | MAE | RMSE | R^2 |
|---|---|---|---|
| k_min | 0.119 | 0.153 | 0.9997 |
| k_max | 0.602 | 1.239 | 0.9795 |
| sigma | 0.034 | 0.045 | 0.9990 |
| beta | 0.111 | 0.163 | 0.9203 |

Single-target ensembles are marginally (but statistically significantly) better than multitask on
`k_min`/`k_max`/`sigma`, and multitask is clearly better on `beta` (R^2 0.920 vs 0.908), at roughly 1/5
the training cost and 1/4 the parameters of running all four single-target ensembles. This is why both
are shipped rather than one being deprecated in favor of the other.

## Limitations

- **Epistemic uncertainty is under-dispersed.** At the nominal 68% confidence level, actual empirical
  coverage on the held-out test set ranges from 27.6% (`beta`) to 62.5% (`k_max`) across targets — the
  ensemble spread systematically underestimates true prediction error. Do not treat the reported
  `epistemic_std` as a calibrated confidence interval; use it only as a relative, cross-image
  uncertainty ranking.
- **No aleatoric (measurement) uncertainty term.** These are point-estimate regression heads
  (`nn.SmoothL1Loss`), not a variance-head/NLL model; all reported uncertainty is ensemble
  (epistemic) spread only.
- **Trained entirely on synthetic pyFC data.** Applying to real telescope data (e.g. GASS HI survey)
  requires `--real-data`, which recenters onto the training scale — it does not correct for any other
  real-vs-synthetic distribution shift (beam effects, noise properties, non-log-normal density
  statistics, etc.). `Inferencer.predict` warns when >5% of input pixels hit the normalization clip
  boundary, a signal of likely out-of-distribution input.
- **`k_max` is the hardest target** — a spectral-tail property, inherently noisier to estimate from a
  single finite image than a large-scale property like `k_min`.

## License

MIT. See the source repository's `LICENSE`.
