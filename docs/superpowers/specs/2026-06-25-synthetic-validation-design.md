# Synthetic Validation — Classical vs CNN — Design

Date: 2026-06-25
Status: Approved (design)

## Goal

Demonstrate that the non-Bayesian joint regression CNN (`ResNet50Joint4`)
recovers fractal parameters from **in-distribution synthetic images** with high
accuracy, and contrast it against a classical power-spectrum baseline. This
both validates the model ("synthetic images must give near-exact results because
they come from the same generator the model trained on") and diagnoses where
`k_max` and `sigma` are weakest — input for the *separate, later* second-training
spec.

This spec covers **only** the validation pipeline. The second training
(re-train of the joint head to improve `k_max`/`sigma`) is a follow-up spec to
be written after validation results are available.

## Deliverable

A new notebook: `notebooks/comparison/synthetic_validation.ipynb`.

## Inputs (existing, reused)

- **Trained CNN**: `outputs/comparison/joint_regression_v2/best_model.pt`
  (`ResNet50Joint4`, 4 outputs: `k_min, k_max, sigma, beta`, Sigmoid head).
- **Classical method**: `fourier_params(img2d, ...)` from
  `observational_images/io-fits.ipynb` — azimuthally-averaged FFT power spectrum
  → spectral index `alpha`, `k_min`, `k_max` (in pyFC k-mode units).
- **Synthetic dataset**: `data/raw/balanced_4param_128x128_100000.h5`
  (`/images` (N,128,128) + `/parameters/{k_min,k_max,sigma,beta}`), continuous
  floats, ground truth known.
- **Normalization / denorm constants** copied verbatim from
  `joint_regression_v2.ipynb` (Cell 1/4/6): `IMG_P1, IMG_P99, KMIN_LO/HI,
  KMAX_LO/HI, LOG_SIG_LO/HI, BETA_LO/HI`, and the same `SEED` + 80/10/10
  `train_test_split` so the test split matches the trained model's held-out set.

## Components & data flow

### 1. Load synthetic test set
Reproduce the exact split from `joint_regression_v2.ipynb`: same `SEED`,
`train_test_split(indices, test_size=0.2)` then split the remainder 50/50 to get
`test_idx`. Evaluate on a configurable subset `N_EVAL` (default 2000) of
`test_idx` for speed. These are images the model did **not** train on — this is
honest validation, not memorization.

### 2. CNN inference
Load the checkpoint into `ResNet50Joint4`. Preprocess images with the identical
global p1/p99 normalization used in training (`JointHDF5Dataset4`). Run forward,
apply the identical `denorm()` to map Sigmoid outputs back to physical units.
Output: predicted `k_min, k_max, sigma` (beta predicted but out of comparison
focus).

### 3. Classical (FFT) inference
Reuse `fourier_params()` on the **same underlying image** — the raw stored
log10 image array (`/images[idx]`), WITHOUT p1/p99 normalization (the FFT shape
is normalization-invariant up to an overall scale, which the log-log fit
absorbs). Returns classical `k_min, k_max` in k-mode units, directly comparable
to pyFC's integer modes. Sigma is **CNN-only** (no classical estimator).

## Metrics & figures

For each parameter, report three views: CNN vs ground truth, Classical vs ground
truth (k_min/k_max only), and the method delta.

- **Metrics table**: MAE, RMSE, R², bias (mean residual) for `cnn` and
  `classical`. Expectation: CNN R² very high (in-distribution); classical
  noisier.
- **Figures**:
  - Pred-vs-true scatter per parameter (CNN and classical overlaid, identity
    line).
  - Residual histograms per method.
  - **Diagnostic**: CNN MAE binned by true value of `k_max` and `sigma` —
    reveals *which range* the model fails in (e.g. high k_max, or small sigma).
- **Outputs**: PNG figures + `metrics.json` written to
  `outputs/comparison/synthetic_validation/`.

## Error handling

- `fourier_params` may return `NaN` when the fit has < 3 usable points → filter
  those samples out of the classical metrics and report the dropped count.
- Assert the checkpoint and dataset files exist before inference, with a clear
  message if missing.

## Out of scope

- The second training run (k_max/sigma improvement) — separate follow-up spec
  after validation results are reviewed.
- Inference on real observational clouds (already covered in `io-fits.ipynb`).
- The Bayesian ensemble model (`ResNet50HeteroJoint`) — validation targets the
  non-Bayesian joint head, which is what the second training will operate on.

## Success criteria

- Notebook runs end-to-end on the synthetic test subset.
- CNN metrics show high accuracy (high R², low MAE) on in-distribution images,
  confirming model veracity.
- Classical baseline computed for k_min/k_max with NaN handling.
- Diagnostic plots clearly localize k_max/sigma error by value range, ready to
  inform the second-training spec.
