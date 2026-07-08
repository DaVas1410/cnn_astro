# MC Dropout Uncertainty for the v3 Joint Regressor — Design

**Date:** 2026-07-08
**Status:** Approved (design), pending spec review
**Author:** David (with Claude)

## Goal

Produce a **per-prediction Bayesian confidence (error bar)** for the v3 joint
regressor, as a **second, independent uncertainty method** to cross-check the
deep-ensemble results already obtained. The deliverable is calibrated
uncertainty and a side-by-side comparison against the ensemble — not improved
point accuracy.

Ensemble numbers to compare against (from the Colab run, test set):

| param | MAE | RMSE | R² | σ_tot |
|-------|-----|------|----|-------|
| k_min | 0.936 | 1.270 | 0.9922 | 2.8243 |
| k_max | 4.189 | 6.266 | 0.8116 | 6.5759 |
| sigma | 0.178 | 0.203 | 0.9800 | 0.4204 |

(The ensemble predicted 3 targets; this method predicts 4 and compares on the 3
shared ones.)

## Why MC Dropout

The v3 model already contains dropout in its head (`Dropout(0.3)`,
`Dropout(0.2)`), so MC Dropout (Gal & Ghahramani, 2016) needs **zero
retraining**: keep dropout active at inference, run T stochastic forward passes,
and use the spread of predictions as epistemic uncertainty. It runs on the
user's Windows PC (CUDA available) in a few minutes and is easy to explain to
tutors. k_max point accuracy stays ~0.8 (an information limit, unchanged by any
uncertainty method); the new content is calibrated confidence + method
cross-check.

## Reused assets (must match exactly)

- **Checkpoint:** `outputs/comparison/joint_regression_v3/best_model.pt`
- **Model:** `ResNet50Joint4` — `resnet50(weights=None)`, `in_channels=1`, head:
  `Linear(2048→256)→ReLU→Dropout(0.3)→Linear(256→64)→ReLU→Dropout(0.2)→Linear(64→4)`,
  final Sigmoid on outputs (matches v3).
- **Targets:** `['k_min', 'k_max', 'sigma', 'beta']` (N_TARGETS = 4).
- **Data:** `data/raw/balanced_4param_128x128_100000_kmaxfix.h5` (the kmax-fixed set).
- **Split:** `SEED=42`, `train_test_split(indices, test_size=0.2)` then
  `test_size=0.5` on the remainder → 80/10/10. **Evaluate on the test split only.**
- **Image norm:** `(img - IMG_P1) / (IMG_P99 - IMG_P1 + 1e-8)`, using the v3
  notebook's global percentiles.
- **Target norm / denorm ranges:** KMIN 1.0–62.0, KMAX 3.0–64.0,
  sigma via log-scale `LOG_SIG_LO=log(0.01)`, `LOG_SIG_HI=log(5.0)`,
  BETA −3.0..−1.0. Reproduce the v3 `denorm` exactly (sigma column uses
  `exp(_back(...))`).

## Method

1. Load `ResNet50Joint4`, load the checkpoint, call `model.eval()`.
2. **Re-enable dropout only:** iterate modules; for every `nn.Dropout` set
   `.train()`. BatchNorm layers stay in eval mode (use running stats). This is
   the correct MC-Dropout configuration.
3. Run **T = 50** stochastic forward passes over the test set (configurable
   `PASSES`; optional `TEST_SUBSET` cap for speed). Collect a
   `(T, N_test, 4)` array of predictions in normalized space.
4. **Denormalize every sample first, then aggregate:** convert all T×N samples
   to physical units, then per (image, target) take:
   - `μ` = mean over T passes (the prediction)
   - `σ` = std over T passes (the confidence)
   Denorm-then-aggregate (not aggregate-then-denorm) so σ for the log-scaled
   `sigma` target is propagated correctly through `exp`.
5. **Deterministic reference:** also run one pass with dropout OFF to report the
   plain v3 point estimate alongside the MC-Dropout μ (they differ slightly).

## Outputs → `outputs/comparison/mc_dropout/`

- `results.json` — per-target metrics + arrays (`y_true`, `y_pred`, `sigma`,
  and the deterministic point estimate).
- Metrics table (printed + `results_summary.txt`, UTF-8 encoded — the σ/²
  glyphs crash cp1252 on Windows; write with `encoding='utf-8'`).
  Columns: MAE, RMSE, R², mean σ — for all 4 targets.
- `scatter.png` — pred vs true with `μ ± σ` error bars, 4 panels.
- `residuals.png` — residual histograms, 4 panels.
- `calibration.png` — reliability diagram (observed coverage vs expected
  confidence), same construction as `evaluate_ensemble.py::plot_calibration`.
- `comparison.png` + a small table — MC-Dropout vs ensemble on the 3 shared
  targets (metrics + overlaid calibration curves). This is the presentation
  figure.

## Deliverable

A single self-contained notebook:
`notebooks/comparison/mc_dropout_uncertainty.ipynb`

Cells, in order:
1. **Config** — paths, `SEED=42`, `PASSES=50`, `TEST_SUBSET=None`, target
   ranges, device selection.
2. **Dataset + split** — `JointHDF5Dataset4` and the 80/10/10 split, copied from
   the v3 notebook so preprocessing is identical (no cross-notebook import).
3. **Model** — `ResNet50Joint4` definition + checkpoint load.
4. **MC-Dropout inference** — enable-dropout helper + T-pass loop → `(T,N,4)`.
5. **Aggregate + denorm** — μ, σ (physical units) + deterministic reference.
6. **Metrics + save** — table, `results.json`, `results_summary.txt`.
7. **Plots** — scatter±σ, residuals, calibration.
8. **Comparison vs ensemble** — table + overlaid calibration figure.

Runs on the user's PC GPU in a few minutes (T=50 × ~10k test images of 128²).
CPU fallback works but is slower; `TEST_SUBSET` mitigates.

## Non-goals / out of scope

- No retraining, no architecture change, no point-accuracy improvement.
- No standalone `scripts/` module (notebook only, per user).
- No new dependencies (torch, numpy, matplotlib, sklearn, h5py already present).

## Risks / notes

- MC-Dropout μ drifts slightly from the deterministic output because dropout is
  active; the deterministic reference row makes this transparent.
- Dropout rates (0.2/0.3) were chosen for regularization, not calibration, so
  MC-Dropout may **under-estimate** uncertainty. The reliability diagram will
  show this; it's an honest, expected finding to discuss with tutors, not a bug.
- Full test set × T=50 is ~500k forward passes; fine on CUDA, use `TEST_SUBSET`
  if running on CPU.
