# Loss Function Benchmark & Sigma Model Improvement — Design Spec

**Date:** 2026-05-03  
**Status:** Approved

---

## Goal

Compare three loss functions (MAE, RMSE, 0.5·RMSE + 0.5·MAE) for predicting fractal turbulence parameters from synthetic cloud images, and fix the sigma regression model which significantly underperforms (R²=0.80 vs R²=0.99 for k_min/k_max).

---

## Phase 1 — New Dataset Generation

### Motivation

The existing large flexible dataset (`flexible_20260425_221800_128x128_100000.h5`) has k_max constant at 64 (Nyquist limit for 128×128 images) because the generation used `KMAX_MODE=auto`. k_max regression is therefore impossible from that dataset.

### New Dataset

**File:** `data/raw/flexible_kmax_{timestamp}_128x128_100000.h5`  
**Generator:** new script `scripts/run_generation_flexible_kmax.py`

| Parameter | Range | Distribution |
|-----------|-------|-------------|
| k_min | [1.0, 32.0] | uniform |
| k_max | [1.0, 64.0] | uniform, resampled per image until k_max > k_min |
| sigma | [0.01, 5.0] | uniform |
| mean | 1.0 (fixed) | — |
| beta | −5/3 (fixed) | — |
| image size | 128×128×1 | — |
| N | 100,000 | — |

**k_max constraint enforcement:** after sampling k_min and k_max independently, if k_max ≤ k_min, resample k_max until k_max > k_min. This avoids pyFC rejections and ensures a valid physical configuration for every image.

This single dataset serves all three regression targets (k_min, k_max, sigma).

---

## Phase 2 — Loss Function Benchmark

### Notebook

**Path:** `notebooks/comparison/loss_benchmark.ipynb`  
**Output dir:** `outputs/comparison/loss_benchmark/{target}/{loss_name}/`

### Loss Functions

| Name | Formula | Notes |
|------|---------|-------|
| `mae` | mean\|pred − target\| | robust to outliers |
| `rmse` | √mean(pred − target)² | penalizes large errors |
| `rmse_mae` | 0.5·RMSE + 0.5·MAE | balanced, α=0.5 fixed |

Huber loss excluded — already tested in existing notebooks.

### Targets

| Target | Param range | Normalization | Phase |
|--------|------------|---------------|-------|
| k_min | [1.0, 32.0] | linear to [0, 1] | 2 |
| k_max | [1.0, 64.0] | linear to [0, 1] | 2 |
| sigma | [0.01, 5.0] | log-scale, see Phase 3 | 3 |

### Architecture

ResNet50 with first conv modified to accept 1-channel input (identical to existing notebooks):
- `conv1`: 1→64 channels, kernel 7×7, stride 2
- Regression head: FC(2048→256)→ReLU→Dropout(0.3)→FC(256→64)→ReLU→Dropout(0.2)→FC(64→1)
- ~24M parameters

### Training Configuration

| Setting | Value |
|---------|-------|
| Epochs | 30 (fixed, no early stopping — fair comparison) |
| Batch size | 16 |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau, factor=0.5, patience=5, min_lr=1e-7 |
| Mixed precision | GradScaler on CUDA |
| Train/val/test split | 80/10/10 |

### Notebook Structure

```
[1] Config        — paths, hyperparams, LOSSES dict, TARGETS dict
[2] Shared code   — HDF5Dataset, ResNet50Regressor, RegressionLoss,
                    train_epoch, eval_epoch, NumpyEncoder (fixes float32 JSON bug)
[3] Benchmark     — nested loop: target → loss_name
                    skip-if-exists: if results.json present, continue
                    saves: best_model.pt + results.json per run
[4] Comparison    — load all results.json → 5 inline plots (plt.show(), not saved)
[5] Sigma section — phase 2 (see below)
```

### Resume / Skip Logic

Each run checks `(out_dir / 'results.json').exists()` before training. Interrupted runs can be restarted without re-running completed experiments. `results.json` contains training history (all epochs) and test metrics (MAE, RMSE, R²).

### NumpyEncoder Fix

All three existing notebooks fail with `TypeError: float32 is not JSON serializable` when saving history. The shared `NumpyEncoder` class converts numpy scalars and arrays to native Python types before JSON serialization. Applied to all `json.dump()` calls.

### Comparison Plots (inline, not saved)

Plots 1–4 are rendered twice: once after Phase 2 (2 rows: k_min/k_max) and once after Phase 3 (3 rows: k_min/k_max/sigma) in the same cells, updated in place.

1. **Training curves grid** (rows=targets × 3 cols=losses): val loss over 30 epochs. All three losses overlaid per target for direct convergence and stability comparison.
2. **Final metrics bar chart** (one row per target): grouped bars for test MAE, RMSE, R² per loss function.
3. **Predicted vs true scatter grid** (rows=targets × 3 cols=losses): identity line overlaid, one scatter per run.
4. **Residual distributions** (rows=targets × 3 cols=losses): histogram of (predicted − true), shows symmetry and tail behavior.
5. **Summary table**: (target, loss) rows × (MAE, RMSE, R², training time) columns, printed to cell output.

---

## Phase 3 — Sigma Model Fixes & Benchmark

Added as a dedicated section at the bottom of `loss_benchmark.ipynb`.

### Root Cause Analysis

| Issue | Impact |
|-------|--------|
| Normalization mismatch: `SIGMA_MIN=0.5` but data min=0.01 | ~10% of targets normalize to negative values, out of distribution for sigmoid-bounded prediction |
| Per-image intensity normalization `(img−min)/(max−min)` | Removes global dynamic range which encodes sigma directly (sigma controls log-normal spread, visible as image contrast) |
| Linear target on [0.01, 5.0] spans 2.5 orders of magnitude | Loss gradient dominated by large sigma values, small sigma errors ignored |

### Fixes Applied

1. **Normalization range:** `SIGMA_MIN=0.01`, `SIGMA_MAX=5.0` (match actual data range).
2. **Global image normalization:** compute p1/p99 percentiles from a random 1k-image sample of the dataset; apply `(img − p1) / (p99 − p1)` uniformly to all images. Preserves inter-image contrast differences that carry sigma information.
3. **Log-scale target:** predict `ln(sigma)` (natural log) instead of `sigma`, normalized to [0, 1] over `[ln(0.01), ln(5.0)]`. Global percentiles (p1/p99) computed once from a 1k-image sample at dataset load time and stored as constants. Denormalize at evaluation: `sigma_pred = exp(pred_denorm)`. Makes the regression target scale-uniform across the full range.

### Sigma Benchmark

Same 3-loss comparison (MAE, RMSE, RMSE+MAE) with the fixed dataset and fixes applied. Uses the same new 100k dataset.

Additional comparison plot: sigma test results alongside k_min/k_max in the summary bar chart, for a unified 3-target × 3-loss comparison view.

---

## Deliverables

| Item | Path |
|------|------|
| Dataset generation script | `scripts/run_generation_flexible_kmax.py` |
| New dataset | `data/raw/flexible_kmax_{timestamp}_128x128_100000.h5` |
| Benchmark notebook | `notebooks/comparison/loss_benchmark.ipynb` |
| Model checkpoints | `outputs/comparison/loss_benchmark/{target}/{loss}/best_model.pt` |
| Results JSON | `outputs/comparison/loss_benchmark/{target}/{loss}/results.json` |

---

## Known Constraints

- RTX 3050 Ti Laptop (4GB VRAM): batch size 16 is the working limit for ResNet50 at 128×128
- 9 total training runs (3 losses × 3 targets) × ~4.5 min/epoch × 30 epochs ≈ **20 hours** — intended as overnight runs with the skip-if-exists resume mechanism
- k_max constraint enforcement adds a small rejection-sampling loop per image in the generator; expected overhead is negligible since k_max > k_min is satisfied most of the time
