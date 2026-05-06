# Notebook Index

Research notebooks organized by track. All active notebooks have been fully or substantially executed.

**Status key:** `complete` = fully executed and results saved | `active` = primary working notebook | `archived` = incomplete or superseded (see `archive/`)

---

## Data Generation

| Notebook | Parameter(s) | Approach | Status | Key Result |
|----------|-------------|----------|--------|------------|
| [flexible_dataset_generation.ipynb](flexible_dataset_generation.ipynb) | all | Flexible continuous sampling | `complete` | Demonstrates per-image random k_min, k_max, sigma sampling |
| [data_generation/dataset_gen.ipynb](data_generation/dataset_gen.ipynb) | all | DatasetGen class workflow | `complete` | Original HDF5 generation pipeline (historical) |
| [data_generation/kmax_dataset.ipynb](data_generation/kmax_dataset.ipynb) | k_max | Balanced dataset creation | `complete` | Balanced k_max distribution for local generation |
| [data_generation/k_mas_dataset.ipynb](data_generation/k_mas_dataset.ipynb) | k_min | Random k_min generation | `complete` | Local-scale k_min dataset generation |
| [data_generation/random_kmin_dataset_local.ipynb](data_generation/random_kmin_dataset_local.ipynb) | k_min | Random k_min (small-scale) | `complete` | Small local dataset for testing |

---

## k_min Prediction

| Notebook | Parameter(s) | Approach | Status | Key Result |
|----------|-------------|----------|--------|------------|
| [kmin_experiments/kmin_regression.ipynb](kmin_experiments/kmin_regression.ipynb) | k_min | PyTorch ResNet50 regression | `complete` | MAE=1.30, R²=0.975 (range 1–32) |
| [kmin_experiments/resnet_nopatchs.ipynb](kmin_experiments/resnet_nopatchs.ipynb) | k_min | Full-resolution ResNet50 regression | `complete` | Full-image training, comprehensive evaluation |
| [kmin_experiments/resnet_nopatchs_discrete.ipynb](kmin_experiments/resnet_nopatchs_discrete.ipynb) | k_min | Full-resolution classification | `complete` | Discrete class variant alongside regression |

---

## k_max Prediction

| Notebook | Parameter(s) | Approach | Status | Key Result |
|----------|-------------|----------|--------|------------|
| [kmax_experiments/kmax_regression.ipynb](kmax_experiments/kmax_regression.ipynb) | k_max | PyTorch ResNet50 regression | `complete` | MAE=6.59, R²=0.715 (range 2–64) |
| [kmax_experiments/kmax_regression_flexible.ipynb](kmax_experiments/kmax_regression_flexible.ipynb) | k_max | Flexible dataset variant | `complete` | Regression on flexible-sampled k_max dataset |
| [kmax_experiments/resnet_kmax.ipynb](kmax_experiments/resnet_kmax.ipynb) | k_max | Classification on new dataset | `complete` | Quick validation of binned classification approach |

---

## sigma Prediction

| Notebook | Parameter(s) | Approach | Status | Key Result |
|----------|-------------|----------|--------|------------|
| [sigma_regression.ipynb](sigma_regression.ipynb) | sigma | PyTorch ResNet50 regression | `complete` | MAE=0.58, R²=0.652 (range 0.5–5.0) |
| [sigma_research.ipynb](sigma_research.ipynb) | sigma | Physics validation | `complete` | Confirms sigma effect on fractal power spectrum |

---

## Comparison & Analysis

| Notebook | Parameter(s) | Approach | Status | Key Result |
|----------|-------------|----------|--------|------------|
| [comparison/model_comparison.ipynb](comparison/model_comparison.ipynb) | k_min, k_max | Cross-model evaluation | `complete` | Thesis-ready comparison figures, R² across models |
| [comparison/dual_output_regression.ipynb](comparison/dual_output_regression.ipynb) | k_min + k_max | Multi-task ResNet50 | `complete` | kmin R²=0.897, kmax R²=0.608 — joint model underperforms singles |
| [comparison/loss_benchmark.ipynb](comparison/loss_benchmark.ipynb) | k_min | MAE vs RMSE vs hybrid losses | `active` | Not yet executed — pending loss function comparison study |

---

## Archived (Incomplete or Superseded)

| Notebook | Reason |
|----------|--------|
| [archive/pytorch_resnet.ipynb](archive/pytorch_resnet.ipynb) | Patch extraction approach — 33% executed, abandoned |
| [archive/multi_parameter_regression.ipynb](archive/multi_parameter_regression.ipynb) | Multi-output (kmin+kmax+sigma) — 0% executed, not yet pursued |
| [archive/compare_models_gradcam.ipynb](archive/compare_models_gradcam.ipynb) | Grad-CAM comparison — 73% executed, incomplete |
