# CNN Astro — Fractal Cloud Parameter Prediction

Deep learning research repository for predicting turbulence-related fractal parameters (`k_min`, `k_max`, `sigma`) from synthetic 2D slices of log-normal fractal density cubes generated with [pyFC](https://bitbucket.org/pandante/pyfc).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Data Generation Pipeline](#4-data-generation-pipeline)
5. [Model Development History](#5-model-development-history)
6. [Best Results Summary](#6-best-results-summary)
7. [Notebook Portfolio](#7-notebook-portfolio)
8. [Observational Data Analysis](#8-observational-data-analysis)
9. [Testing](#9-testing)
10. [HPC Integration](#10-hpc-integration)
11. [Quick Start](#11-quick-start)

---

## 1. Project Overview

The goal is to recover three physical parameters of a log-normal fractal turbulence field from a single 2D image:

| Parameter | Meaning | Typical range |
|-----------|---------|---------------|
| `k_min` | Minimum injection scale (wavenumber) | 1 – 62 |
| `k_max` | Maximum dissipation scale (wavenumber) | 2 – 64 |
| `sigma` | Log-normal width of the power spectrum | 1.5 – 4.0 |

The pipeline covers: **synthetic data generation → HDF5 datasets → CNN training → evaluation → inference on real GASS HI observations**.

---

## 2. Repository Structure

```text
cnn_astro/
├── src/
│   ├── dataset_generator.py       # DatasetGen: core HDF5 generation pipeline
│   ├── param_sampler.py           # ParameterSampler: per-image continuous sampling
│   ├── plot_performance.py        # Generation throughput plotting
│   ├── fractal_classifier_01/     # TensorFlow/Keras framework (classification-first)
│   ├── fractal_classifier_02/     # PyTorch lightweight regression framework
│   └── pyFC_lib/                  # Bundled pyFC source (log-normal fractal cubes)
├── scripts/
│   ├── run_generation_flexible.py # Primary generation script (env-var driven)
│   ├── run_generation_balanced.py # Balanced k_min sampling variant
│   ├── job_flexible.sh            # SLURM job script
│   └── archive/                   # Legacy generation scripts
├── notebooks/
│   ├── comparison/                # Head-to-head model experiments (primary track)
│   ├── kmin_experiments/          # Single-parameter k_min studies
│   ├── kmax_experiments/          # Single-parameter k_max studies
│   ├── data_generation/           # Dataset construction notebooks
│   └── archive/                   # Earlier exploratory notebooks
├── observational_images/
│   ├── *.fits.gz                  # GASS HI survey cubes
│   └── io-fits.ipynb              # FITS loading, Fourier analysis, CNN inference
├── configs/
│   ├── dataset_config.ini         # Production generation config
│   └── test_config.ini            # Small-scale test config
├── data/raw/                      # Generated HDF5 datasets (not tracked by git)
├── outputs/                       # Model checkpoints, plots, JSON results
└── tests/                         # Dataset generator unit tests
```

---

## 3. Environment Setup

```bash
# Conda (Python 3.9) — recommended for full pyFC compatibility
conda env create -f environment.yml
conda activate py39

# Or pip (Python 3.11)
pip install -r requirements_py311.txt
```

`src/pyFC_lib/` is inserted into `sys.path` automatically by all generation scripts.

---

## 4. Data Generation Pipeline

### 4.1 Core Generator — `src/dataset_generator.py`

`DatasetGen` is an INI-driven pipeline around `pyFC.LogNormalFractalCube`:

- Safe expression parsing for config fields (e.g. `np.sqrt(5.)`)
- Serial or parallel (`num_workers`) generation
- Batched writes to HDF5; two layouts supported:
  - **Flat** — `/images` (N, H, W) + `/parameters/{k_min, k_max, sigma, ...}` (primary)
  - **Grouped** — `/{HxW}/images` + `/{HxW}/parameters/...` (multi-resolution merges)
- `_detect_layout()`, `merge_datasets()`, `append_dataset()` handle both layouts transparently

### 4.2 Parameter Sampler — `src/param_sampler.py`

`ParameterSampler` drives the flexible workflow with per-image continuous sampling:

- Distributions: `uniform` or `exponential` (skewed toward larger values)
- Enforces Nyquist limit: `k_max ≤ floor(max(ni,nj,nk)/2)`
- Enforces ordering: `k_min < k_max`, `k_min ≥ 1`, `sigma > 0`

### 4.3 Generation Strategy Evolution

| Script | Strategy | Status |
|--------|----------|--------|
| `run_generation.py` | Fixed discrete `k_min` sweep, job-array style | Archive |
| `run_generation_random_kmin.py` | Random `k_min`, auto `k_max` | Archive |
| `run_generation_random_both.py` | Random `k_min` + `k_max`, optional balanced `k_max` | Archive |
| `run_generation_flexible.py` | Per-image continuous sampling of all 4 params | **Primary** |
| `run_generation_balanced.py` | Uniform `k_min` sampling (corrects dataset bias) | Active |

### 4.4 Key Dataset Files

| File | Size | Notes |
|------|------|-------|
| `uniform_kmin_128x128_100000.h5` | 100K images | Uniform `k_min` — used for joint regression |
| `balanced_4param_128x128_100000.h5` | 100K images | 4-param balanced sampling |
| `flexible_20260425_221800_128x128_100000.h5` | 100K images | Flexible (biased `k_min`) |
| `flexible_kmax_20260504_*_128x128_100000.h5` | 100K × 2 | `k_max`-focused variants |

---

## 5. Model Development History

### 5.1 Early Phase — Classification and Single-Parameter Regression

**`src/fractal_classifier_01`** (TensorFlow/Keras) was the initial framework:
- Classification-first approach with centroid interpolation
- YAML-driven config, CLI workflows, TensorBoard, Grad-CAM support
- Rich kernel analysis: export callbacks, checkpoint-wise statistics, convergence trajectories
- Best result: `k_min` classification ~99% accuracy on a discrete-label dataset

**`src/fractal_classifier_02`** (PyTorch) pivoted to regression:
- Depthwise-separable CNN (~180K params), Huber loss
- Demonstrated that continuous regression outperforms classification for `k_min`
- Explicit multi-resolution HDF5 support

### 5.2 Loss Benchmark — `notebooks/comparison/loss_benchmark.ipynb`

Systematic comparison of MAE / RMSE / hybrid (0.5·RMSE+0.5·MAE) loss functions on the single-parameter task, using `flexible_kmax_20260504_*` datasets:

| Target | Loss | MAE | RMSE | R² |
|--------|------|-----|------|----|
| k_min | RMSE | 1.43 | 1.76 | 0.957 |
| k_min | MAE | 2.76 | 4.18 | 0.755 |
| k_min | Hybrid | 2.88 | 3.32 | 0.845 |
| k_max | Hybrid | 6.44 | 6.44 | — |
| sigma | Hybrid | 1.01 | 1.29 | — |

**Conclusion:** RMSE is the best single-target loss. The hybrid (MAE+RMSE) performed worst overall. `k_max` and `sigma` single-head models failed without proper per-image normalisation.

### 5.3 Dual-Output Regression — `notebooks/comparison/dual_output_regression.ipynb`

First attempt at predicting `k_min` + `k_max` jointly. Demonstrated the need for proper normalisation and a uniform `k_min` sampling distribution.

### 5.4 Joint Regression (v1) — `notebooks/comparison/joint_regression.ipynb`

**Dataset:** `uniform_kmin_128x128_100000.h5` (100K, uniform `k_min`)  
**Architecture:** Custom depthwise-separable CNN, 3-head output (`k_min`, `k_max`, `sigma`)  
**Normalisation:** global p1/p99 clipping: `IMG_P1 = -2.0818`, `IMG_P99 = 0.9409`  
**Training:** 50 epochs, ~184 min, best val loss **0.1239**

| Param | MAE | RMSE | R² | % of range |
|-------|-----|------|----|-----------|
| k_min | 0.475 | 0.641 | 0.9987 | 0.8% |
| k_max | 3.917 | 6.747 | 0.7597 | 6.6% |
| sigma | 0.080 | 0.114 | 0.9937 | 1.6% |

`k_min` and `sigma` are recovered with excellent accuracy. `k_max` remains harder due to its strong coupling with image texture at scales that overlap `k_min`.

### 5.5 Joint Regression v2 — `notebooks/comparison/joint_regression_v2.ipynb`

Extended the joint model to predict **4 outputs** (`k_min`, `k_max`, `sigma`, `beta`):  
**Dataset:** `balanced_4param_128x128_100000.h5`  
**Training:** 50 epochs, ~192 min, best val loss **0.268**

| Param | MAE | RMSE | R² |
|-------|-----|------|----|
| k_min | 0.520 | 0.679 | 0.9977 |
| k_max | 4.936 | 8.000 | 0.692 |
| sigma | 0.105 | 0.154 | 0.989 |
| beta | 0.178 | 0.260 | 0.795 |

Checkpoint: `outputs/comparison/joint_regression_v2/best_model.pt`

### 5.6 ResNet-50 with ImageNet Normalisation — `notebooks/comparison/resnet_imagenet_norm.ipynb`

Transfer learning baseline using `torchvision.models.resnet50` (IMAGENET1K_V2 pretrained):

- 3-channel input (grayscale repeated), ImageNet mean/std normalisation
- 5 epoch frozen warm-up → 45 epoch fine-tune with ReduceLROnPlateau
- Shared backbone, 3-head output (`k_min`, `k_max`, `sigma`)
- **Training:** 50 epochs, ~186 min, best val loss **0.124**

| Param | MAE | RMSE | R² |
|-------|-----|------|----|
| k_min | **0.469** | **0.598** | **0.9989** |
| k_max | **3.974** | **6.622** | **0.768** |
| sigma | **0.087** | **0.123** | **0.993** |

ResNet-50 with ImageNet pretraining matches or slightly beats the custom CNN on all three targets with a similar training budget. Checkpoint: `outputs/comparison/resnet_imagenet_norm/best_model.pt`

### 5.7 DINOv2 ViT-S/14 — `notebooks/comparison/dino_regression.ipynb`

Vision Transformer foundation model as a feature extractor:

- **Model:** `dinov2_vits14` (ViT-S/14, 384-dim [CLS] token, ~21M params)
- 3-channel input (grayscale repeated), ImageNet normalisation, resized to 224×224
- Frozen backbone warm-up → CosineAnnealing fine-tune of head + last ViT blocks
- Latent space visualisation: t-SNE of [CLS] tokens coloured by `k_min`/`k_max`/`sigma`
- Outputs: `outputs/comparison/dino_regression/`

**Key plots produced:** `latent_tsne.png`, `scatter.png`, `training_curves.png`

> DINOv2 explores whether self-supervised ViT features encode fractal texture structure better than supervised CNNs. Results show clear clustering in t-SNE space by `k_min`, suggesting the ViT learned scale-sensitive features despite training on natural images.

A Colab-adapted version exists at `notebooks/comparison/dino_regression_colab.ipynb` for running on free GPU instances.

---

## 6. Best Results Summary

All results on the `uniform_kmin_128x128_100000.h5` test split (or equivalent held-out set), 3-output joint models:

| Model | k_min R² | k_min MAE | k_max R² | k_max MAE | sigma R² | sigma MAE |
|-------|----------|-----------|----------|-----------|----------|-----------|
| Custom joint CNN (v1) | 0.999 | 0.475 | 0.760 | 3.917 | 0.994 | 0.080 |
| ResNet-50 ImageNet | **0.999** | **0.469** | **0.768** | **3.974** | **0.993** | **0.087** |
| Joint CNN v2 (4-head) | 0.998 | 0.520 | 0.692 | 4.936 | 0.989 | 0.105 |

**Take-away:** Custom CNNs and ResNet-50 transfer learning perform comparably on this task. `k_min` and `sigma` are near-perfectly recovered (R² > 0.99). `k_max` remains the hard target (R² ≈ 0.75–0.77).

---

## 7. Notebook Portfolio

### `notebooks/comparison/` — main experiment track

| Notebook | Purpose |
|----------|---------|
| `loss_benchmark.ipynb` | MAE / RMSE / hybrid loss comparison, single-head |
| `dual_output_regression.ipynb` | First joint `k_min`+`k_max` attempt |
| `joint_regression.ipynb` | Best 3-head custom CNN (v1) |
| `joint_regression_v2.ipynb` | 4-head custom CNN (`k_min`, `k_max`, `sigma`, `beta`) |
| `resnet_imagenet_norm.ipynb` | ResNet-50 transfer learning, 3-head |
| `dino_regression.ipynb` | DINOv2 ViT-S/14 regression + t-SNE latent analysis |
| `dino_regression_colab.ipynb` | Colab-compatible DINOv2 version |
| `model_comparison.ipynb` | Side-by-side summary across trained models |

### `notebooks/kmin_experiments/` — single-parameter `k_min` track

| Notebook | Notes |
|----------|-------|
| `kmin_regression.ipynb` | Early lightweight regression |
| `resnet_nopatchs.ipynb` | ResNet without patch extraction |
| `resnet_nopatchs_discrete.ipynb` | Discrete-label variant |

### `notebooks/kmax_experiments/` — single-parameter `k_max` track

| Notebook | Notes |
|----------|-------|
| `kmax_regression.ipynb` | Baseline regression |
| `kmax_regression_flexible.ipynb` | Flexible-dataset variant |
| `resnet_kmax.ipynb` | ResNet-based `k_max` regression |

### `notebooks/data_generation/`

Dataset construction and exploration notebooks for each generation strategy.

### `notebooks/archive/`

Earlier experiments: `pytorch_resnet.ipynb`, `multi_parameter_regression.ipynb`, `compare_models_gradcam.ipynb`.

---

## 8. Observational Data Analysis

### `observational_images/io-fits.ipynb`

Applies trained models to real GASS (Galactic All-Sky Survey) HI spectral cubes:

**Data:** `gass_314_-28_1774393621.fits.gz` — shape (1201, 125, 125), split into full integration + 5 velocity channels.

**Fourier analysis pipeline:**
1. 2D FFT + fftshift → 2D power spectrum
2. 1D projection (mean over `ky`) + azimuthal radial P(k)
3. Power-law fit via log-log linear regression; reports spectral index α vs Kolmogorov references (−11/3 for 3D, −8/3 for 2D)
4. `k_min`/`k_max` estimated from ±0.5 dex residual threshold on fitted power law
5. `fourier_params(img2d)` helper for batch channel analysis

**CNN inference:**
- Uses `outputs/comparison/joint_regression/best_model.pt`
- Bar chart comparison: FFT-derived vs CNN-predicted `k_min`/`k_max` across all channels
- `sigma` from CNN only (not accessible via Fourier analysis)

---

## 9. Testing

Tests cover `DatasetGen` (not model packages):

```bash
# Run all tests
python tests/run_tests.py -v

# Run a specific class
python tests/run_tests.py TestConfigParsing -v

# Environment sanity check
python tests/check_dependencies.py
```

Test suite covers: config parsing, parameter combinations, worker behaviour, HDF5 operations, dataset population, merge/append, memory utilities.

---

## 10. HPC Integration

```bash
# Flexible generation on SLURM
sbatch scripts/job_flexible.sh

# Example local run
NUM_IMAGES=1000 KMIN_LOW=1 KMIN_HIGH=32 KMAX_MODE=auto SIGMA_LOW=2.0 SIGMA_HIGH=2.5 \
  python scripts/run_generation_flexible.py
```

`job_flexible.sh` supports environment-variable overrides for all generation parameters and logs to `logs/`.

---

## 11. Quick Start

```bash
# 1. Set up environment
conda env create -f environment.yml && conda activate py39

# 2. Generate a small test dataset
NUM_IMAGES=500 KMIN_LOW=1 KMIN_HIGH=32 KMAX_MODE=auto SIGMA_LOW=2.0 SIGMA_HIGH=3.0 \
  python scripts/run_generation_flexible.py

# 3. Run tests
python tests/run_tests.py -v

# 4. Open the best joint regression notebook
jupyter notebook notebooks/comparison/joint_regression.ipynb

# 5. (Optional) DINOv2 experiment
jupyter notebook notebooks/comparison/dino_regression.ipynb
```

---

## Citation

```bibtex
@thesis{fractal_cnn_2026,
  title  = {Deep Learning for Fractal Cloud Parameter Prediction},
  author = {J.D.V.V},
  year   = {2026}
}
```

## License

MIT License
