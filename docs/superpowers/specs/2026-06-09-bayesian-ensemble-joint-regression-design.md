# Bayesian Ensemble Joint Regression — Design Spec

**Date:** 2026-06-09  
**Status:** Approved  

---

## 1. Goal

Upgrade the current best joint regression model (ResNet50, 3-output Sigmoid, point estimates) to a Bayesian deep ensemble that:

1. Improves `k_max` accuracy (currently R²≈0.76, the weak target)
2. Outputs calibrated uncertainty estimates (μ ± σ) for all three parameters
3. Runs training on an HPC via SLURM, with fully automatic pipeline
4. Supports local inference on Windows PC via a Jupyter notebook

---

## 2. Approach: Deep Ensembles + Heteroscedastic Head

**Why this approach:**
- Deep ensembles are the gold standard for practical uncertainty quantification — well-calibrated, no exotic libraries, pure PyTorch
- Heteroscedastic heads (predict μ + σ per output) capture aleatoric (data) uncertainty; ensemble variance captures epistemic (model) uncertainty
- Combining both gives total uncertainty decomposed into aleatoric + epistemic components
- Gaussian NLL loss naturally adapts to heteroscedastic noise, which likely helps `k_max` whose difficulty varies with `k_min`

---

## 3. Model Architecture

**Backbone:** ResNet50 (same as current best model, `torchvision.models.resnet50`, weights=None for HPC training from scratch)

**Head:** Replace the current 3-output Sigmoid head with a 6-output heteroscedastic head:

```
ResNet50 features (2048-d)
  → Linear(2048, 256) → ReLU → Dropout(0.3)
  → Linear(256, 64)   → ReLU → Dropout(0.2)
  → Linear(64, 6)
  → outputs: [μ_kmin, log_σ_kmin, μ_kmax, log_σ_kmax, μ_sigma, log_σ_sigma]
```

- `μ` outputs pass through Sigmoid → bounded [0, 1] in normalised space
- `log_σ` outputs are unconstrained; at inference `σ = softplus(log_σ)` to ensure σ > 0
- Input: single-channel 128×128, global p1/p99 normalisation (IMG_P1=-2.0818, IMG_P99=0.9409)

**Ensemble:** 5 independent instances, each trained with `seed = base_seed + member_id` (default `base_seed=42`, so seeds 42–46)

---

## 4. Loss Function

Gaussian negative log-likelihood (NLL), summed across 3 parameters with per-param weights:

```
L = Σ_i  w_i · [ (y_i - μ_i)² / (2·σ_i²) + log(σ_i) ]

w_kmin  = 1.0
w_kmax  = 2.0   ← upweighted to improve k_max accuracy
w_sigma = 1.0
```

All targets remain normalised to [0, 1] before loss computation (same normalisation as current model).

---

## 5. Uncertainty Combination at Inference

Given 5 ensemble members each outputting `(μ_i, σ_i)` per parameter:

```
μ_ensemble  = (1/5) Σ μ_i                           # point estimate
σ_aleatoric = sqrt( (1/5) Σ σ_i² )                  # average data noise
σ_epistemic = std( μ_1, ..., μ_5 )                  # model disagreement
σ_total     = sqrt( σ_aleatoric² + σ_epistemic² )   # total uncertainty
```

Reported in original parameter units after denormalisation.

---

## 6. File Structure

```
scripts/hpc_ensemble/
├── config.yaml            # single config — HPC fields clearly marked
├── submit.sh              # professor runs once; submits array + evaluation
├── job_ensemble.sh        # SLURM array --array=0-4 (one job per member)
├── job_evaluate.sh        # SLURM single job, auto-triggered via afterok dependency
├── setup_env.sh           # one-time environment setup on HPC
├── train_member.py        # trains one ensemble member (--config, --member-id)
├── evaluate_ensemble.py   # combines 5 checkpoints, writes results + plots
├── model.py               # ResNet50HeteroJoint (heteroscedastic head)
├── dataset.py             # JointHDF5Dataset (adapted from joint_regression.ipynb)
├── loss.py                # WeightedGaussianNLL
└── requirements.txt       # pinned PyTorch + deps

notebooks/comparison/
└── bayesian_ensemble_inference.ipynb   # local inference on PC
```

---

## 7. config.yaml Schema

```yaml
# ── HPC SETTINGS (professor configures these) ──────────────────────
hpc:
  partition: "gpu"               # CHANGE: GPU partition name
  gpus_per_task: 1               # CHANGE: GPUs per member job
  cpus_per_task: 8               # CHANGE: CPUs for data loading
  mem_gb: 32                     # CHANGE: RAM per job (GB)
  time_limit: "06:00:00"         # CHANGE: wall-clock limit per member
  conda_env: "py311"             # CHANGE: conda environment name
  project_dir: "/path/to/cnn_astro"  # CHANGE: absolute path on HPC

# ── TRAINING ────────────────────────────────────────────────────────
training:
  epochs: 80
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 1.0e-4
  base_seed: 42                  # member i uses seed = base_seed + i
  scheduler: "reduce_on_plateau" # factor=0.5, patience=5, min_lr=1e-7

# ── ENSEMBLE ────────────────────────────────────────────────────────
ensemble:
  n_members: 5

# ── LOSS WEIGHTS ────────────────────────────────────────────────────
loss_weights:
  k_min: 1.0
  k_max: 2.0
  sigma: 1.0

# ── DATA ────────────────────────────────────────────────────────────
data:
  file: "data/raw/uniform_kmin_128x128_100000.h5"
  val_split: 0.1
  test_split: 0.1
  img_p1: -2.0818
  img_p99: 0.9409
  kmin_lo: 1.0
  kmin_hi: 62.0
  kmax_lo: 5.0
  kmax_hi: 64.0
  log_sigma_lo: -4.6052   # log(0.01)
  log_sigma_hi: 1.6094    # log(5.0)

# ── OUTPUT ──────────────────────────────────────────────────────────
output:
  dir: "outputs/hpc_ensemble"
```

---

## 8. SLURM Pipeline

**`submit.sh`** — professor runs once:
```bash
ARRAY_JOB_ID=$(sbatch --parsable job_ensemble.sh)
sbatch --dependency=afterok:$ARRAY_JOB_ID job_evaluate.sh
```

**`job_ensemble.sh`:**
- `#SBATCH --array=0-4`
- Calls `python train_member.py --config config.yaml --member-id $SLURM_ARRAY_TASK_ID`
- Each member writes checkpoint to `outputs/hpc_ensemble/member_{id}/best_model.pt`
- If any member fails, evaluation job is automatically cancelled (SLURM `afterok` semantics)

**`job_evaluate.sh`:**
- Runs after all 5 members succeed
- Calls `python evaluate_ensemble.py --config config.yaml`

---

## 9. Evaluation Outputs

`evaluate_ensemble.py` writes to `outputs/hpc_ensemble/`:

| File | Content |
|------|---------|
| `results.json` | per-image μ, σ_total, σ_aleatoric, σ_epistemic for all 3 params (test set) |
| `results_summary.txt` | MAE, RMSE, R², mean σ_total per param — human readable |
| `scatter.png` | predicted vs true with ±1σ error bars per param |
| `uncertainty_calibration.png` | reliability diagram (is σ_total calibrated?) |
| `residuals.png` | residual histograms per param |

---

## 10. Local Inference Notebook

**`notebooks/comparison/bayesian_ensemble_inference.ipynb`**

Sections:
1. Config — path to ensemble checkpoint folder, normalisation constants
2. Load all 5 heteroscedastic ResNet50 checkpoints (CPU or GPU, auto-detected)
3. Synthetic test set inference — μ ± σ_total scatter plots, calibration plot
4. GASS inference — load FITS, run ensemble per channel, bar chart μ ± σ_total per param per channel (extends current `io-fits.ipynb` workflow)

Works on Windows PC with or without GPU (falls back to CPU automatically).

---

## 11. What Changes vs Current Best Model

| Aspect | Current (joint_regression.ipynb) | New |
|--------|----------------------------------|-----|
| Head outputs | 3 (μ only, Sigmoid) | 6 (μ + log_σ, Sigmoid + softplus) |
| Loss | Joint RMSE, equal weights | Gaussian NLL, k_max weighted 2× |
| Uncertainty | None | σ_aleatoric + σ_epistemic + σ_total |
| Training | 1 model, 50 epochs, ~184 min | 5 models, 80 epochs, HPC parallel |
| Inference | Point estimate | μ ± σ_total per parameter |
| Dataset | Same — uniform_kmin_128x128_100000.h5 | Same |
