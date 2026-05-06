# Repo Cleanup & Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organize the cnn_astro thesis workspace by archiving superseded work, deleting corrupted/empty items, restructuring the src/ package layout, and adding a deletion audit trail plus notebook index.

**Architecture:** All operations are file moves, deletions, and new document creation. No code logic changes. Archive directories act as a "soft delete" — files are preserved but clearly marked as inactive. A `DELETED.md` log at the repo root provides a permanent human-readable audit trail.

**Tech Stack:** Bash (mv, rm, mkdir), Markdown

---

### Task 1: Create DELETED.md audit log

**Files:**
- Create: `DELETED.md`

- [ ] **Step 1: Create DELETED.md with header and all planned entries**

Create `/home/davas/Documents/cnn_astro/DELETED.md` with this exact content:

```markdown
# Deletion & Archive Log

This file records every file or directory removed or archived from this workspace, with the date and reason. It serves as the audit trail for the thesis/paper project.

---

## 2026-05-05

### Deleted

| Path | Reason |
|------|--------|
| `notebooks/kmin_experiments/fractal_classifier_v02.ipynb` | JSON corruption — file unreadable and unrecoverable |
| `notebooks/kmax_experiments/kmax_binned_classification.ipynb` | JSON corruption — file unreadable and unrecoverable |
| `experiments/kmin/` | Empty placeholder directory, never used |
| `experiments/kmax/` | Empty placeholder directory, never used |
| `experiments/dual_output/` | Empty placeholder directory, never used |

### Archived

| Original Path | Archive Path | Reason |
|--------------|-------------|--------|
| `scripts/run_generation.py` | `scripts/archive/run_generation.py` | Legacy — discrete k_min sweep; superseded by flexible workflow |
| `scripts/run_generation_random_both.py` | `scripts/archive/run_generation_random_both.py` | Legacy — random k_min+k_max; superseded by flexible workflow |
| `scripts/run_generation_random_kmin.py` | `scripts/archive/run_generation_random_kmin.py` | Legacy — random k_min only; superseded by flexible workflow |
| `scripts/run_generation_random_kmax.py` | `scripts/archive/run_generation_random_kmax.py` | Legacy — random k_max only; superseded by flexible workflow |
| `scripts/run_generation_flexible_kmax.py` | `scripts/archive/run_generation_flexible_kmax.py` | Specialized kmax variant; superseded by main flexible script |
| `scripts/job.sh` | `scripts/archive/job.sh` | Legacy SLURM array job; superseded by job_flexible.sh |
| `notebooks/kmin_experiments/pytorch_resnet.ipynb` | `notebooks/archive/pytorch_resnet.ipynb` | Only 33% executed — patch extraction approach abandoned |
| `notebooks/multi_parameter_regression.ipynb` | `notebooks/archive/multi_parameter_regression.ipynb` | 0% executed — multi-output (kmin+kmax+sigma) not yet pursued |
| `notebooks/compare_models_gradcam.ipynb` | `notebooks/archive/compare_models_gradcam.ipynb` | 73% executed — Grad-CAM comparison incomplete |
| `outputs/kmin/v02/` | `outputs/archive/kmin_v02/` | Early TF/Keras classification (6 k-values); superseded by regression approach |
| `outputs/kmin/v03/` | `outputs/archive/kmin_v03/` | Iteration on v02; superseded by regression approach |
| `outputs/kmin/v1/` | `outputs/archive/kmin_v1/` | Early classification baseline; superseded by regression approach |
| `outputs/kmin/resnet50_patches/` | `outputs/archive/kmin_resnet50_patches/` | Patch-based ResNet50 experiment; superseded by full-image regression |
```

- [ ] **Step 2: Verify file created**

```bash
ls -la /home/davas/Documents/cnn_astro/DELETED.md
```

Expected: file exists, non-zero size.

---

### Task 2: Archive legacy scripts

**Files:**
- Create: `scripts/archive/` (directory)
- Move: 6 files from `scripts/` to `scripts/archive/`

- [ ] **Step 1: Create archive directory**

```bash
mkdir -p /home/davas/Documents/cnn_astro/scripts/archive
```

- [ ] **Step 2: Move all 6 legacy scripts**

```bash
cd /home/davas/Documents/cnn_astro/scripts
mv run_generation.py archive/
mv run_generation_random_both.py archive/
mv run_generation_random_kmin.py archive/
mv run_generation_random_kmax.py archive/
mv run_generation_flexible_kmax.py archive/
mv job.sh archive/
```

- [ ] **Step 3: Verify scripts/ now contains only active files**

```bash
ls /home/davas/Documents/cnn_astro/scripts/
```

Expected output:
```
archive/
job_flexible.sh
run_generation_flexible.py
```

---

### Task 3: Archive incomplete notebooks

**Files:**
- Create: `notebooks/archive/` (directory)
- Move: 3 notebooks into archive

- [ ] **Step 1: Create archive directory**

```bash
mkdir -p /home/davas/Documents/cnn_astro/notebooks/archive
```

- [ ] **Step 2: Move the 3 incomplete notebooks**

```bash
cd /home/davas/Documents/cnn_astro/notebooks
mv kmin_experiments/pytorch_resnet.ipynb archive/
mv multi_parameter_regression.ipynb archive/
mv compare_models_gradcam.ipynb archive/
```

- [ ] **Step 3: Verify moves**

```bash
ls /home/davas/Documents/cnn_astro/notebooks/archive/
```

Expected:
```
compare_models_gradcam.ipynb
multi_parameter_regression.ipynb
pytorch_resnet.ipynb
```

---

### Task 4: Delete corrupted notebooks and empty directories

**Files:**
- Delete: 2 corrupted notebooks
- Delete: 3 empty experiment directories

- [ ] **Step 1: Delete corrupted notebooks**

```bash
rm /home/davas/Documents/cnn_astro/notebooks/kmin_experiments/fractal_classifier_v02.ipynb
rm /home/davas/Documents/cnn_astro/notebooks/kmax_experiments/kmax_binned_classification.ipynb
```

- [ ] **Step 2: Delete empty experiment placeholder directories**

```bash
rmdir /home/davas/Documents/cnn_astro/experiments/kmin
rmdir /home/davas/Documents/cnn_astro/experiments/kmax
rmdir /home/davas/Documents/cnn_astro/experiments/dual_output
```

- [ ] **Step 3: Verify deletions**

```bash
ls /home/davas/Documents/cnn_astro/notebooks/kmin_experiments/
ls /home/davas/Documents/cnn_astro/notebooks/kmax_experiments/
ls /home/davas/Documents/cnn_astro/experiments/
```

Expected: neither corrupted notebook appears; experiments/ shows only `archive/`.

---

### Task 5: Archive old kmin outputs

**Files:**
- Create: `outputs/archive/` (directory)
- Move: 4 old kmin output directories

- [ ] **Step 1: Create archive directory**

```bash
mkdir -p /home/davas/Documents/cnn_astro/outputs/archive
```

- [ ] **Step 2: Move the 4 superseded output dirs**

```bash
cd /home/davas/Documents/cnn_astro/outputs
mv kmin/v02 archive/kmin_v02
mv kmin/v03 archive/kmin_v03
mv kmin/v1 archive/kmin_v1
mv kmin/resnet50_patches archive/kmin_resnet50_patches
```

- [ ] **Step 3: Verify kmin/ now contains only regression/**

```bash
ls /home/davas/Documents/cnn_astro/outputs/kmin/
```

Expected:
```
pytorch_resnet/
regression/
```

```bash
ls /home/davas/Documents/cnn_astro/outputs/archive/
```

Expected:
```
kmin_resnet50_patches/
kmin_v02/
kmin_v1/
kmin_v03/
```

---

### Task 6: Move fractal_classifier_02 to src/ top level

**Files:**
- Move: `src/fractal_classifier_01/fractal_classifier_02/` → `src/fractal_classifier_02/`
- Modify: `CLAUDE.md:76`

- [ ] **Step 1: Move the package**

```bash
mv /home/davas/Documents/cnn_astro/src/fractal_classifier_01/fractal_classifier_02 \
   /home/davas/Documents/cnn_astro/src/fractal_classifier_02
```

- [ ] **Step 2: Verify src/ structure**

```bash
ls /home/davas/Documents/cnn_astro/src/
```

Expected:
```
dataset_generator.py
fractal_classifier_01/
fractal_classifier_02/
param_sampler.py
plot_performance.py
__pycache__/
pyFC_lib/
```

- [ ] **Step 3: Verify fractal_classifier_02 contents intact**

```bash
ls /home/davas/Documents/cnn_astro/src/fractal_classifier_02/
```

Expected: same files as before the move (config/, data/, models/, scripts/, training/, README.md, etc.)

- [ ] **Step 4: Update CLAUDE.md line 76**

In `CLAUDE.md`, find the line:

```
**`src/fractal_classifier_01/fractal_classifier_02/`** — Lightweight regression framework (nested inside v0.1):
```

Replace with:

```
**`src/fractal_classifier_02/`** — Lightweight regression framework (PyTorch, depthwise-separable CNN ~180K params):
```

- [ ] **Step 5: Verify CLAUDE.md no longer references nested path**

```bash
grep "fractal_classifier_01/fractal_classifier_02" /home/davas/Documents/cnn_astro/CLAUDE.md
```

Expected: no output (empty).

---

### Task 7: Fix requirements filename typo

**Files:**
- Rename: `requirements_py39.txt.txt` → `requirements_py39.txt`

- [ ] **Step 1: Rename the file**

```bash
mv /home/davas/Documents/cnn_astro/requirements_py39.txt.txt \
   /home/davas/Documents/cnn_astro/requirements_py39.txt
```

- [ ] **Step 2: Verify**

```bash
ls /home/davas/Documents/cnn_astro/requirements_py39*
```

Expected:
```
requirements_py39.txt
```

---

### Task 8: Create notebooks/NOTEBOOKS.md index

**Files:**
- Create: `notebooks/NOTEBOOKS.md`

- [ ] **Step 1: Create the index file**

Create `/home/davas/Documents/cnn_astro/notebooks/NOTEBOOKS.md` with this content:

```markdown
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
```

- [ ] **Step 2: Verify file created**

```bash
wc -l /home/davas/Documents/cnn_astro/notebooks/NOTEBOOKS.md
```

Expected: 70+ lines.

---

### Task 9: Final verification

- [ ] **Step 1: Verify top-level structure matches spec**

```bash
ls /home/davas/Documents/cnn_astro/
```

Expected to include: `DELETED.md`, `README.md`, `CLAUDE.md`, `environment.yml`, `requirements_py311.txt`, `requirements_py39.txt` (no double extension).

- [ ] **Step 2: Verify no broken nested path remains**

```bash
find /home/davas/Documents/cnn_astro/src/fractal_classifier_01 -name "fractal_classifier_02" -type d
```

Expected: no output.

- [ ] **Step 3: Confirm active scripts only in scripts/**

```bash
ls /home/davas/Documents/cnn_astro/scripts/
```

Expected:
```
archive/
job_flexible.sh
run_generation_flexible.py
```

- [ ] **Step 4: Confirm active kmin notebooks only in kmin_experiments/**

```bash
ls /home/davas/Documents/cnn_astro/notebooks/kmin_experiments/
```

Expected (3 notebooks, no fractal_classifier_v02.ipynb):
```
kmin_regression.ipynb
resnet_nopatchs.ipynb
resnet_nopatchs_discrete.ipynb
```

- [ ] **Step 5: Confirm experiments/ has only archive/**

```bash
ls /home/davas/Documents/cnn_astro/experiments/
```

Expected:
```
archive/
```
