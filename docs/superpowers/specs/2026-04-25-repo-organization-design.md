---
name: Repo Organization - Option A (Surgical Fixes)
description: Minimal reorganization of cnn_astro for active daily research
type: project
---

# Repo Organization Design — Option A

## Goal

Fix the highest-friction daily issues without disrupting active research workflows.

## Changes

### 1. Unnest fractal_classifier_02
- Move `src/fractal_classifier_01/fractal_classifier_02/` → `src/fractal_classifier_02/`
- Both packages become siblings under `src/`

### 2. Remove empty placeholder dirs
- Delete `experiments/kmin/`, `experiments/kmax/`, `experiments/dual_output/`

### 3. Organize loose top-level notebooks
- Create `notebooks/multi_param/`
- Move into it: `multi_parameter_regression.ipynb`, `sigma_regression.ipynb`, `sigma_research.ipynb`, `flexible_dataset_generation.ipynb`

### 4. Fix requirements filename typo
- Rename `requirements_py39.txt.txt` → `requirements_py39.txt`

### 5. Update README
- Update the compact repository map to reflect new structure

## Out of Scope
- `uve/` directory (unrelated Go tool, leave in place)
- `outputs/` loose files
- `data/raw/perf/` location
- Notebook import paths (notebooks are self-contained)
