# Repo Cleanup & Documentation Design
**Date**: 2026-05-05  
**Status**: Approved

## Goal

Organize and document the `cnn_astro` research workspace for thesis and paper submission. The repo is a personal record folder (not the primary git repo). All deletions must be logged in `DELETED.md` for audit trail.

## Guiding Principles

- **Archive, don't delete** superseded work — it may be cited in the thesis
- **Delete only** what is broken (corrupted) or truly empty
- **Document everything** — DELETED.md for removals, NOTEBOOKS.md as a research index
- All datasets are kept regardless of age — disk space is not a concern

---

## 1. Archive Structure

Three new `archive/` subdirectories are created. Nothing in these folders is active work, but all files are preserved.

### `scripts/archive/`
Move the following legacy generation scripts (superseded by `run_generation_flexible.py`):
- `run_generation.py`
- `run_generation_random_both.py`
- `run_generation_random_kmin.py`
- `run_generation_random_kmax.py`
- `run_generation_flexible_kmax.py` (specialized variant, superseded by flexible)
- `job.sh` (superseded by `job_flexible.sh`)

Active scripts that stay in `scripts/`:
- `run_generation_flexible.py` — primary workflow
- `job_flexible.sh` — primary SLURM script

### `notebooks/archive/`
Move incomplete or abandoned notebooks:
- `kmin_experiments/pytorch_resnet.ipynb` — only 33% executed, approach abandoned
- `multi_parameter_regression.ipynb` — 0% executed, not yet pursued
- `compare_models_gradcam.ipynb` — 73% executed, incomplete

### `outputs/archive/`
Move superseded kmin experiment output directories:
- `outputs/kmin/v02/`
- `outputs/kmin/v03/`
- `outputs/kmin/v1/`
- `outputs/kmin/resnet50_patches/`

---

## 2. Deletions

These items are permanently deleted. Each is logged in `DELETED.md` with date and reason.

| Item | Reason |
|------|--------|
| `notebooks/kmin_experiments/fractal_classifier_v02.ipynb` | JSON corruption — file unreadable and unrecoverable |
| `notebooks/kmax_experiments/kmax_binned_classification.ipynb` | JSON corruption — file unreadable and unrecoverable |
| `experiments/kmin/` | Empty placeholder directory, never used |
| `experiments/kmax/` | Empty placeholder directory, never used |
| `experiments/dual_output/` | Empty placeholder directory, never used |

---

## 3. Source Restructure

Move `src/fractal_classifier_01/fractal_classifier_02/` → `src/fractal_classifier_02/`

**Reason**: The two model frameworks are siblings (v0.1 = TF/Keras classification, v0.2 = PyTorch lightweight regression). Nesting v0.2 inside v0.1 implies a parent-child relationship that doesn't exist. After the move, `src/` has a flat, parallel structure:

```
src/
├── fractal_classifier_01/   — TF/Keras classification framework
├── fractal_classifier_02/   — PyTorch lightweight regression framework
├── dataset_generator.py
├── param_sampler.py
├── plot_performance.py
└── pyFC_lib/
```

After the move, update:
- `src/fractal_classifier_02/README.md` — any path references
- Root `README.md` — Architecture section paths

---

## 4. Minor Fixes

- Rename `requirements_py39.txt.txt` → `requirements_py39.txt` (double extension typo)

---

## 5. Documentation Added

### `DELETED.md` (repo root)
One entry per deleted or archived item. Format:

```markdown
## YYYY-MM-DD

### Deleted
| Path | Reason |
|------|--------|
| ... | ... |

### Archived
| Original Path | Archive Path | Reason |
|--------------|-------------|--------|
| ... | ... | ... |
```

### `notebooks/NOTEBOOKS.md`
A human-readable index of all notebooks organized by research track. Each entry includes:
- Notebook name and link
- Parameter(s) studied
- Approach (classification / regression / analysis)
- Status: `complete` | `active` | `archived`
- Key result (metric or finding, one line)

---

## Final Directory Structure (After Cleanup)

```
cnn_astro/
├── configs/
│   ├── dataset_config.ini
│   └── test_config.ini
├── data/
│   └── raw/
│       ├── *.h5              (all 10 datasets kept)
│       └── perf/
├── docs/
│   └── superpowers/
│       ├── plans/
│       └── specs/
├── experiments/
│   └── archive/
│       └── first_steps/      (unchanged)
├── notebooks/
│   ├── archive/              (NEW — 3 incomplete notebooks)
│   ├── comparison/
│   ├── data_generation/
│   ├── kmax_experiments/     (2 notebooks remain)
│   ├── kmin_experiments/     (3 notebooks remain)
│   ├── NOTEBOOKS.md          (NEW — full index)
│   ├── flexible_dataset_generation.ipynb
│   ├── sigma_regression.ipynb
│   └── sigma_research.ipynb
├── outputs/
│   ├── archive/              (NEW — old kmin outputs)
│   ├── comparison/
│   ├── kmax/
│   ├── kmin/
│   │   └── regression/       (only active result kept here)
│   ├── pytorch_resnet/
│   └── sigma/
├── scripts/
│   ├── archive/              (NEW — 6 legacy scripts)
│   ├── run_generation_flexible.py
│   └── job_flexible.sh
├── src/
│   ├── fractal_classifier_01/
│   ├── fractal_classifier_02/   (MOVED — was nested inside 01)
│   ├── dataset_generator.py
│   ├── param_sampler.py
│   ├── plot_performance.py
│   └── pyFC_lib/
├── tests/
├── CLAUDE.md
├── DELETED.md                (NEW — audit trail)
├── README.md                 (updated)
├── environment.yml
├── requirements_py311.txt
└── requirements_py39.txt     (renamed from .txt.txt)
```

---

## Out of Scope

- Running or completing unexecuted notebooks (`loss_benchmark.ipynb`, `multi_parameter_regression.ipynb`)
- Adding model unit tests
- Writing thesis methodology documentation
- Modifying any data or trained model weights
