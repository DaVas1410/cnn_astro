# Bayesian Ensemble — Colab Training Notebook

**Date:** 2026-07-04
**Status:** Approved

## Goal

Provide a Google Colab notebook to train the existing Bayesian deep-ensemble
joint-regression model (`k_min` / `k_max` / `sigma` with heteroscedastic
uncertainty) — the same model as `scripts/hpc_ensemble/` and `scripts/local_run/`
— on a Colab GPU, using code and dataset both hosted in the user's Google Drive.

## Decisions

- **Code + data location:** both live in Google Drive. The notebook mounts Drive
  and imports the existing `scripts/hpc_ensemble/` modules directly (no repo
  clone, no code duplication).
- **Members:** 3 (real epistemic uncertainty via member disagreement, ~60% of the
  full 5-member time). Configurable via one variable.
- **Run mode:** single run, no resume/checkpoint-restart logic. Results written to
  a Drive folder so they survive VM recycling.
- **Tracked config untouched:** overrides are applied in-memory and written to a
  throwaway `config_colab.yaml` (same pattern as `scripts/local_run/run.sh`).

## Architecture

Reuse, unchanged, from `scripts/hpc_ensemble/`:
- `model.ResNet50HeteroJoint` — ResNet50 + heteroscedastic head, output `(B, 3, 2)`
  = (μ sigmoid-bounded, log_σ).
- `loss.WeightedGaussianNLL` — k_max weighted 2×.
- `dataset.load_splits`, `denorm`, `uncertainty_to_orig` — HDF5 loading, p1/p99
  image norm, [0,1] target norm, split by `base_seed`.
- `config.load_config`, `output_dir`.
- Training loop from `train_member.py` and the ensemble-combine + metrics + plots
  from `evaluate_ensemble.py`.

**Adaptation vs HPC:** the SLURM path shells out to `python train_member.py` per
member; the notebook imports the functions and calls them in-process so output
renders inline. Normalization, model, loss, and split logic are identical, so
metrics are directly comparable to HPC/local runs.

## Notebook cells

1. GPU check — `nvidia-smi`, `torch.cuda.is_available()`, warn if CPU-only.
2. Mount Google Drive.
3. **EDIT-ME config cell** — `REPO_DIR`, `DATA_FILE`, `OUTPUT_DIR`, `N_MEMBERS=3`,
   `EPOCHS`, `BATCH_SIZE`.
4. Install deps Colab lacks (`h5py`, `pyyaml`); add `scripts/hpc_ensemble` to
   `sys.path`.
5. Build runtime config — load tracked `config.yaml`, override
   `hpc.project_dir` / `data.file` / `output.dir` / `ensemble.n_members` /
   `training.epochs` / `training.batch_size`, write `config_colab.yaml`.
6. Dataset sanity check — open the `.h5`, print image shape and sample param
   ranges, confirm they fall inside the config's normalization bounds.
7. Train members 0..N-1 sequentially; per-epoch train/val loss printed;
   `best_model.pt` + `history.json` saved per member to Drive.
8. Evaluate ensemble — metrics table (MAE/RMSE/R² + σ_total), `results.json`,
   `results_summary.txt`, and 3 plots (scatter, residuals, calibration) shown
   inline and saved to Drive.
9. Training-curve plot per member.

## Out of scope

- Resume / checkpoint-restart logic.
- Repo cloning (code is in Drive).
- Any edit to the tracked `config.yaml`.
- Inference on real GASS FITS data (covered by the existing inference notebook).

## Output artifact

`notebooks/comparison/bayesian_ensemble_colab_train.ipynb`
