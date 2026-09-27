# Fractal Image Regression Pipeline

A configuration-driven, resumable PyTorch workflow for regression of physical parameters (`k_min`, `k_max`, `sigma`, `beta`) from fractal images. Supports a shared multitask model and four independently optimized single-target models, followed by deep-ensemble evaluation, latent-representation analysis, and a paired multitask-versus-single-target comparison.

**Release:** `2.0.0-local`
**Canonical experiment:** `local_v2`
**Execution model:** local Linux GPU server/workstation, no Slurm dependency

The current search design was refined from the completed `local_v1` experiment using post-hoc marginal effects, pairwise interaction analysis, conditional-regime analysis, and small-data to large-data rank transfer. The rationale is documented in [`docs/GRID_REFINEMENT_RATIONALE.md`](docs/GRID_REFINEMENT_RATIONALE.md).

## Scientific workflow

```text
existing train / validation / test HDF5 splits
                 |
                 v
         validate_data
                 |
                 v
     108-run small-data grid
                 |
                 v
 validation-only grouped selection
   8 candidates / architecture
          24 configurations
                 |
                 v
 large-data candidate retraining
       2 seeds / configuration
             48 runs
                 |
                 v
 validation-only final configuration
                 |
                 v
       5-member deep ensemble
                 |
                 +---------------------+
                 |                     |
                 v                     v
       final validation/test     interpretability
           evaluation           PCA / CCA / UMAP
                 |
                 v
              report
```

The same search and ensemble procedure is available for `k_min`, `k_max`, `sigma`, and `beta` individually. Once all five final ensembles exist, `compare_multitask_single.py` performs the paired final comparison.

## Repository layout

```text
turbulens/training/
├── README.md, CHANGELOG.md, VERSION.txt
├── pipeline.py, train_regression.py, select_experiments.py
├── evaluate_ensemble.py, interpret_embeddings.py, analyze_hyperparameter_importance.py
├── compare_multitask_single.py, generate_single_target_configs.py, check_environment.py
├── run_pipeline.sh, start_pipeline.sh, stop_pipeline.sh, status_pipeline.sh
├── run_all_experiments.sh, start_all_experiments.sh
├── configs/
│   ├── README.md, config_base.yaml, config_multitask.yaml, comparison.yaml
│   └── config_single_{k_min,k_max,sigma,beta}.yaml
├── pipeline_lib/
│   └── common.py, config.py, data.py, gpu_guard.py, metrics.py, model.py, pca_backend.py
├── tools/
│   └── interpret_legacy_ensemble.py   # analyzes ensembles from the pre-refactor workflow only
├── tests/
│   └── test_config_and_grid.py, test_model_and_metrics.py
└── docs/
    ├── METHODOLOGY.md, RUNBOOK.md, GRID_REFINEMENT_RATIONALE.md
    ├── HYPERPARAMETER_ANALYSIS.md, CONFIGURATION_REFERENCE.md
    ├── SCIENTIFIC_BEHAVIOR.md, OUTPUT_LAYOUT.md
    ├── REPRODUCIBILITY_CHECKLIST.md, CONFIGURABILITY_AUDIT.md
    └── MIGRATION_FROM_SLURM.md
```

## Refined `local_v2` grid

The canonical search space in [`configs/config_base.yaml`](configs/config_base.yaml) is:

| Parameter | Values |
|---|---|
| Architecture | ResNet18, ResNet34, ResNet50 |
| Input channels | 1, 3 |
| Initialization | ImageNet pretrained only |
| Dropout | 0.2, 0.3, 0.4 |
| Learning rate | 5e-4, 1e-3, 2e-3 |
| Batch size | 64, 128 |
| Weight decay | 1e-4 |
| Loss | Smooth L1 |
| Smooth L1 beta | 0.05 |

This gives `3 x 2 x 1 x 3 x 3 x 2 = 108` grid configurations per task.

The exact torchvision pretrained recipes are pinned to avoid version-dependent `Weights.DEFAULT` behavior:

- ResNet18: `IMAGENET1K_V1`
- ResNet34: `IMAGENET1K_V1`
- ResNet50: `IMAGENET1K_V2`

Training batch size is a true search parameter. There is no automatic batch-size probe, fallback batch size, reference batch size, or runtime substitution.

## Key scientific safeguards

- **No test-set model selection.** Grid and candidate stages are configuration-validated with `evaluate_test: false`. Test evaluation is enabled only for the final selected ensemble.
- **Training-only image normalization.** Robust 0.5/99.5 percentiles are estimated from training images only and then reused for validation/test images.
- **Validation-only latent fitting.** PCA, CCA preprocessing, and UMAP are fitted on validation embeddings; held-out test embeddings are transformed afterward.
- **Explicit model and optimizer behavior.** Dropout policy, AdamW options, ReduceLROnPlateau settings, early stopping, pretrained weight recipes, seeds, and target ranges are all represented in configuration/provenance.
- **No target-specific loss weighting.** `local_v2` uses ordinary Smooth L1 on range-normalized targets.
- **Consistent ensemble aggregation.** Reported bounded predictions are the equal-weight mean of member-wise bounded predictions. Raw normalized means are retained separately for diagnostics.
- **Uncertainty language is conservative.** Across-member SD is descriptive ensemble spread, not automatically a calibrated prediction/confidence interval. Metric bootstrap intervals are reported separately.
- **Paired final comparison.** The multitask-versus-single-target comparison checks matching samples/ground truth, reports paired bootstrap differences and sign-flip randomization tests, and applies Holm-Bonferroni adjustment across the four target-wise tests.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full methods-level description and [`docs/SCIENTIFIC_BEHAVIOR.md`](docs/SCIENTIFIC_BEHAVIOR.md) for reproducibility invariants.

## Data format

Each HDF5 file must contain:

```text
/images
/parameters/k_min
/parameters/k_max
/parameters/sigma
/parameters/beta
```

Supported image storage layouts are `(N,H,W)`, `(N,C,H,W)`, and `(N,H,W,C)` with 1 or 3 stored channels. Task-specific configurations only require the target datasets used by that task. The pipeline does **not** create random train/validation/test splits; the existing split directories (configured under `data.small_root`/`data.large_root` in `configs/config_base.yaml`) are authoritative and are validated before training.

## Installation

This pipeline is part of the `turbulens` package; its dependencies are the root project's `training` (and, for tests, `test`) extras:

```bash
uv sync --extra training --extra test
# or: pip install -e ".[training,test]"
```

`umap-learn` is optional; with the default `interpretability.umap.policy: optional`, UMAP is skipped cleanly when it is unavailable.

The shell launchers do **not** source `conda.sh`. They use `$CONDA_ENV/bin/python`, `$PIPELINE_PYTHON`, or the active `python` directly, so they are safe under `nohup` and non-interactive shells.

## Preflight before production

From `turbulens/training/`:

```bash
python -u check_environment.py --config configs/config_multitask.yaml
```

The canonical multitask preflight should report:

```text
Grid runs        : 108
Grid candidates  : 24
Candidate runs   : 48
Ensemble members : 5
```

Then inspect the execution plan without training:

```bash
bash run_pipeline.sh configs/config_multitask.yaml --dry-run --print-plan
```

## Recommended production workflow

For manuscript-quality calculations, use a two-phase multitask run so the refined search can be inspected before final ensemble/test evaluation.

### Phase 1: search and large-data confirmation

```bash
nohup bash run_pipeline.sh configs/config_multitask.yaml \
  --to-stage select_final \
  > multitask_local_v2_search.log 2>&1 &
```

Monitor:

```bash
tail -f multitask_local_v2_search.log
```

### Analyze the refined search

After `select_final` completes:

```bash
python -u analyze_hyperparameter_importance.py \
  --run-root <output_root>/multitask/local_v2 \
  --overwrite
```

The analysis is read-only. It evaluates marginal effects, pairwise interactions, conditional best-regime effects, random-forest permutation importance, per-target metrics, candidate learning curves, and small-grid versus large-candidate transfer.

### Phase 2: final ensemble and held-out evaluation

If the refined search is accepted, continue the **same experiment/configuration**:

```bash
nohup bash run_pipeline.sh configs/config_multitask.yaml \
  --from-stage train_ensemble \
  > multitask_local_v2_final.log 2>&1 &
```

Using the same YAML preserves the configuration hash and provenance across both phases.

## Single-target models and final comparison

Run the four single-target pipelines individually:

```bash
bash run_pipeline.sh configs/config_single_k_min.yaml
bash run_pipeline.sh configs/config_single_k_max.yaml
bash run_pipeline.sh configs/config_single_sigma.yaml
bash run_pipeline.sh configs/config_single_beta.yaml
```

Or, once satisfied with the canonical grid and wanting the complete sequential suite:

```bash
bash run_all_experiments.sh          # foreground
bash start_all_experiments.sh        # nohup background
```

`run_all_experiments.sh`/`start_all_experiments.sh` execute multitask, the four single-target tasks, then the final paired comparison, in order:

```text
multitask -> single_k_min -> single_k_max -> single_sigma -> single_beta -> final comparison
```

Valid completed stages/experiments are skipped safely on rerun; the sequence stops immediately if one experiment fails.

To run only the final comparison after all ensembles exist:

```bash
python -u compare_multitask_single.py \
  --config configs/comparison.yaml \
  --overwrite
```

## Resume, status, and safe stopping

Rerun the same command to resume. Completed stages and completed individual experiments are protected by completion markers and configuration hashes.

```bash
bash status_pipeline.sh configs/config_multitask.yaml
bash stop_pipeline.sh configs/config_multitask.yaml   # graceful SIGTERM
```

Do not edit scientific YAML values while reusing the same `experiment.version`. For a changed scientific configuration, increment the version (e.g. `local_v3`) instead of bypassing the hash guard.

## GPU auto-wait

On a shared workstation, leave the guard enabled in `configs/config_base.yaml`:

```yaml
runtime:
  gpu_wait:
    enabled: true
    poll_seconds: 30
    stable_seconds: 60
    timeout_seconds: 0   # wait indefinitely
```

To disable it on a dedicated machine, set `enabled: false`. Environment overrides for ad hoc runs:

```bash
PIPELINE_PYTHON=/path/to/env/bin/python bash run_pipeline.sh configs/config_multitask.yaml
CONDA_ENV=/path/to/conda/env bash run_pipeline.sh configs/config_multitask.yaml
PIPELINE_GPUS="0 1" bash run_pipeline.sh configs/config_multitask.yaml
```

## Results and provenance

Multitask outputs are written under `<output_root>/multitask/local_v2/` (`<output_root>` is `experiment.output_root` in `configs/config_base.yaml`; in this repo that resolves under `outputs/`, matched by `turbulens.models.registry.EnsembleRegistry`). Important provenance files:

```text
resolved_config.yaml, resolved_config.json, resolved_config_hash.txt
environment.json, source_provenance.json
data_validation/, pipeline_state/
grid_search/selection/, candidate_retraining/selection/
ensemble/final_configuration.json, ensemble/evaluation/
report/, hyperparameter_analysis/
```

Grid and candidate runs use compact storage profiles. Final ensemble members retain checkpoints, predictions, plots, histories, and environment information required for final evaluation and reproducibility. See [`docs/OUTPUT_LAYOUT.md`](docs/OUTPUT_LAYOUT.md).

## Interpretability

The integrated `interpretability` stage analyzes every final ensemble member and a consensus representation: extracts the shared 256-dimensional representation (or the backbone, if configured), fits PCA and CCA on validation embeddings only, projects held-out test embeddings into those axes, correlates components with true/predicted physical parameters, and optionally runs UMAP. `tools/interpret_legacy_ensemble.py` is included only for analyzing ensembles produced by the older pre-refactor workflow.

## Tests and release validation

```bash
python -m pytest turbulens/training/tests
```

The release validation report is [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md). It records configuration/grid checks, static checks, a synthetic CPU end-to-end pipeline run, and resume verification; these checks validate orchestration and invariants rather than production results. The production `local_v2` GPU experiment (multitask + all four single-target configurations, plus the multitask-vs-single-target comparison) has since been run; see `VALIDATION_REPORT.md`'s "Production run addendum" and `outputs/comparison/local_v2/comparison_report.md` for results.

## Documentation

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): manuscript-oriented scientific methodology
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): exact operational workflow and commands
- [`docs/GRID_REFINEMENT_RATIONALE.md`](docs/GRID_REFINEMENT_RATIONALE.md): evidence supporting the `local_v2` search
- [`docs/HYPERPARAMETER_ANALYSIS.md`](docs/HYPERPARAMETER_ANALYSIS.md): post-hoc analysis outputs and interpretation
- [`docs/CONFIGURATION_REFERENCE.md`](docs/CONFIGURATION_REFERENCE.md): configuration field reference
- [`docs/SCIENTIFIC_BEHAVIOR.md`](docs/SCIENTIFIC_BEHAVIOR.md): scientific/reproducibility invariants
- [`docs/OUTPUT_LAYOUT.md`](docs/OUTPUT_LAYOUT.md): result-tree and retention policy
- [`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md): pre-publication QC checklist
- [`docs/CONFIGURABILITY_AUDIT.md`](docs/CONFIGURABILITY_AUDIT.md): audit of hidden/default behavior
- [`docs/MIGRATION_FROM_SLURM.md`](docs/MIGRATION_FROM_SLURM.md): differences from the earlier scheduler-based workflow
- [`configs/README.md`](configs/README.md): canonical and task-specific configurations

## Reproducibility and leakage protection

- Train/validation/test directories are validated before use.
- Image normalization is estimated only from training images.
- Hyperparameter and final-model selection use validation metrics only.
- The held-out test set is reserved for final reporting/evaluation.
- Ensemble members use explicit independent seeds and a shared normalization seed.
- Important JSON/config/checkpoint files use atomic writes.
- Completion/failure markers make resume behavior explicit.
- Existing run roots are protected by a resolved configuration hash.
- PCA/CCA/UMAP latent axes are fitted on validation data and test data are transformed afterward.

## Publication use

For a paper, archive the exact release (`VERSION.txt`, `MANIFEST.sha256`) together with the resolved configuration, environment capture, data identifiers/checksums, final selection files, evaluation outputs, and analysis report.
