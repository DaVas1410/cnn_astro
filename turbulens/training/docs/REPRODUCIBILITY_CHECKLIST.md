# Reproducibility and Release Checklist

Use this checklist before treating a run as a manuscript-quality production result.

## Before training

- [ ] Use a clean release directory; do not mix files from older pipeline versions.
- [ ] Record `VERSION.txt` and the repository archive checksum.
- [ ] Activate the intended Python environment.
- [ ] Run `python -u check_environment.py --config configs/config_multitask.yaml`.
- [ ] Confirm the preflight reports 108 grid runs, 24 candidates, 48 candidate runs, and 5 ensemble members.
- [ ] Confirm both dataset roots and all train/validation/test HDF5 files validate successfully.
- [ ] Confirm the resolved run root is the intended `.../multitask/local_v2` directory.
- [ ] Confirm the GPU list and visible devices are correct.
- [ ] Run `bash run_pipeline.sh configs/config_multitask.yaml --dry-run --print-plan`.
- [ ] Archive the exact YAML files used for the run if they differ from the release.

## During training

- [ ] Do not edit YAML files for an active `experiment.version`.
- [ ] Use graceful termination (`stop_pipeline.sh` or SIGTERM) rather than SIGKILL when possible.
- [ ] Retain the main launcher log and per-run logs until QC is complete.
- [ ] Check for failed runs before accepting a selection stage.
- [ ] Verify grid/candidate stages never evaluate the held-out test split.

## After refined search

- [ ] Run `analyze_hyperparameter_importance.py` on the completed multitask `local_v2` search.
- [ ] Review marginal effects, interaction effects, and conditional-regime diagnostics together.
- [ ] Review small-grid to large-candidate rank transfer before further narrowing the search.
- [ ] Inspect candidate learning curves for convergence, instability, and systematic early stopping.
- [ ] Record the validation-selected final configuration before final test evaluation.

## Final ensemble

- [ ] Confirm five completed ensemble members with the configured seeds.
- [ ] Confirm members use the same selected architecture/hyperparameters.
- [ ] Confirm members use the same training-derived normalization statistics.
- [ ] Confirm final ensemble evaluation uses the held-out test split only after selection.
- [ ] Treat across-member SD as descriptive epistemic spread, not a calibrated confidence interval.
- [ ] Report bootstrap confidence intervals for final metrics separately.

## Multitask versus single-target comparison

- [ ] Confirm identical test sample keys and ground-truth arrays.
- [ ] Confirm the same data roots, split definitions, normalization procedure, and ensemble seeds.
- [ ] Report paired bootstrap differences.
- [ ] Report sign-flip randomization p-values for paired absolute-error differences.
- [ ] Use the reported Holm-Bonferroni adjusted p-values across the four target-wise absolute-error tests.

## Archival provenance

Archive at minimum:

```text
resolved_config.yaml
resolved_config.json
resolved_config_hash.txt
environment.json
source_provenance.json
data_validation/
pipeline_state/
grid_search/selection/
candidate_retraining/selection/
ensemble/final_configuration.json
ensemble/evaluation/
report/
hyperparameter_analysis/
```

For long-term reproducibility also retain the exact release archive, its SHA-256 checksum, the dataset version/checksum or immutable source identifier, and the software environment used for the production run.
