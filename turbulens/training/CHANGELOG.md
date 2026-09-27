# Changelog

## local_v2 production run (2026-09-16)

- Executed the production `local_v2` grid for real (multitask + all four single-target configurations, full `validate_data -> ... -> report` chain each) plus the `compare_multitask_single.py` comparison stage. Previously only the CPU packaging smoke test had been run (see `VALIDATION_REPORT.md`).
- Results: multitask test R2 of 0.9997 (`k_min`), 0.9795 (`k_max`), 0.9990 (`sigma`), 0.9203 (`beta`).
- The multitask-vs-single-target comparison found single-target models statistically significantly but only marginally ahead of multitask on `k_min`/`k_max`/`sigma`, multitask clearly ahead on `beta`, and the single-target system costing roughly 5x the total training time and 4x the parameters of one shared multitask model. This reverses the working assumption recorded in `turbulens/models/architecture.py`'s docstring prior to this run; the docstring has been updated accordingly.

## 2.0.0-local

Scientific/configuration cleanup and refined-search release.

### Refined search

- Promoted the evidence-based `local_v2` search to the canonical configuration.
- Grid reduced to 108 configurations: 3 architectures x 2 channel modes x pretrained only x 3 dropout values x 3 learning rates x 2 batch sizes.
- Learning rates refined to `[5e-4, 1e-3, 2e-3]` around the first study's best boundary.
- Batch sizes refined to `[64, 128]` based on interaction-aware analysis and large-data candidate behavior.
- Scratch initialization removed from the refined grid because pretraining strongly dominated the first study.
- Candidate retention widened to 8 configurations per architecture (24 total) because small-data to large-data ranking transfer was weak.
- Candidate retraining increased to 2 independent seeds per configuration (48 large-data candidate runs).

### Reproducibility and scientific transparency

- Canonical experiment version is now `local_v2` for multitask and all single-target configurations.
- `comparison.yaml` now targets the `local_v2` result roots.
- Removed the extra `config_multitask_refined_v2.yaml`; the canonical config contains the refined science, and CLI `--to-stage` / `--from-stage` controls phases without changing the configuration hash.
- Made the model's layer-specific dropout policy explicit in YAML instead of hiding it in `model.py`.
- Made AdamW betas/epsilon/AMSGrad and ReduceLROnPlateau threshold/cooldown/epsilon explicit in YAML and pass them explicitly to PyTorch.
- Added strict integer validation so values such as a floating-point batch size cannot be silently truncated.
- Added resolved-file split-overlap detection to catch the same HDF5 file/symlink appearing in multiple splits.
- Fixed HDF5 metadata inspection to cache image shape while the file handle is open.
- Removed the inactive target-weighted loss pathway from the production release. `local_v2` uses ordinary `smooth_l1` only, eliminating the historical `loss_weights` ambiguity.
- Pinned exact torchvision pretrained-weight recipes (`IMAGENET1K_V1` for ResNet18/34 and `IMAGENET1K_V2` for ResNet50) instead of relying on `Weights.DEFAULT`.
- Made final bounded-ensemble aggregation mathematically explicit as the mean of member-wise bounded predictions.
- Added Holm-Bonferroni correction across the four target-wise paired absolute-error randomization tests in the final multitask-vs-single-target comparison.

### Orchestration

- Retained the corrected `run_all_experiments.sh` local-variable handling that avoids the earlier `label: unbound variable` failure.
- Sequence logs now default to the results tree (`<output_root>/sequence_logs/<version>`) instead of the source repository.
- No Slurm-specific submission logic or machine-specific project/environment paths are present.
- Training batch size remains a direct grid parameter; there is no automatic batch-size probe or replacement.

### Analysis

- Hyperparameter analysis includes pairwise interaction diagnostics and a conditional-best-regime analysis in addition to marginal effects and random-forest permutation importance.
- Selection-score reconstruction remains available to verify that post-hoc analysis matches the pipeline's stored validation ranking.

### Documentation and validation

- Rewrote the README around the refined two-phase workflow.
- Added `docs/METHODOLOGY.md` as a methods-level scientific description.
- Added `docs/RUNBOOK.md` with exact operational commands.
- Added `docs/CONFIGURATION_REFERENCE.md`.
- Added `docs/GRID_REFINEMENT_RATIONALE.md` with evidence from the first study.
- Updated output, scientific-behavior, configurability, hyperparameter-analysis, and Slurm-migration documentation.
- Added `docs/REPRODUCIBILITY_CHECKLIST.md` and `configs/README.md`.
- Added pytest tests plus development dependency/configuration files.
- Child YAML configurations can now narrow the architecture list cleanly; inherited pretrained-weight entries for inactive architectures are pruned from the resolved configuration.

## 1.2.2-local

- Added the first refined multitask configuration and interaction-aware hyperparameter analysis.

## 1.2.1-local

- Made `training.loss_weights` optional for ordinary `smooth_l1`.

## 1.2.0-local

- Removed machine-specific paths and hidden scientific Python defaults; audited batch-size handling and model configurability.

## 1.1.0-local

- Added post-hoc hyperparameter-importance and learning-curve analysis.
