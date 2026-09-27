# Scientific Behavior and Reproducibility Guarantees

## Local execution versus Slurm

This repository executes training directly on a local Linux server/workstation. Removing Slurm changes orchestration only. It does not intentionally alter the model, preprocessing, optimization, validation selection, final evaluation, ensemble aggregation, or interpretability mathematics.

## Configuration source of truth

Scientific settings are supplied by YAML. `pipeline_lib/config.py` validates them but does not contain a hidden fallback copy of the search grid.

The canonical experiment version is `local_v2`.

## Batch size

Training batch size is enumerated directly from `search.space.batch_size` and passed unchanged to the PyTorch DataLoader.

There is no automatic GPU batch-size probe, fallback/reference batch size, clamping, or replacement logic in this package.

Analysis-stage batching parameters such as `interpretability.batch_size` and PCA/UMAP transform batch sizes are computational controls and are not training hyperparameters.

## Explicit fixed methodology

Several settings that were previously implicit library/model defaults are now explicit in YAML, including:

- AdamW betas, epsilon, and AMSGrad flag;
- ReduceLROnPlateau threshold, threshold mode, cooldown, and epsilon;
- the mapping from searched base dropout to shared/head dropout values;
- the exact torchvision ImageNet weight recipe for each ResNet backbone.

This improves auditability without changing the intended numerical architecture of the prior model.

## Target weighting

The canonical `local_v2` search uses ordinary `smooth_l1`. No target-specific loss weights are active.

No target-specific loss-weight pathway is active or shipped in this release. The canonical and implemented training loss is ordinary `smooth_l1` on range-normalized targets.

## Randomness

Seeds are explicit for grid screening, candidate retraining, ensemble members, augmentation, and normalization sampling.

`training.deterministic: false` preserves the previous high-performance CUDA behavior. Therefore independent runs with the same seed are not guaranteed to be bitwise identical across GPU architectures, CUDA/cuDNN versions, or all kernels.

Set deterministic mode only as a deliberate new experimental configuration because it can alter available kernels/performance.

## Test-set isolation

Grid and candidate configurations set `evaluate_test: false`. Final test evaluation happens only after the final hyperparameter configuration has been selected on validation data.

## Interpretability isolation

PCA, CCA preprocessing, and UMAP are fitted on validation embeddings. Held-out test embeddings are transformed afterward.

## Ensemble uncertainty

Across-member standard deviation is descriptive ensemble spread. It is not automatically a calibrated confidence or prediction interval. Bootstrap confidence intervals for evaluation metrics are reported separately.

## Ensemble aggregation

Final reported ensemble predictions average the member-wise bounded predictions. The raw normalized mean is retained separately for diagnostic metrics. This explicit distinction prevents the nonlinear clipping operation from being applied in a mathematically inconsistent order.

## Paired final comparison

The multitask-versus-single-target comparison verifies matching test sample keys and ground truth, uses paired bootstrap differences, and reports a paired sign-flip randomization test for absolute-error differences. Holm-Bonferroni adjusted p-values are reported across the four target-wise randomization tests.
