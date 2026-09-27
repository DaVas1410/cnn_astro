# Methodology and Reproducibility

This document defines the scientific workflow implemented by `fractal-regression-pipeline-v2.0.0`. It is written as a methods-level reference so that the computational experiment can be reproduced and described consistently in a manuscript.

## 1. Objective

The pipeline estimates one or more physical parameters from simulated or experimental fractal images:

- `k_min`
- `k_max`
- `sigma`
- `beta`

Two learning modes are supported:

1. **Multitask regression:** one shared ResNet encoder and shared latent representation jointly predict all configured targets.
2. **Single-target regression:** one independent model family is optimized for one target at a time.

The default study compares the final multitask ensemble against four independently optimized single-target ensembles.

## 2. Data organization and split policy

Datasets are supplied as HDF5 files in pre-existing `train`, `val`, and `test` directories. The pipeline does not reshuffle or regenerate these splits.

For each data scale (`data_small` and `data_large`) the expected layout is:

```text
train/*.h5
val/*.h5
test/*.h5
```

Each HDF5 file must provide:

```text
/images
/parameters/<target>
```

for each configured target.

The validation stage checks:

- required HDF5 datasets;
- image dimensionality and channel layout;
- consistent image geometry across files;
- target vector length;
- finite target values;
- target values within the configured physical bounds;
- nonempty train/validation/test splits;
- no overlap in resolved HDF5 file paths across train/validation/test.

The last check catches accidental use of the same file or symlink in more than one split. It does not attempt expensive content-level duplicate detection across independently copied files.

## 3. Image representation

Supported individual image layouts are:

```text
(H, W)
(C, H, W)
(H, W, C)
```

where `C` is 1 or 3.

A one-channel model can consume an RGB-stored image by channel averaging. A three-channel model can consume a single-channel stored image by repeating the channel three times.

### 3.1 Robust intensity normalization

Intensity normalization statistics are estimated **only from training images**.

The default percentiles are:

```text
p_low  = 0.5th percentile
p_high = 99.5th percentile
```

Sampling is bounded by `normalization.max_images` and `normalization.max_pixels` and uses a fixed normalization seed.

Each image is transformed as:

```text
x_scaled = clip((x - p_low) / (p_high - p_low), 0, 1)
```

Non-finite image pixels, if present, are replaced by the fitted lower normalization value before scaling.

### 3.2 ImageNet standardization

When `input_standardization: imagenet`, robustly scaled inputs are standardized after augmentation.

For three-channel models, the standard ImageNet mean and standard deviation are applied per channel.

For one-channel models, the channel-wise ImageNet means and standard deviations are averaged to form one scalar mean and one scalar standard deviation.

## 4. Target normalization

Physical targets are mapped to `[0,1]` using predefined physical ranges rather than statistics estimated from the validation or test sets.

For target `y` with configured range `[a,b]`:

```text
y_norm = (y - a) / (b - a)
```

The default multitask ranges are:

```text
k_min : [ 1, 32]
k_max : [34, 64]
sigma : [ 0,  5]
beta  : [-3, -1]
```

Predictions remain unconstrained during optimization. During reported evaluation they are also evaluated in a clamped normalized form, `clip(prediction, 0, 1)`, and converted back to physical units.

## 5. Data augmentation

Training augmentation consists of orientation-preserving symmetries appropriate for isotropic image statistics:

- random rotations by 0, 90, 180, or 270 degrees;
- optional horizontal flip;
- optional vertical flip.

The augmentation transform for a sample is deterministically generated from the training seed, epoch number, and sample index. Validation and test samples are never augmented.

Because 90-degree rotation is used, the default configuration requires square images.

## 6. Model architecture

The model uses an ImageNet ResNet backbone selected from:

```text
ResNet18
ResNet34
ResNet50
```

The original fully connected classification layer is removed.

When `preserve_resolution: true`, the standard ResNet max-pooling layer after the first convolution is replaced by an identity operation. This is an intentional architectural choice preserved from the prior pipeline.

### 6.1 Pretraining and channel adaptation

The refined `local_v2` search uses ImageNet-pretrained backbones only because pretraining strongly dominated scratch initialization in the completed `local_v1` study.

For a pretrained one-channel model, the first convolutional kernel is initialized by averaging the three pretrained RGB input-channel kernels. For a pretrained three-channel model, standard ImageNet weights are used unchanged. The exact torchvision recipes are pinned rather than resolved through a `DEFAULT` alias: ResNet18 and ResNet34 use `IMAGENET1K_V1`, while ResNet50 uses `IMAGENET1K_V2`.

### 6.2 Shared representation

The default shared multilayer perceptron is:

```text
backbone features
 -> Linear(..., 512) -> LayerNorm -> GELU -> Dropout
 -> Linear(512, 256) -> LayerNorm -> GELU -> Dropout
```

Each target then has an independent head:

```text
Linear(256, 96) -> GELU -> Dropout -> Linear(96, 1)
```

### 6.3 Explicit dropout policy

The searched `dropout` value is a base dropout `d`. The layer-specific mapping is explicitly recorded in YAML:

```text
shared layer i = min(shared_max, d + i * shared_increment_per_layer)
head           = min(head_max, d * head_multiplier)
```

Default fixed policy:

```text
shared_increment_per_layer = 0.10
shared_max                 = 0.60
head_multiplier            = 0.50
head_max                   = 0.35
```

This reproduces the previous model behavior while eliminating a hidden architectural transformation.

## 7. Loss and optimization

The refined search uses ordinary Smooth L1 loss on normalized targets:

```text
loss = SmoothL1(prediction_norm, target_norm; beta=0.05)
```

All targets therefore contribute equally after target-range normalization.

No target-specific loss weights are active in `local_v2`.

The optimizer is AdamW with explicitly recorded settings:

```text
betas   = (0.9, 0.999)
eps     = 1e-8
amsgrad = false
```

Learning rate and weight decay come from the selected hyperparameter configuration.

Gradient norm clipping uses a maximum L2 norm of 1.0.

Mixed precision is enabled for CUDA training.

## 8. Learning-rate schedule and early stopping

The scheduler is `ReduceLROnPlateau`, monitoring validation loss with:

```text
factor         = 0.5
patience       = 5
min_lr         = 1e-7
threshold      = 1e-4
threshold_mode = rel
cooldown       = 0
eps            = 1e-8
```

Training allows up to 200 epochs.

Early stopping monitors validation loss with:

```text
patience = 20
min_delta = 0
```

The best checkpoint is the epoch with the lowest validation loss.

## 9. Refined hyperparameter search

The `local_v2` grid is:

```text
architecture    : resnet18, resnet34, resnet50
input channels  : 1, 3
pretraining     : true
dropout         : 0.2, 0.3, 0.4
learning rate   : 5e-4, 1e-3, 2e-3
batch size      : 64, 128
weight decay    : 1e-4
loss            : smooth_l1
Smooth L1 beta  : 0.05
```

The factorial grid contains:

```text
3 x 2 x 1 x 3 x 3 x 2 = 108 configurations
```

One seed (`42`) is used in the small-data screening stage.

The grid is intentionally concentrated on the factors supported by the first experiment. `weight_decay` and `smooth_l1_beta` remain fixed because the original study did not vary them, so their effects cannot be inferred from those results.

## 10. Validation-only model selection

For direct comparability with the first experiment, the refined study retains the previous selection score.

For each configuration, per-target normalized MSE and R2 are calculated on the validation set. The pipeline forms macro averages across configured targets.

For all configurations in the selection pool:

```text
MSE_component = minmax(macro normalized MSE)
R2_component  = minmax(1 - macro normalized R2)
score         = MSE_component + R2_component
```

Lower scores are better.

The test set is not used to calculate this ranking.

The first study showed that MSE and R2 were nearly redundant in this dataset. The objective is nevertheless held fixed for `local_v2` so that the effect of changing the search region can be evaluated without simultaneously changing the ranking criterion.

## 11. Two-stage search strategy

### 11.1 Small-data screening

All 108 configurations are trained on `data_small`.

To avoid over-pruning based on a screening dataset whose ranking transferred only weakly to the large-data stage in `local_v1`, selection is stratified by architecture:

```text
8 best ResNet18 configurations
8 best ResNet34 configurations
8 best ResNet50 configurations
= 24 candidates
```

### 11.2 Large-data candidate confirmation

The 24 candidate configurations are retrained on `data_large` with two independent seeds:

```text
31415
27182
```

This produces 48 candidate runs.

For each configuration, validation MSE, RMSE, and R2 are aggregated across its repetitions. Final hyperparameter selection uses the aggregate validation score only.

## 12. Final deep ensemble

The single best large-data configuration is retrained as a five-member deep ensemble with seeds:

```text
42, 123, 456, 789, 2026
```

All members use the same training-only image-normalization sampling seed so that differences between ensemble members reflect model initialization/training stochasticity rather than different normalization samples.

The reported bounded ensemble prediction is the equal-weight arithmetic mean of the member-wise bounded predictions. In normalized space, this is `mean(clamp(member prediction, 0, 1))`; it is intentionally not replaced by `clamp(mean(raw member prediction), 0, 1)`, because clipping and averaging do not commute. The raw normalized ensemble mean is retained separately as a diagnostic quantity.

The sample standard deviation across bounded physical member predictions is reported as descriptive epistemic spread. It is not assumed to be a calibrated predictive or confidence interval.

## 13. Final evaluation

Only after model selection is complete does the final ensemble evaluate the held-out test split.

Reported metrics include, where applicable:

- MAE
- MSE
- RMSE
- R2
- Pearson correlation
- Spearman correlation
- bias
- residual standard deviation
- regression slope/intercept
- out-of-range prediction fraction

Bootstrap confidence intervals are produced for evaluation metrics. These metric confidence intervals are conceptually separate from the descriptive spread across ensemble members.

## 14. Multitask versus single-target comparison

Each single-target model family independently repeats the same refined search/candidate/ensemble procedure for one target.

The final comparison uses matching held-out test samples and evaluates multitask versus corresponding single-target predictions with paired analyses, bootstrap intervals, and a paired sign-flip randomization test on absolute-error differences. Because that randomization test is performed once for each of the four targets, Holm-Bonferroni adjusted p-values are reported across those four tests in addition to the raw p-values.

## 15. Interpretability

Interpretability is performed on final ensemble members only.

The default representation is the final shared 256-dimensional latent vector.

For each member:

1. embeddings are extracted for validation and test;
2. PCA is fitted on validation embeddings;
3. held-out test embeddings are transformed with the fitted PCA;
4. PCs are correlated with true and predicted physical parameters;
5. CCA is fitted using validation embeddings and true physical parameters;
6. the fitted CCA transformation is evaluated on held-out test data;
7. optional UMAP is fitted according to the configured policy.

### 15.1 Ensemble consensus representation

Raw latent coordinates from independently trained networks are not averaged because separate models can represent equivalent latent spaces up to rotations, sign flips, and permutations.

Instead:

1. each member embedding is standardized using validation statistics;
2. standardized member blocks are concatenated;
3. PCA/CCA/UMAP operate on this concatenated consensus representation.

## 16. Hyperparameter-importance analysis

`analyze_hyperparameter_importance.py` is post-hoc and read-only. It analyzes stored grid and candidate outputs and includes:

- marginal parameter-level summaries;
- one-factor eta-squared;
- random-forest permutation importance;
- pairwise two-factor interaction diagnostics;
- a conditional-best-regime analysis to detect marginal-effect reversals;
- per-target performance summaries;
- candidate training/validation curves;
- matched small-grid versus large-candidate comparisons;
- cross-stage rank and metric correlations.

These analyses are descriptive/model-diagnostic. They do not establish causal hyperparameter effects.

## 17. Reproducibility controls

The pipeline records:

- resolved YAML configuration;
- SHA-256 hash of the resolved scientific configuration;
- dataset manifests and fingerprints;
- train/validation/test sample indices;
- random seeds;
- normalization statistics;
- training history;
- selected configuration manifests;
- model checkpoints for final members;
- package/environment information;
- completion/failure markers.

Output roots are protected by the resolved configuration hash. A changed scientific configuration must use a new experiment version unless the user explicitly overrides the protection.

`training.deterministic` is `false` by default to preserve the previously used high-performance CUDA behavior. Therefore exact bitwise replication across independent GPU runs is not guaranteed. Seeded runs remain methodologically reproducible, but CUDA/library-level nondeterminism can produce small numerical differences.

## 18. Leakage policy

The intended information flow is:

```text
training data
  -> training + image normalization

validation data
  -> early stopping
  -> LR scheduling
  -> grid ranking
  -> candidate ranking
  -> final hyperparameter selection
  -> fitting PCA/CCA/UMAP axes

test data
  -> final ensemble evaluation
  -> final interpretability transformation/evaluation
  -> multitask-vs-single-target comparison
```

No test metric is used for hyperparameter ranking or model selection in the configured pipeline.

## 19. Methodological decisions intentionally not optimized in local_v2

The following parameters are held fixed because `local_v1` did not provide evidence for their optimization or because changing them would confound the targeted refinement:

- `weight_decay = 1e-4`
- `smooth_l1_beta = 0.05`
- shared MLP widths `[512,256]`
- target-head width `96`
- optimizer family AdamW
- scheduler family and patience settings
- early-stopping patience
- percentile normalization `0.5/99.5`
- augmentation policy

If `local_v2` confirms the new dominant search region, these can be studied in a later focused experiment rather than multiplied into the current factorial grid.
