# Hyperparameter-Importance Analysis

`analyze_hyperparameter_importance.py` performs a read-only, CPU-side analysis of an already completed `grid_search` and `candidate_retraining` result tree. It does not retrain models and does not use the held-out test set for model selection.

## Run

For the current refined multitask experiment:

```bash
python -u analyze_hyperparameter_importance.py \
  --run-root <project>/turbulens/training/outputs/multitask/local_v2 \
  --overwrite
```

Default output:

```text
<run-root>/hyperparameter_analysis/
```

## Inputs

The script reads the compact artifacts retained by the pipeline, including configuration JSON, validation metrics, training history, completion markers, and stored selector outputs.

It analyzes both:

```text
<run-root>/grid_search/experiments/
<run-root>/candidate_retraining/experiments/
```

## Selection-score reconstruction

The analyzer reconstructs the exact validation-only selector score from stored metrics. If the pipeline's `ranked_configurations.json` exists, reconstructed scores/ranks are compared against the stored selector output.

This is an integrity check: the importance analysis should study the same objective that actually selected models.

## Marginal parameter effects

For each varied parameter and metric, the analyzer reports level-wise:

- sample/configuration count;
- mean;
- standard deviation;
- standard error;
- approximate 95% interval for the mean;
- median;
- quartiles;
- minimum and maximum.

Metrics include the selection score, validation loss, MAE, MSE, RMSE, R2, and per-target summaries.

## One-factor eta-squared

For each parameter, one-factor eta-squared is reported as a descriptive fraction of metric variance associated with differences between that factor's observed levels.

This is a screening statistic, not a causal effect size. In a factorial hyperparameter search, interactions can make marginal effects misleading.

## Random-forest permutation importance

For the grid stage, a random-forest surrogate model is trained with cross-validation and permutation importance is evaluated on held-out folds. This provides a nonlinear diagnostic that can capture interactions missed by marginal averages.

It remains descriptive: it measures predictive importance within the sampled grid, not causality.

## Pairwise interaction diagnostics

The analyzer computes descriptive two-factor interaction eta-squared for every pair of varied factors. The interaction term is based on departures of cell means from the additive main-effect expectation.

Output:

```text
grid_search/interaction_importance_summary.csv
```

This was added because the first experiment showed a strong conditional reversal for batch size: the unconditional marginal ranking differed from the ranking within the dominant pretrained / high-learning-rate regime.

## Conditional-best-regime analysis

The analyzer identifies the strongest observed levels of dominant factors and recomputes remaining marginal effects within that restricted subset.

Outputs:

```text
grid_search/conditional_best_regime/context.json
grid_search/conditional_best_regime/parameter_level_summary.csv
grid_search/conditional_best_regime/marginal_importance_summary.csv
grid_search/conditional_best_regime/parameter_effects/
```

This analysis is specifically intended to reveal interaction-driven reversals. Because the conditioning levels are selected from the same validation grid, the result should be treated as a diagnostic for designing the next experiment, not as an unbiased inferential estimate.

## Small-grid versus large-candidate transfer

Configurations are matched by `configuration_id` across the two stages. The script reports Pearson/Spearman associations between small-data and large-data validation metrics and ranks.

Outputs:

```text
cross_stage/grid_candidate_comparison.csv
cross_stage/grid_candidate_correlations.json
```

Weak rank correlation suggests that the small dataset should be used for broad screening rather than aggressive final pruning.

## Learning curves

The script creates:

- one train/validation loss plot for every candidate run;
- candidate overlays grouped by architecture;
- matched small-grid versus large-candidate curves for identical configurations;
- architecture-level median/IQR envelopes over grid runs.

These plots help diagnose optimization speed, overfitting, scheduler behavior, and whether the epoch budget is adequate.

## Main outputs

```text
hyperparameter_analysis/
├── ANALYSIS_REPORT.md
├── analysis_summary.json
├── selection_reconstruction_verification.json
├── grid_search/
│   ├── configuration_metrics.csv
│   ├── run_metrics.csv
│   ├── parameter_level_summary.csv
│   ├── marginal_importance_summary.csv
│   ├── rf_permutation_importance.csv
│   ├── interaction_importance_summary.csv
│   ├── conditional_best_regime/
│   ├── importance/
│   ├── parameter_effects/
│   └── per_target/
├── candidate_retraining/
├── cross_stage/
└── curves/
```

## Interpretation discipline

Use the analysis to design a **new experiment version**. Do not alter hyperparameters inside an already completed run root.

For example, after analyzing `local_v2`, a scientifically changed follow-up should use `local_v3` rather than modifying `local_v2` in place.
