# Refined Grid Rationale: local_v1 -> local_v2

## Purpose

This document records why the `local_v2` hyperparameter space differs from the first completed multitask search. The decisions are based on the completed `local_v1` post-hoc analysis rather than generic tuning heuristics.

## Evidence from local_v1

The completed grid contained 432 configurations:

```text
3 architectures
x 2 input-channel modes
x 2 pretraining states
x 3 dropout values
x 3 learning rates
x 4 batch sizes
= 432
```

Eighteen configurations were subsequently retrained on the large dataset.

### Learning rate

Learning rate was the dominant marginal factor for the validation selection score, with one-factor eta-squared of approximately `0.62`. The random-forest permutation analysis independently ranked learning rate as the strongest predictor of grid performance.

`1e-3` clearly outperformed `1e-4` and `1e-5` and was the best tested boundary. Therefore `local_v2` brackets this region by a factor of two:

```text
5e-4, 1e-3, 2e-3
```

### Pretraining

Pretraining had a strong main effect (eta-squared approximately `0.16`). All 18 configurations that survived into large-data candidate retraining were pretrained.

Therefore scratch initialization is removed from `local_v2`:

```text
pretrained = true
```

### Batch size and interaction effects

The unconditioned marginal summary suggested a small batch size could be favorable. That conclusion was not stable after accounting for interaction with the dominant factors.

Within the scientifically relevant subset:

```text
pretrained = true
learning_rate = 1e-3
```

mean validation selection score was approximately:

```text
batch 128 : 0.0299
batch  64 : 0.0332
batch  32 : 0.0560
batch  16 : 0.0757
```

Lower is better.

Within this conditional regime, the batch-size effect was substantial (conditional eta-squared approximately `0.60`). The large-data candidate stage independently favored batch 128 on average across several performance summaries.

Accordingly, `local_v2` retains only the two already demonstrated feasible larger batches:

```text
64, 128
```

The grid does not extrapolate to batch sizes above 128 because resource feasibility has not yet been demonstrated for every retained architecture/channel combination and the pipeline intentionally does not alter batch size automatically.

### Pairwise interactions

The strongest descriptive interactions in the first grid included approximately:

```text
pretraining x learning rate    eta2 ~ 0.115
architecture x learning rate  eta2 ~ 0.019
learning rate x batch size    eta2 ~ 0.014
pretraining x batch size      eta2 ~ 0.006
```

These values support retaining interaction-aware diagnostics in the analysis script. They also demonstrate why a marginal parameter plot alone is insufficient for deciding the next grid.

### Architecture

ResNet18 produced the best single large-data candidate, but ResNet34 and ResNet50 remained competitive and showed target-dependent strengths. Because the final project also compares single-target models, eliminating architecture families at this stage would be premature.

Retained:

```text
resnet18, resnet34, resnet50
```

### Input channels

The global channel-count effect was weak and did not show a stable reason to eliminate one- or three-channel models.

Retained:

```text
1, 3
```

### Dropout

Dropout had a weak global marginal effect but architecture-dependent behavior among strong large-data candidates. The best ResNet18 candidates included both 0.3 and 0.4.

Retained:

```text
0.2, 0.3, 0.4
```

### Small-grid to large-data transfer

The exact 18 configurations evaluated at both stages showed weak rank transfer. Approximate Spearman correlations included:

```text
grid rank vs candidate rank :  0.19
validation MSE              :  0.18
validation R2               :  0.16
validation MAE              : -0.53
```

This is the central reason for *reducing* the small-data grid while retaining a *wider* set of candidates for large-data confirmation.

## local_v2 design

The resulting grid is:

```text
architecture    3 values
channels        2 values
pretraining     1 value
dropout         3 values
learning rate   3 values
batch size      2 values
```

Total:

```text
108 grid configurations
```

Candidate strategy:

```text
top 8 per architecture = 24 configurations
24 x 2 large-data seeds = 48 candidate runs
```

Final ensemble:

```text
1 selected configuration x 5 seeds
```

## Parameters deliberately not expanded

`weight_decay` and `smooth_l1_beta` each had only one level in the first study, so their effects are not estimable from `local_v1`. Expanding both now would multiply the search substantially and obscure the targeted learning-rate/batch-size refinement.

They therefore remain:

```text
weight_decay   = 1e-4
smooth_l1_beta = 0.05
```

A later focused experiment can vary them after `local_v2` localizes the dominant region.

## Selection objective

The first study showed near-perfect redundancy between normalized MSE and R2 in the composite selection score. Nevertheless, `local_v2` retains the original objective so that the refinement changes the search space without simultaneously changing the ranking definition.

After `local_v2`, alternative objectives can be evaluated explicitly, for example target-balanced MAE or a criterion designed to reduce the relatively larger error observed for `beta`.
