# Configuration layout

The repository ships one canonical scientific base configuration and thin task-specific overrides.

## Canonical experiment

`config_base.yaml` defines the complete `local_v2` methodology and refined search space. It is the single source of truth for shared scientific settings.

Task files inherit from the base:

```text
config_multitask.yaml
config_single_k_min.yaml
config_single_k_max.yaml
config_single_sigma.yaml
config_single_beta.yaml
```

There is intentionally no separate "refined" multitask YAML. Use CLI stage boundaries to stop and resume the same scientific configuration:

```bash
bash run_pipeline.sh configs/config_multitask.yaml --to-stage select_final
bash run_pipeline.sh configs/config_multitask.yaml --from-stage train_ensemble
```

This keeps one resolved configuration hash for one experiment.

## Refined local_v2 grid

```text
architecture    : resnet18, resnet34, resnet50
input channels  : 1, 3
pretraining     : true
base dropout    : 0.2, 0.3, 0.4
learning rate   : 5e-4, 1e-3, 2e-3
batch size      : 64, 128
weight decay    : 1e-4
loss            : smooth_l1
Smooth L1 beta  : 0.05
```

Total: 108 small-data grid configurations.

The exact torchvision pretraining recipes are pinned in `model.pretrained_weights` rather than using a version-dependent `Weights.DEFAULT` alias.

## Changing the experiment

Do not edit a scientific parameter and reuse `experiment.version: local_v2`. For a new study, copy or edit the base configuration and change the version, for example:

```yaml
experiment:
  version: local_v3
```

The pipeline refuses to reuse an existing result root when the resolved configuration hash differs, unless the explicit safety override is enabled. The override should not be used to mix scientifically different runs.

## Comparison configuration

`comparison.yaml` is used only after the multitask and four single-target final ensembles are complete. It points to the corresponding `local_v2` result roots and defines the bootstrap/permutation repetition counts for the paired final comparison.
