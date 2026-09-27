# Configurability Audit

This audit distinguishes true search/fixed scientific settings from runtime-only controls and implementation constraints.

## Training batch size

`search.space.batch_size` is a genuine hyperparameter grid dimension. The configured value is:

1. enumerated by `pipeline.py`;
2. stored in `optimization.batch_size`;
3. passed directly to `DataLoader(batch_size=...)`;
4. copied unchanged into candidate retraining for selected configurations;
5. copied unchanged into final ensemble training for the winning configuration.

There is no automatic batch-size probing or replacement logic.

## Search parameters

The canonical refined grid varies:

```text
architecture
input channels
dropout
learning rate
batch size
```

Pretraining is fixed to `true` based on the completed first analysis. Weight decay, loss family, and Smooth L1 beta are fixed methodology parameters in this refinement.

## No hidden scientific defaults

`pipeline_lib/config.py` uses an empty scientific default mapping. Required scientific sections must be supplied by YAML.

The validator rejects missing/invalid configuration rather than silently restoring values from an older experiment.


## YAML inheritance and narrowed search spaces

Task or experimental child YAML files may narrow `model.architectures`. Because mapping inheritance is deep-merged, the resolver prunes inherited `model.pretrained_weights` entries for inactive architectures after validating that every active architecture has an explicit supported weight recipe. This prevents stale inherited entries from affecting the resolved configuration hash while keeping narrowed experiments easy to express.

## Explicit model dropout policy

The model previously derived layer-specific dropout from the searched base dropout internally. The same mapping is now explicit in `model.dropout_policy`, making the architecture auditable from the resolved configuration.

## Explicit optimizer/scheduler values

AdamW and ReduceLROnPlateau settings that previously relied partly on PyTorch defaults are now explicit in YAML and passed explicitly to PyTorch.

## Machine-specific paths

Executable Python, shell, and YAML files do not contain project-specific `/home/...` or `/mnt/...` paths. Project/data/output paths resolve from relative configuration or user-provided environment variables.

## Structural restrictions retained intentionally

The following are implementation capabilities rather than hidden hyperparameter choices:

- supported targets are `k_min`, `k_max`, `sigma`, `beta`;
- supported backbones are ResNet18/34/50;
- model input channels are 1 or 3;
- the current production loss implementation is `smooth_l1`; adding another loss family requires explicit implementation and validation;
- final evaluation requires validation and test predictions;
- interpretability supports the implemented embedding/PCA/CCA/UMAP pathways.

Extending these capabilities requires code implementation and validation, not merely adding arbitrary YAML values.
