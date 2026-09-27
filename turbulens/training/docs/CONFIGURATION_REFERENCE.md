# Configuration Reference

The canonical scientific configuration is `configs/config_base.yaml`. Task-specific YAML files inherit from it.

## `experiment`

- `project_root`: project root, resolved relative to the YAML file.
- `output_root`: result root, resolved relative to `project_root`.
- `version`: immutable experiment version used in result paths.
- `allow_config_change`: safety override; normally `false`.

## `data`

- `small_root`: screening dataset.
- `large_root`: candidate/final dataset.
- `split_directories`: authoritative split folder names.
- `require_square_for_augmentation`: require square images for 90-degree rotation augmentation.
- `enforce_disjoint_files`: reject a resolved HDF5 file path appearing in multiple splits.
- `normalization`: training-only robust percentile normalization settings.

## `model`

- `architectures`: backbone families included in the grid.
- `in_channels`: one- or three-channel model inputs.
- `pretrained`: ImageNet initialization choices included in the grid.
- `pretrained_weights`: exact torchvision ImageNet recipe pinned for each configured architecture.
- `preserve_resolution`: replaces the standard ResNet max-pool with identity when true.
- `input_standardization`: `imagenet` or `none`.
- `shared_dimensions`: widths of the shared MLP.
- `head_hidden_dimension`: target-specific head width.
- `dropout_policy`: explicit mapping from searched base dropout to shared/head dropout values.

## `search`

- `repetitions_per_configuration`: repetitions of each small-data grid point.
- `seeds`: one seed per repetition.
- `space`: factorial search dimensions.
- `selection.metric`: `mse`, `rmse`, `r2`, or `mse_r2`.
- `selection.target`: `overall` or one configured target.
- `selection.space`: normalized or physical metric space. Overall error selection uses normalized space.
- `selection.group_by`: grouping variable for candidate retention; `architecture` in `local_v2`.
- `selection.top_per_group`: actual number retained within each group.
- `selection.top_k`: size of the reporting table; it does not cap grouped selection.
- `selection.require_all`: require all expected runs before selection.

## `candidate_training`

Controls large-data confirmation of selected grid configurations.

- `repetitions_per_configuration`: independent large-data repeats per candidate.
- `seeds`: one seed per repeat.
- `selection`: final hyperparameter-selection settings.

With `group_by: null`, `top_per_group` is effectively the number of final configurations selected. `local_v2` selects one.

## `ensemble`

- `members`: final deep-ensemble size.
- `seeds`: independent final member seeds.
- `normalization_seed`: shared image-normalization sampling seed across members.
- `interval_levels`: levels used for descriptive ensemble-spread coverage summaries.
- `bootstrap_repetitions`: bootstrap repetitions for final metric confidence intervals.

## `training`

- `epochs`: maximum epochs.
- `gradient_clip_norm`: L2 gradient clipping threshold; zero disables clipping.
- `optimizer`: explicit fixed AdamW settings.
- `scheduler`: explicit ReduceLROnPlateau settings.
- `early_stopping`: validation-loss early stopping.
- `augmentation`: training-only spatial augmentation.
- `amp`: CUDA automatic mixed precision.
- `deterministic`: request deterministic PyTorch algorithms.
- `cudnn_benchmark`: cuDNN autotuning; automatically disabled if deterministic mode is true.
- `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`: DataLoader performance controls, not scientific search factors.

## `evaluation`

Defines which splits are evaluated in each stage. The shipped configuration intentionally sets test evaluation false for grid and candidate stages and true only for final models.

## `runtime`

- `gpus`: allowed physical/logical GPU indices.
- `max_parallel_jobs`: maximum concurrent training subprocesses.
- `device`: `cuda`, `cpu`, or `auto`.
- `python_executable`: `current` uses the exact interpreter that launched `pipeline.py`; an explicit executable/path can also be supplied.
- `gpu_wait`: availability guard based on `nvidia-smi`.
- `environment`: environment variables exported to child processes.

The runtime GPU guard does not alter training batch size.

## `checkpointing`

Controls resumable checkpoints and verification.

## `storage`

Different retention profiles are used for grid, candidate, and final runs. Screening runs remain compact; final ensemble members retain inference checkpoints, predictions, plots, and environment information.

## `interpretability`

Controls embedding extraction, PCA, CCA, optional UMAP, sample caps, batching, and output retention. These batch sizes are analysis/inference batching controls and are unrelated to the training batch-size hyperparameter.

## `stages`

The canonical configuration includes the complete workflow. Use CLI stage boundaries such as:

```bash
--to-stage select_final
--from-stage train_ensemble
```

rather than creating scientifically identical YAML files with different stage lists. This preserves one configuration hash for one scientific experiment.
