.. docs/turbulens/training_pipeline/scientific_invariants.rst

Scientific Invariants and Reproducibility
=========================================

These rules hold for every pipeline run. Several are enforced by the
code; the rest are conventions that changing a configuration must not
break.

Leakage guards
--------------

- **Normalization from training data only.** The pixel-normalization
  percentiles (:ref:`normalization`) are estimated from a seeded sample
  of training images. Validation and test images never contribute.
- **Selection on validation data only.** Early stopping, the
  learning-rate schedule, grid ranking, candidate ranking and final
  selection all use validation metrics (:doc:`hyperparameter_search`).
  Grid and candidate runs are configured not to evaluate the test split
  at all; only final ensemble members do, and only after selection.
- **Interpretability fitted on validation data.** Standardization, PCA,
  CCA and UMAP are fitted on validation embeddings; test embeddings are
  only transformed afterwards.
- **Disjoint files.** A resolved HDF5 path may appear in only one
  split. This catches the same file or symlink used twice, but not
  independently copied duplicates. Disjointness of the *content* comes
  from generating each split with its own master seed, so no parameter
  draw is shared; a few per-image noise seeds coincide by chance
  (:ref:`synthetic-data`).

Target handling
---------------

- All targets are scaled to :math:`[0, 1]` with fixed physical ranges,
  never with statistics estimated from validation or test data.
- The loss is plain smooth L1 on these scaled targets with equal weight
  per target. No target-specific loss weights are used.
- Outputs are unconstrained during training. Reported predictions clip
  each member to the target range *before* averaging,
  :math:`\mathrm{mean}(\mathrm{clip}(\cdot))`, because clipping and
  averaging do not commute. The unclipped mean is kept only as a
  diagnostic.

Configuration integrity
-----------------------

- Every scientific setting is in the YAML configuration. The code
  validates it but has no hidden fallback values for the search space.
  Settings that used to be library defaults (AdamW moments, scheduler
  thresholds, the dropout schedule, the exact pretrained-weight recipe
  per backbone) are written out explicitly.
- Batch size is used exactly as configured. There is no automatic
  probing, fallback or clamping based on GPU memory.
- An experiment version is bound to the hash of its resolved scientific
  configuration. Changing a scientific setting requires a new version
  name (:doc:`configuration`).

Randomness and determinism
--------------------------

- Seeds are explicit for grid screening, candidate retraining, every
  ensemble member, augmentation (seed, epoch and sample index), and the
  normalization sample.
- GPU training uses fast, non-deterministic convolution algorithms by
  default (``deterministic: false``, ``cudnn_benchmark: true``). Runs with
  identical seeds are methodologically reproducible but not
  bit-for-bit identical across hardware or library versions, so a
  numerically different rerun is not by itself a bug. Switching
  deterministic mode on is a new experimental configuration, since it
  can change which kernels are used.

Uncertainty reporting
---------------------

- The across-member standard deviation is descriptive epistemic spread,
  not a calibrated confidence or prediction interval (:ref:`uncertainty`).
- Bootstrap confidence intervals on evaluation metrics measure a
  different thing, the finite size of the test set, and are reported
  separately (:ref:`evaluation-metrics`).

Release checklist
-----------------

Before treating a run as final:

**Before training.** Use a clean output location; run
``check_environment.py`` and confirm the expected counts (108 grid runs,
24 candidates, 48 candidate runs, 5 members); confirm every HDF5 file
validates; confirm the resolved run directory and GPU list; review the
``--dry-run --print-plan`` output; archive the exact YAML files.

**During training.** Do not edit the YAML of an active version; stop
gracefully (``stop_pipeline.sh`` or ``SIGTERM``) rather than with
``SIGKILL``; keep logs until checks are done; check for failed runs
before accepting a selection stage; confirm that grid and candidate
stages never evaluated the test split.

**After the search.** Run the post-hoc hyperparameter analysis and
review marginal, interaction and conditional effects together, along with
small-to-large rank transfer and candidate learning curves. Record the
selected configuration before any final test evaluation.

**Final ensemble.** Confirm five completed members with the configured
seeds, the same selected configuration and the same normalization
statistics.

**Comparison.** Confirm identical test sample keys and ground truth
across the multitask and single-target ensembles, and report paired
differences, sign-flip :math:`p`-values and their Holm-adjusted values.

**Archive.** Keep the resolved configuration and its hash, the
environment and source provenance, data validation manifests, stage
markers, selection outputs, the final configuration, evaluation and
report directories, and the hyperparameter analysis, together with the
exact code version, the dataset version, and the software environment.
