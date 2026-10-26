.. docs/turbulens/training_pipeline/hyperparameter_search.rst

Hyperparameter Search and Model Selection
=========================================

The published ensembles use one configuration, chosen by a two-stage
search that only ever looks at validation data. This page describes
the search space, why it has this shape, and how the winner is picked.
The scores of individual configurations are part of the run outputs and
the accompanying paper, not of this documentation.

Search space
------------

The refined search (experiment version ``local_v2``) is a full factorial
grid:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Factor
     - Levels
   * - backbone
     - ResNet-18, ResNet-34, ResNet-50
   * - input channels
     - 1, 3
   * - ImageNet pretraining
     - yes
   * - base dropout :math:`p`
     - 0.2, 0.3, 0.4
   * - learning rate
     - :math:`5\times10^{-4}`, :math:`10^{-3}`, :math:`2\times10^{-3}`
   * - batch size
     - 64, 128

That is :math:`3 \times 2 \times 1 \times 3 \times 3 \times 2 = 108`
configurations. Weight decay (:math:`10^{-4}`), the smooth-L1 threshold
(:math:`\delta = 0.05`), the trunk and head widths, the optimizer, the
schedule, early stopping, the normalization percentiles and the
augmentation are held fixed (:ref:`training`).

Why this space
--------------

The grid was narrowed from an earlier, broader search (``local_v1``)
that also varied pretraining and more batch sizes. A post-hoc analysis
of that search shaped the refinement:

- **Learning rate** was the dominant factor, and the best tested value,
  :math:`10^{-3}`, sat at the edge of the tested range. The new grid
  brackets it by a factor of two on each side.
- **Pretraining** clearly beat training from scratch, and every
  configuration that reached the large-data stage was pretrained.
  Scratch initialization was dropped.
- **Batch size** looked unimportant, or even favored small batches, when
  averaged over everything. Among pretrained runs at the best learning
  rate, the larger batches were consistently better. The new grid keeps
  the two largest feasible values. This is why the analysis conditions
  on the strongest factors before reading off weaker ones.
- **Backbone, input channels and dropout** had weak or target-dependent
  effects, so all their levels were kept.
- **Rankings transferred weakly** from the small screening dataset to
  the large dataset: configurations that ranked well on small data did
  not reliably rank well on large data. This is the central reason the
  refined search trains a *smaller* grid on small data but carries a
  *wider*, architecture-stratified set of candidates to the large-data
  stage.
- Weight decay and the smooth-L1 threshold had only one level in the
  earlier search, so nothing could be learned about them. They stay
  fixed rather than multiplying the grid; a later focused experiment
  can vary them.

Selection score
---------------

For each configuration, per-target mean squared error and :math:`R^2`
are computed on the validation set in normalized target units
(:ref:`training`) and macro-averaged over targets. Across all
configurations being compared,

.. math::
   :label: eq-selection-score

   \mathrm{score} =
   \operatorname{minmax}\!\left(\overline{\mathrm{MSE}}\right) +
   \operatorname{minmax}\!\left(1 - \overline{R^2}\right),

where :math:`\operatorname{minmax}` rescales a quantity to
:math:`[0, 1]` over the compared set. Lower is better. The two terms are
nearly redundant for these data, but the score was kept identical to
the earlier search, so the refinement changed only the search space and
not the ranking rule at the same time. The test set never enters this
score.

Two-stage selection
-------------------

1. **Screening.** All 108 configurations are trained once (seed 42) on
   the small dataset. The 8 best configurations *per backbone* are kept,
   giving 24 candidates. Stratifying by backbone guards against the weak
   small-to-large rank transfer eliminating a whole backbone family too
   early.
2. **Confirmation.** The 24 candidates are retrained on the large
   dataset with two seeds (31415 and 27182), giving 48 runs. Each
   configuration's validation metrics are averaged over its two seeds,
   and the configuration with the best seed-averaged score is selected.
3. **Ensemble.** The selected configuration is retrained with five seeds
   (42, 123, 456, 789, 2026) and one shared normalization seed
   (:ref:`uncertainty`).

.. _selected-configurations:

Each single-target experiment runs the same procedure independently for
its own target. The configurations selected for the published ensembles
(experiment version ``local_v2``, all ImageNet-pretrained) are:

.. list-table::
   :header-rows: 1
   :widths: 22 18 14 14 16 16

   * - Ensemble
     - Backbone
     - Channels
     - Dropout :math:`p`
     - Learning rate
     - Batch size
   * - ``multitask``
     - ResNet-34
     - 3
     - 0.4
     - :math:`10^{-3}`
     - 128
   * - ``single_k_min``
     - ResNet-50
     - 1
     - 0.2
     - :math:`5\times10^{-4}`
     - 64
   * - ``single_k_max``
     - ResNet-50
     - 3
     - 0.4
     - :math:`10^{-3}`
     - 64
   * - ``single_sigma``
     - ResNet-18
     - 1
     - 0.3
     - :math:`5\times10^{-4}`
     - 128
   * - ``single_beta``
     - ResNet-50
     - 1
     - 0.4
     - :math:`2\times10^{-3}`
     - 128

Each member's ``configuration.json`` records its full configuration
(:doc:`outputs`).

Post-hoc analysis
-----------------

``analyze_hyperparameter_importance.py`` reads the stored grid and
candidate outputs without changing them. It reports per-level summaries
with uncertainty, one-factor :math:`\eta^2`, random-forest permutation
importance, pairwise interaction diagnostics, a conditional analysis
that checks whether a factor's effect reverses inside the best regime,
per-target summaries, learning curves, and how well rankings transfer
from the small to the large dataset. These analyses are descriptive and
model-diagnostic; they do not establish causal effects of
hyperparameters.
