.. docs/turbulens/method/evaluation.rst

.. _evaluation-metrics:

Evaluation Metrics
==================

Intuition
---------

The training pipeline scores every model on synthetic images it has not
trained on, where the true parameters are known. This page defines the
scores it computes, so that reported numbers can be read correctly. The
values themselves belong to the accompanying paper and are not repeated
in this documentation.

Scores are computed separately for each target, in physical units, on
the validation set (used for model selection) and the test set (used
only for final reporting; :doc:`/training_pipeline/scientific_invariants`).

Formalism
---------

Let :math:`y_i` be the true value and :math:`\bar y_i` the ensemble
prediction (:eq:`eq-ensemble`) for image :math:`i = 1,\dots,n`, with
residual :math:`r_i = \bar y_i - y_i`.

**Point accuracy.**

.. math::
   :label: eq-point-metrics

   \mathrm{MAE} = \frac{1}{n}\sum_i |r_i|, \qquad
   \mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i r_i^2}, \qquad
   \mathrm{bias} = \frac{1}{n}\sum_i r_i,

.. math::
   :label: eq-r2

   R^2 = 1 - \frac{\mathrm{SS}_\mathrm{res}}{\mathrm{SS}_\mathrm{tot}}
       = 1 - \frac{\sum_i r_i^2}{\sum_i (y_i - \langle y\rangle)^2}.

:math:`R^2` is the fraction of the label variance the predictions
explain. It is 1 for perfect predictions and 0 for always predicting the
mean. Because the labels are uniform over known ranges, :math:`R^2`
values are comparable across targets in a way RMSE values (in different
units) are not. Pearson and Spearman (rank) correlations between
:math:`\bar y` and :math:`y` are also reported.

**Spread–error relation.** A useful uncertainty grows with the actual
error. The pipeline reports the Pearson and Spearman correlation between
the ensemble spread :math:`s_i` and :math:`|r_i|`, together with
percentiles of :math:`s_i`.

**Interval coverage.** For a nominal level :math:`q` (0.68, 0.90 and
0.95), the interval is :math:`\bar y_i \pm z_q s_i` with
:math:`z_q = \Phi^{-1}\!\left(\tfrac{1+q}{2}\right)`, and

.. math::
   :label: eq-coverage

   \mathrm{coverage}_q = \frac{1}{n}\sum_i
   \mathbf{1}\!\left[\,|y_i - \bar y_i| \le z_q s_i\,\right].

A calibrated spread has :math:`\mathrm{coverage}_q \approx q`;
coverage well below :math:`q` means the spread is too small
(over-confident) :cite:p:`guo2017`. The mean interval width is reported
alongside, since a spread can reach nominal coverage by being uselessly
wide.

**Bootstrap confidence intervals.** MAE, RMSE, :math:`R^2`, the Pearson
and Spearman correlations and the bias each get a 95% confidence
interval from the nonparametric bootstrap :cite:p:`efron1979`: resample
the :math:`n` test images with replacement (2000 times by default),
recompute the metric, and take the 2.5th and 97.5th percentiles. This
reflects uncertainty in the metric due to the finite test set, not
uncertainty in individual predictions.

**Multitask versus single-target comparison.** To compare the shared
multitask network with dedicated single-target networks, both are
evaluated on the *same* test images, so differences can be paired:

- a paired bootstrap of the difference in each metric between the two
  approaches, resampling images jointly for both;
- a sign-flip permutation test on the per-image difference in absolute
  error: under the null hypothesis that neither approach is better, each
  difference is equally likely to have either sign;
- the four per-target :math:`p`-values corrected for multiple testing
  with the Holm step-down procedure :cite:p:`holm1979`, which controls
  the family-wise error rate.

A statistically significant difference can still be practically
negligible on test sets this large, so effect sizes (the paired
differences and their intervals) matter more than the :math:`p`-values.

**Interpretability.** To see what the network has learned, the
pipeline extracts the 256-dimensional shared-trunk embedding (or the
backbone features) of every image and analyses it with:

- principal component analysis :cite:p:`pearson1901`, correlating the
  leading components with each target;
- canonical correlation analysis :cite:p:`hotelling1936`, finding the
  linear combinations of embedding dimensions and targets that are most
  correlated;
- optionally UMAP :cite:p:`mcinnes2018`, a nonlinear 2D map of the
  embedding for visual inspection.

All three are fitted on validation-set embeddings only (standardization,
PCA, CCA and UMAP alike); test embeddings are only transformed with the
fitted maps afterwards, so the test set never shapes the analysis.

These are descriptive tools. A target that is easy to read off linearly
from the embedding is well represented there, but correlation in the
embedding is not proof of a physical mechanism.

In turbulens
------------

- These metrics are computed by the training pipeline, not by the
  `turbulens` inference package: ``evaluate_ensemble.py``,
  ``compare_multitask_single.py`` and ``interpret_embeddings.py`` in
  ``turbulens/training/``, with shared code in ``pipeline_lib/metrics.py``.
- Outputs are written under each run's ``ensemble/evaluation/`` and
  ``ensemble/interpretability/`` directories
  (:doc:`/training_pipeline/outputs`).
