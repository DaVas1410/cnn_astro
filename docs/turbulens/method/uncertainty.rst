.. docs/turbulens/method/uncertainty.rst

.. _uncertainty:

Uncertainty: Deep Ensembles
===========================

Intuition
---------

Train the same network several times from different random starting
points and the copies will not agree exactly. Where the training data
pin the answer down, they agree closely. Where the data leave room, they
drift apart. The spread between copies is therefore a useful signal of
how much the model *itself* is unsure. `turbulens` reports it next to
every prediction.

This spread is not an error bar in the usual sense. It does not include
the noise or ambiguity in the image itself, it has not been calibrated
against the true errors, and it can be confidently small on an image
unlike anything in the training set. Read it as "how much do
independently trained models disagree here", not as "the true value lies
within this range with a given probability".

Formalism
---------

**Deep ensembles.** A deep ensemble :cite:p:`lakshminarayanan2016` is a
set of :math:`M` networks with the same architecture and hyperparameters,
trained independently with different random seeds, which change the
weight initialization of the newly added layers, the data order, dropout
masks and augmentations. In `turbulens` all members share the same
training data and the same pixel normalization (a common normalization
seed). Their outputs are therefore directly comparable, and the spread
reflects training randomness rather than differences in preprocessing.

**Aggregation.** For one image, let :math:`\hat y_{m}` be member
:math:`m`'s output for one target, already clipped and mapped to
physical units with that member's own target range
(:eq:`eq-denormalization`). The reported value and spread are

.. math::
   :label: eq-ensemble

   \bar y = \frac{1}{M}\sum_{m=1}^{M} \hat y_{m}, \qquad
   s = \sqrt{\frac{1}{M-1}\sum_{m=1}^{M}\left(\hat y_{m} - \bar y\right)^2}.

Each member is mapped to physical units *before* averaging. If members
had different target ranges, averaging the raw :math:`[0,1]` outputs
first would give a wrong mean and spread. The sample standard deviation
(:math:`M-1` denominator) needs at least two members, which is why
`turbulens` refuses to load an ensemble with fewer. The published
ensembles have :math:`M = 5`.

**Epistemic and aleatoric uncertainty.** Predictive uncertainty is
usually split into two parts :cite:p:`kendall2017`. *Epistemic*
uncertainty comes from the model: limited training data and the
randomness of training. More data or more members can reduce it.
*Aleatoric* uncertainty comes from the data: an image may simply not
contain enough information to pin a parameter down (for example
:math:`k_\mathrm{max}` when the fine-scale power is weak), and no amount
of training removes that. A network can estimate aleatoric uncertainty
only if it is trained to output it, for example a variance alongside each
prediction with a likelihood loss.

The `turbulens` networks are trained with a point-prediction loss
(:eq:`eq-smooth-l1`) and output no variance. The ensemble spread
:math:`s` is therefore an estimate of **epistemic uncertainty only**.
That is why its column is named ``epistemic_std``.

**Calibration and dataset shift.** An uncertainty is *calibrated* if,
for example, intervals of :math:`\bar y \pm s` contain the true value in
the fraction of cases a Gaussian would predict. Modern networks are
often poorly calibrated :cite:p:`guo2017`, and ensemble spread tends to
become over-confident when the inputs differ from the training data
:cite:p:`ovadia2019`, which real observations do. The intervals in
`turbulens` are descriptive spreads, not calibrated confidence
intervals. How well they cover the truth on synthetic test data is
measured during evaluation (:ref:`evaluation-metrics`) and reported in
the accompanying paper, not assumed here.

In turbulens
------------

- `turbulens.inference.Inferencer` loads every member of an ensemble,
  applies each member's own normalization, and computes
  :eq:`eq-ensemble` with ``ddof=1``. Its output table has the columns
  ``value`` (:math:`\bar y`) and ``epistemic_std`` (:math:`s`).
- `turbulens.models.registry.EnsembleRegistry.discover_members` counts
  only members that finished training (those with a ``COMPLETED.json``).
- Large spread is a reason for caution. Small spread on a real
  observation is **not** evidence that the prediction is right; see
  :doc:`/user_guide/interpreting_results`.
