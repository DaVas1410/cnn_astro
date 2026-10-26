.. docs/turbulens/method/training.rst

.. _training:

Training
========

Intuition
---------

Training adjusts the network's weights so that its four outputs match
the known labels of the training images. The loss is averaged over many
images and all four targets at once. Training stops when performance on
a separate validation set stops improving, and the weights from the best
validation epoch are the ones kept. Each training image is also shown
in random rotations and reflections, since a turbulent field's statistics
do not depend on its orientation.

Formalism
---------

**Targets.** Each label :math:`y_j` is mapped to :math:`[0, 1]` with its
training range :math:`[a_j, b_j]`,
:math:`\tilde y_j = (y_j - a_j) / (b_j - a_j)`, the inverse of
:eq:`eq-denormalization`. All four targets therefore enter the loss on
the same scale and with equal weight.

**Loss.** The loss is the smooth L1 (Huber) loss :cite:p:`huber1964`, in
the form popularized for regression in deep networks by
:cite:t:`girshick2015`. For a residual :math:`r = \hat y - \tilde y`,

.. math::
   :label: eq-smooth-l1

   \ell_\delta(r) =
   \begin{cases}
     \dfrac{r^2}{2\delta}, & |r| < \delta,\\[4pt]
     |r| - \dfrac{\delta}{2}, & |r| \ge \delta,
   \end{cases}

with :math:`\delta = 0.05` in normalized units (5% of each target's
range). Below :math:`\delta` the loss is quadratic, like a mean squared
error; above it, it grows only linearly, so a few hard or ambiguous
images (for example a :math:`k_\mathrm{max}` the image barely
constrains) cannot dominate the gradient. The training loss is the mean
of :math:`\ell_\delta` over all images in a batch and all targets.

The loss is a pure point-prediction loss. The network does **not**
predict a variance or any other per-image uncertainty; all uncertainty
`turbulens` reports comes from the ensemble (:ref:`uncertainty`).

**Optimizer.** Weights are updated with AdamW
:cite:p:`loshchilov2017`: Adam's adaptive moment estimates
:cite:p:`kingma2014` with weight decay applied directly to the weights
instead of through the gradient. All published models use weight decay
:math:`10^{-4}` and :math:`(\beta_1, \beta_2) = (0.9, 0.999)`; learning
rate and batch size were chosen per ensemble by the search (the
multitask ensemble uses :math:`10^{-3}` and 128; see
:ref:`selected configurations <selected-configurations>`).

**Schedule and stopping.** The learning rate is halved whenever the
validation loss has not improved for 5 epochs (down to
:math:`10^{-7}`). Training stops after 20 epochs without improvement,
or at most 200 epochs. The checkpoint with the lowest validation loss is
kept, and the test set plays no part in either decision
(:doc:`/training_pipeline/scientific_invariants`).

**Stabilization.** Gradients are clipped to a global norm of 1
:cite:p:`pascanu2012`, and the forward and backward passes run in mixed
16/32-bit precision with dynamic loss scaling
:cite:p:`micikevicius2017`.

**Augmentation.** Each training image is rotated by a random multiple of
90° and randomly flipped horizontally and vertically. These eight
symmetries of the square map the pixel grid onto itself exactly, with no
interpolation, and leave all four labels unchanged, because the
generating spectrum is isotropic. Validation and test images are never
augmented.

**Determinism.** GPU training uses non-deterministic but faster
convolution algorithms (``deterministic: false``,
``cudnn_benchmark: true``). Two training runs with identical settings
therefore give numerically different, statistically equivalent weights.
This is by design and is not a bug.

In turbulens
------------

- Training lives in the separate training pipeline,
  ``turbulens/training/``; its single-model loop is
  ``train_regression.py``, used unchanged for hyperparameter search,
  candidate retraining and the final ensemble members
  (:doc:`/training_pipeline/overview`).
- Apart from learning rate and batch size, which were searched, the
  settings above are shared by all published ensembles. Every value
  is a configuration key (:doc:`/training_pipeline/configuration`), and
  each member's full resolved configuration is saved in its
  ``configuration.json``.
- The inference package never trains or fine-tunes; it only loads the
  best checkpoint of each member.
