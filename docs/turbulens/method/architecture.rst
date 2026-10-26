.. docs/turbulens/method/architecture.rst

.. _architecture:

Network Architecture
====================

Intuition
---------

A convolutional neural network (CNN) learns its own image features:
small filters respond to local patterns, and stacking many layers lets
later filters respond to larger and more abstract structure. Here the
network must turn a 2D field into four numbers that describe its power
spectrum and contrast. All four depend on the same underlying
structure, so `turbulens` uses one network that looks at the image once
and then branches into four small output heads. This is *multi-task*
learning: the shared part is pushed to learn features useful for every
target at once.

.. figure:: /_static/architecture.svg
   :alt: Diagram: normalized image into a ResNet backbone, then a shared
         two-layer trunk, then four small heads predicting k_min,
         k_max, sigma and beta.
   :width: 100%

   The multitask network. Single-target models use the same network
   class with one head, with backbone and hyperparameters chosen by
   their own search.

Formalism
---------

**Residual backbone.** The backbone is a residual network (ResNet)
:cite:p:`he2016`. Each residual block learns a correction
:math:`\mathcal{F}` to its input instead of a whole new representation,

.. math::
   :label: eq-residual

   \mathbf{h}_{l+1} = \mathbf{h}_l + \mathcal{F}(\mathbf{h}_l;\ \theta_l),

which keeps gradients flowing through deep stacks. Within each block,
convolutions are followed by batch normalization :cite:p:`ioffe2015`.
After the last stage, global average pooling turns the feature maps into
one feature vector, whatever the input size.

Two changes adapt the standard ImageNet ResNet to this problem:

- The early max-pooling layer is replaced by the identity, so feature
  maps keep twice the spatial resolution through the network. Fine
  structure near :math:`k_\mathrm{max}` would otherwise be averaged away
  early.
- The backbone starts from ImageNet-pretrained weights
  :cite:p:`deng2009` and is then trained on the fractal images
  (fine-tuning). Models with three input channels therefore receive the
  single-channel image replicated three times and standardized with
  ImageNet statistics (:ref:`normalization`). Models configured for one
  input channel instead initialize the first convolution with the channel
  mean of the pretrained RGB filters.

**Shared trunk.** The pooled feature vector passes through two fully
connected layers of width 512 and 256. Each is followed by layer
normalization :cite:p:`ba2016`, the GELU activation
:cite:p:`hendrycks2016` and dropout :cite:p:`srivastava2014`. The
256-dimensional output is the representation every head reads.

**Heads.** Each target has its own small head: a 96-unit hidden layer
with GELU and dropout, then a single linear output :math:`\hat y`. The
output is unconstrained during training and clipped to :math:`[0, 1]`
only at inference, before it is mapped to physical units
(:ref:`output mapping <denormalization>`).

**Hard parameter sharing.** Sharing the backbone and trunk across
targets is *hard parameter sharing* :cite:p:`caruana1997,ruder2017`. It
acts as a regularizer, since features must serve all targets. It costs
far less than separate networks, and it lets correlated targets inform
each other, for example a steep spectrum (:math:`\beta`) changing how
visible the fine-scale edge (:math:`k_\mathrm{max}`) is. Whether a
dedicated network per target does better is an empirical question; the
training pipeline trains both kinds so they can be compared
(:doc:`/training_pipeline/overview`).

**Dropout schedule.** One base rate :math:`p` (a hyperparameter) sets
all three dropout rates: :math:`p` after the first trunk layer,
:math:`\min(0.6,\ p + 0.1)` after the second, and
:math:`\min(0.35,\ p/2)` in each head. Deeper, wider shared layers get
more regularization than the small target-specific heads.

**Published configurations.** The published multitask ensemble uses a
ResNet-34 backbone (512-dimensional pooled features) with three input
channels and :math:`p = 0.4`, chosen by the validation-only
hyperparameter search. Each single-target ensemble was chosen by its own
search and uses a different backbone, channel count or dropout; the
selected configuration of every published ensemble is listed in
:ref:`selected configurations <selected-configurations>`. ResNet-18, ResNet-34 and ResNet-50
backbones are supported.

In turbulens
------------

- `turbulens.models.architecture.ResNetPhysicalRegressor` is the network;
  `turbulens.models.architecture.build_model_from_config` rebuilds it
  from a member's ``configuration.json``.
- The trunk widths are fixed at 512 and 256, so that every checkpoint can
  be reloaded with the same code.
- `turbulens.models.checkpoint.load_member` restores a trained member:
  the configuration, the weights from ``checkpoints/best_checkpoint.pt``,
  and the normalization statistics stored with them.
