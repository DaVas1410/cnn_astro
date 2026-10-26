.. docs/turbulens/method/preprocessing.rst

.. _normalization:

Preprocessing and Normalization
===============================

Intuition
---------

A neural network expects its inputs on a fixed, familiar scale, and
produces outputs on one too. `turbulens` therefore puts every image
through the same transformation the network saw during training, and
maps the network's outputs back to physical units afterwards. The
transformation has to be reproduced *exactly*: a mismatch does not
raise an error, it quietly produces wrong numbers.

Real observations add a complication. A GASS brightness-temperature map
is in kelvin, not in units of mean density, and it is not
log-transformed. Its overall level is arbitrary as far as the
turbulence is concerned, but its *spread* in log space is exactly what
carries the information about :math:`\sigma`. The real-data path
therefore corrects the level and deliberately leaves the spread alone.

Formalism
---------

**Training-set normalization.** Let :math:`x` be a pixel of a
:math:`\log_{10}\rho` training image. The normalization is a single
affine map, the same for every image:

.. math::
   :label: eq-pixel-normalization

   \tilde{x} = \operatorname{clip}\!\left(\frac{x - q_\mathrm{lo}}{q_\mathrm{hi} - q_\mathrm{lo}},\ 0,\ 1\right),

where :math:`q_\mathrm{lo}` and :math:`q_\mathrm{hi}` are the 0.5th and
99.5th percentiles of the pixel values of the **training set only**:
pixels from up to 10,000 randomly chosen training images, subsampled to
at most :math:`10^7` pixels with a fixed seed. Validation and test images
never contribute, so no information about them leaks into the model
(:doc:`/training_pipeline/scientific_invariants`). Non-finite pixels are
set to :math:`q_\mathrm{lo}` before scaling.

Because the map is global rather than per image, two training images
with different contrast :math:`\sigma` keep different spreads of
:math:`\tilde x`. That spread is the signal the :math:`\sigma` head
learns from.

**Network input.** For models with three input channels, the normalized
single-channel image is replicated to three identical channels and
standardized channel-wise with the ImageNet mean and standard deviation
that pretrained ResNet weights expect. Models with one input channel use
the average of the three ImageNet means and of the three standard
deviations instead (:ref:`architecture`).

.. _ood-warning:

**Clipping as a distribution-shift signal.** For a training-like image,
on average about 1% of pixels fall outside
:math:`[q_\mathrm{lo}, q_\mathrm{hi}]`, by construction of the
percentiles. The fraction grows with contrast, and some high-:math:`\sigma`
synthetic images already exceed the warning threshold. `turbulens.inference.Inferencer` records the fraction
of pixels that hit either clip boundary and warns when it exceeds 5% for
any member. A high fraction means the image's pixel distribution is
unlike the training data, and the predictions should not be trusted.

.. _real-data-normalization:

**Real observations: recenter, don't rescale.** For a real map with
positive linear intensities :math:`I`, `turbulens` computes

.. math::
   :label: eq-real-normalization

   x = \log_{10} I, \qquad
   x' = x + \left(c_\mathrm{train} - c_\mathrm{image}\right),
   \qquad
   c = \tfrac{1}{2}\left(q_{0.5} + q_{99.5}\right),

where :math:`c_\mathrm{image}` is the midpoint of the image's own 0.5th
and 99.5th percentiles and :math:`c_\mathrm{train}` the midpoint of the
training values :math:`q_\mathrm{lo}, q_\mathrm{hi}`. Then :math:`x'` goes
through :eq:`eq-pixel-normalization` with the *training*
:math:`q_\mathrm{lo}, q_\mathrm{hi}`. Non-positive and NaN pixels (noise
below zero, blanked borders) are first raised to the image's smallest
positive value.

The shift removes the arbitrary overall level. Under the logarithm this
includes any calibration factor, distance or channel width, since a
multiplicative constant in :math:`I` is an additive constant in
:math:`x`. The width :math:`q_\mathrm{hi} - q_\mathrm{lo}` is *not*
re-derived from the image. A per-image rescaling would stretch every map
to the same apparent contrast and erase the amplitude information the
:math:`\sigma` prediction depends on.

.. _denormalization:

**Outputs.** Each network head produces a value :math:`\hat y`, which is
mapped to physical units with the target's training range
:math:`[a, b]`:

.. math::
   :label: eq-denormalization

   y = a + (b - a)\,\operatorname{clip}(\hat y, 0, 1).

The clip means a prediction can never leave the training range: an
image whose true :math:`\beta` is :math:`-3.5` can at most be reported
as :math:`-3`. This mapping is applied to each ensemble member separately,
before members are combined (:ref:`uncertainty`).

In turbulens
------------

- `turbulens.models.checkpoint.preprocess_image` implements
  :eq:`eq-pixel-normalization` and the channel handling, and returns the
  clipped-pixel fraction.
- `turbulens.models.checkpoint.real_observation_normalization`
  implements :eq:`eq-real-normalization`. The CLI uses it with
  ``--real-data``, and `turbulens.inference.Inferencer.predict` with
  ``real_data=True``. It is wrong to omit it for real maps, and wrong to
  use it for synthetic :math:`\log_{10}\rho` images, which are already on
  the training scale.
- `turbulens.models.checkpoint.denormalize_targets` implements
  :eq:`eq-denormalization`.
- Each member stores its own :math:`q_\mathrm{lo}, q_\mathrm{hi}` and
  target ranges in its checkpoint and configuration; `turbulens` reads
  them from there rather than assuming values.
- Images are not resized. A model trained on :math:`128 \times 128`
  images can be applied to other sizes, but its :math:`k` predictions
  then refer to scales measured in training-image pixels; see
  :doc:`/user_guide/interpreting_results`.
