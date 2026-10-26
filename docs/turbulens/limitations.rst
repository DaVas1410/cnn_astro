.. docs/turbulens/limitations.rst

Limitations
===========

`turbulens` makes model-based estimates under specific assumptions. The
main limitations are collected here; each links to the page that
explains it.

Training data
-------------

- **Synthetic only.** Every model was trained and tested on synthetic
  2D log-normal fractal fields (:ref:`synthetic-data`). No model has been
  trained or validated on real observations with known turbulence
  parameters, because such ground truth does not exist.
- **Idealized fields.** The training fields are isotropic, exactly
  log-normal by construction, and have a strictly band-limited power-law
  spectrum with sharp edges (:ref:`lognormal-fields`). Real interstellar
  gas is anisotropic, intermittent, magnetized and shaped by shocks,
  gravity and feedback.
- **2D fields, not projections.** The training images are 2D fields
  generated directly. They match the *spectral shape* of a projected 3D
  field but not its one-point statistics, since projection changes the
  density PDF (:ref:`projection-and-slices`).
- **Bounded parameter space.** Predictions are clipped to the training
  ranges (:math:`k_\mathrm{min} \in [1, 32]`,
  :math:`k_\mathrm{max} \in [34, 64]`, :math:`\sigma \in [0, 5]`,
  :math:`\beta \in [-3, -1]`) and cannot report values outside them
  (:ref:`output mapping <denormalization>`). No training field has a
  narrow band with both edges low or both high.

Real observations
-----------------

- **Instrumental shaping.** The beam, noise and field of view of an
  observation set its apparent spectral edges, so predicted
  :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}` on real maps mix
  physical and observational scales
  (:doc:`user_guide/interpreting_results`).
- **Velocity structure and optical depth.** Channel maps are shaped by
  velocity fluctuations; optically thick or self-absorbed gas breaks the
  link between intensity and column density (:ref:`projection-and-slices`).
- **Normalization assumption.** Real maps are log-transformed and
  shifted onto the training scale but not rescaled. The predicted
  :math:`\sigma` therefore assumes the log-intensity contrast of the map
  is comparable to the log-density contrast of the training fields
  (:ref:`real-data normalization <real-data-normalization>`).
- **Image size.** Inputs are not resized; the models were trained on
  :math:`128 \times 128` images, and their notion of scale is tied to
  that size.

Uncertainty
-----------

- **Epistemic only.** The reported ``epistemic_std`` is the spread of
  five independently trained networks. There is no aleatoric term: the
  networks do not estimate how much information an individual image
  contains (:ref:`uncertainty`).
- **Not calibrated.** The spread is not a calibrated confidence
  interval. Ensemble spreads tend to be too small, especially on inputs
  unlike the training data, which is exactly the situation of real
  observations.
- **No detection of every failure.** Only a mismatched pixel
  distribution triggers a warning (:ref:`distribution-shift warning <ood-warning>`).
  Other differences from the training data pass silently.

Parameters
----------

- **The upper band edge is physically hard to see.** The power near
  :math:`k_\mathrm{max}` is weak in steep spectra, and on real maps beam
  and noise act at those scales. How well it is recovered is reported in
  the paper.
- **No physical conversion.** `turbulens` does not convert
  :math:`\sigma` into a Mach number or :math:`k` into a physical length;
  both need assumptions it cannot check (:eq:`eq-sigma-mach`).

Reproducibility
---------------

- GPU training is deliberately non-deterministic, so retraining
  reproduces the models statistically, not bit for bit
  (:doc:`training_pipeline/scientific_invariants`).
- The small screening dataset used for the grid search is not published
  (:doc:`training_pipeline/overview`).
