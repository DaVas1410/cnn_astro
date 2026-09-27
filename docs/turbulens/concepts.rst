.. docs/turbulens/concepts.rst

Physical Background
====================

What ``k_min``, ``k_max``, ``sigma``, and ``beta`` mean
---------------------------------------------------------

`turbulens` predicts the parameters of a log-normal fractal density
field, the model implemented by the bundled `pyFC
<https://bitbucket.org/pandante/pyfc>`_ library
(``archive/src/pyFC_lib/pyFC/clouds.py``, :class:`~pyFC.FractalCube`). A field
generated this way has a power spectrum that follows a power law,
:math:`D(k) \propto k^{\beta}`, between two wavenumber cutoffs:

- ``k_min``: the lowest wavenumber (largest spatial scale) included in
  the power spectrum.
- ``k_max``: the highest wavenumber (smallest spatial scale) included.
- ``beta``: the power-law slope of the spectrum between ``k_min`` and
  ``k_max``.
- ``sigma``: the standard deviation of the field's single-point
  (log-normal) intensity distribution, independent of the spectral
  shape.

Physically, ``k_min`` and ``k_max`` bound the range of turbulent
eddy sizes present in the simulated cloud, and ``beta`` is the
turbulent energy cascade's spectral index (a Kolmogorov-like cascade
has :math:`\beta \approx -5/3` in 3D, `pyFC`'s own default). `sigma`
sets the overall contrast (how far density fluctuates from the mean),
independent of which spatial scales those fluctuations occur at.
Recovering these parameters from an observed or synthetic 2D image is
what every model `turbulens` loads was trained to do.

``beta = -5/3`` is `pyFC`'s own constructor default
(``archive/src/pyFC_lib/pyFC/clouds.py``, ``FractalCube.__init__``), so the
Kolmogorov-cascade reading above is not just a general turbulence-theory
convention applied from outside -- it is the specific value this
generator falls back to when no ``beta`` is given.

Why ``k_max`` is the hardest target
-------------------------------------

Every phase of this repository's model development
(``archive/docs/RESEARCH_JOURNEY.md``, phases 3-7) found `k_max` recovery
capped around :math:`R^2 \approx 0.69` to :math:`0.81`, regardless of
backbone (custom CNN, ResNet-50, or a frozen DINOv2 ViT), loss function,
or uncertainty method. The root cause, found in phase 8
(``archive/docs/bugs/pyfc-kmax-not-enforced.md``), was not the model at all:
`pyFC`'s upper spectral cutoff was not actually enforced in the
generator used to produce the labeled training data, so every dataset
up to that point had a `k_max` label that corresponded only weakly to
an actual image feature. Once the generator was fixed to band-limit the
spectrum to ``[k_min, k_max]``, classical (non-CNN) `k_max` recovery
jumped from unrecoverable to :math:`R^2 = 0.978` on a validation pilot.
`turbulens` itself ships no checkpoints; it loads whatever ensemble
you point ``--model`` at (see :ref:`getting-an-ensemble`). Per
``archive/docs/RESEARCH_JOURNEY.md`` phases 8-12, the training pipeline
regenerated its dataset with the band-limiting fix in place and the
post-fix data and retrain became the production pipeline, so an
ensemble produced by the current pipeline is trained on correctly
labeled `k_max`. The bug is documented here because it explains why
`k_max` epistemic uncertainty may still run somewhat higher than
`k_min` or `sigma` in practice: even a correctly labeled `k_max` is a
spectral-tail property, inherently noisier to estimate from a single
finite image than a large-scale property like `k_min`. If you are
given an ensemble of unknown provenance, check when it was trained
before trusting its `k_max` predictions.

Classical estimation without a model: ``--method fft-band`` / ``fft-powerlaw``
---------------------------------------------------------------------------------

`turbulens infer --method` also supports two classical (non-CNN)
estimators of ``k_min``/``k_max`` (:mod:`turbulens.spectral`) that need
no trained ensemble at all -- useful when no ``--model`` is available,
or as a sanity check on CNN predictions. They measure the same image
property in two different, non-interchangeable ways, matched to two
different kinds of image:

- ``fft-band`` (`turbulens.spectral.estimate_band_edges`): reads
  ``k_min``/``k_max`` off a sharp spectral band edge. This is exactly
  right for synthetic `pyFC` images, whose power is *exactly zero*
  outside ``[k_min, k_max]`` by construction (:math:`R^2 = 0.978` on
  the validation pilot above). It is **not** appropriate for real data,
  and warns when its answer sits at the edge of the image's own
  wavenumber grid rather than at a real spectral feature -- the
  expected outcome when it is run on data with no sharp cutoff at all.
- ``fft-powerlaw`` (`turbulens.spectral.fit_inertial_range`): fits a
  power law :math:`P(k) \propto k^{\alpha}` over the mid-range of the
  spectrum and reads ``k_min``/``k_max`` off as the injection and
  dissipation/noise scales where the spectrum departs from that fitted
  line. Real turbulent clouds have a continuous power-law inertial
  range rather than a sharp cutoff, so band-edge detection is
  physically inappropriate for them; this method was developed for and
  validated against real GASS HI survey data in
  ``archive/observational_images/io-fits.ipynb``, which found
  :math:`\alpha \approx -3.7` for one such cloud, close to the
  Kolmogorov 3D value of :math:`-11/3 \approx -3.67`. It also reports
  :math:`R^2` for the fitted line (as both a table row and, via
  `turbulens.io.breakdown.CubeBreakdown.spectral_estimate_power_law`,
  an `inspect` field): a low or negative :math:`R^2` means the image
  has no real power-law structure at all (e.g. white noise, or a
  sharply band-limited synthetic image, for which ``fft-band`` is the
  appropriate estimator instead), so `alpha`/`k_min`/`k_max` should not
  be trusted even though they are still reported as plain numbers.

``turbulens inspect`` reports both estimates (labeled accordingly) for
any input that is already a single 2D image; a real telescope
``(x, y, velocity)`` cube must first be integrated down to one image
(``infer``'s ``--integrate``/``--velocity``, matching the notebook's
own "full integration" step) before either estimate is meaningful.

The band-limiting fix itself is still present in the bundled generator
today (``archive/src/pyFC_lib/pyFC/clouds.py``'s active ``func_target_spec``
explicitly enforces both cutoffs, citing
``archive/docs/bugs/pyfc-kmax-not-enforced.md`` in a code comment), and every
production ensemble under ``outputs/`` was trained on data generated by
this same bundled copy, so a checkpoint produced by the current
``turbulens/training`` pipeline is trained on correctly labeled `k_max`
by construction. If you are given an ensemble of unknown provenance
(e.g. trained against a different, older copy of `pyFC`), verify the fix
is present before trusting its `k_max` predictions.

Why an ensemble, not a single model
--------------------------------------

`turbulens.inference.Inferencer` requires at least two ensemble
members and reports both the mean prediction and the
member-to-member standard deviation (`epistemic_std`) for every
target. This follows the same deep-ensemble approach as
``archive/docs/RESEARCH_JOURNEY.md`` phase 7: training several independently
initialized models on the same data and treating their disagreement
as an estimate of epistemic uncertainty (uncertainty from limited
model/data knowledge, as opposed to aleatoric uncertainty inherent to
the measurement itself). A single point-estimate model has no way to
signal "I am not confident about this prediction"; an ensemble does,
via the spread across its members.

Multitask vs. single-target: five ensembles, not one
---------------------------------------------------------

The training pipeline produces five deep ensembles under one
``output_root``: one **multitask** ensemble (a single shared backbone
predicting all four targets) and four **single-target** ensembles
(``k_min``, ``k_max``, ``sigma``, ``beta``), each its own independently
trained network. ``turbulens infer --model auto`` resolves the
multitask ensemble by default, or one single-target ensemble with
``--target {k_min,k_max,sigma,beta}`` (see :ref:`getting-an-ensemble`).

The ``local_v2`` multitask-vs-single-target comparison
(``outputs/comparison/local_v2/comparison_report.md``) found single-target
models a statistically significant but practically small improvement
over multitask on ``k_min``, ``k_max``, and ``sigma``; multitask
significantly better on ``beta``; and the single-target system costing
roughly 5x the total training time and 4x the trainable parameters of
one shared multitask model, since it trains one full network per target
instead of one. This is why both are exposed as first-class,
individually selectable options here rather than one being deprecated
in favor of the other -- see `turbulens.models.architecture`'s module
docstring for the same finding stated as an architectural rationale.

Why the input preprocessing pipeline is shaped the way it is
----------------------------------------------------------------

Before an image reaches a model, `turbulens.models.checkpoint.preprocess_image`
applies, in order: non-finite pixel replacement, percentile-based
affine scaling to ``[0, 1]`` (clipped), channel replication, and
optionally ImageNet mean/std standardization. Each step exists for a
concrete reason:

- **Percentile-based scaling** (rather than min/max scaling) makes the
  normalization robust to a small number of extreme outlier pixels,
  which are common in real telescope data (cosmic ray hits, edge
  artifacts) but would otherwise dominate a min/max range.
- **Clipping to [0, 1]** bounds the model's input to the range it was
  trained on; `Inferencer.predict` surfaces the fraction of pixels that
  hit this boundary as a distribution-shift warning, since a real
  observation whose intensity range differs substantially from the
  synthetic training data will clip heavily.
- **ImageNet standardization** is applied because the default backbone
  is initialized from ImageNet-pretrained weights (`build_resnet_backbone`);
  matching the input statistics the backbone was originally trained on
  is standard transfer-learning practice.

References
----------

- `pyFC <https://bitbucket.org/pandante/pyfc>`_: the log-normal
  fractal cube generator that defines what `k_min`, `k_max`, `sigma`,
  and `beta` mean.
- Repository research narrative: ``archive/docs/RESEARCH_JOURNEY.md``.
- The `k_max` labeling bug: ``archive/docs/bugs/pyfc-kmax-not-enforced.md``.
- The multitask-vs-single-target comparison:
  ``outputs/comparison/local_v2/comparison_report.md``.
