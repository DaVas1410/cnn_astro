.. docs/turbulens/user_guide/interpreting_results.rst

Interpreting Results
====================

`turbulens` always returns four numbers. Whether they mean anything for
a given map depends on how closely that map resembles the synthetic
fields the models learned from. This page lists what to check, roughly
in order of importance.

Is the map like the training data?
----------------------------------

The models have only ever seen 2D log-normal fields with a sharply
band-limited power-law spectrum (:ref:`lognormal-fields`). A real map
differs in several ways at once:

- **Spectral shape.** Real spectra are continuous, without sharp band
  edges, and are shaped at both ends by the observation: the field of
  view at low :math:`k`, the beam and the noise at high :math:`k`
  (:ref:`projection-and-slices`).
- **One-point statistics.** Projection along the line of sight narrows
  and changes the density PDF, and emission is not density where gas is
  optically thick.
- **Velocity structure.** Narrow channels mix velocity and density
  structure.
- **Image size and resolution.** The models were trained on
  :math:`128 \times 128` images and inputs are not resized.

Each of these can shift the predictions. None of them is detected
automatically, except a mismatch of the pixel distribution.

Warning signs
-------------

**The clip-fraction warning.** If more than 5% of pixels fall outside
the training pixel range, `turbulens` warns, and the table's
``clipped_pixel_fraction_max`` records the fraction (:ref:`distribution-shift
warning <ood-warning>`). Treat any prediction made under this warning as
unreliable. For real data, check first that ``--real-data`` was used.

**Values at the edge of a training range.** Outputs are clipped to the
training ranges (:ref:`output mapping <denormalization>`):
:math:`k_\mathrm{min} \in [1, 32]`, :math:`k_\mathrm{max} \in [34, 64]`,
:math:`\sigma \in [0, 5]`, :math:`\beta \in [-3, -1]`. A prediction at or
very near a bound may be a clipped value whose true counterpart lies
outside the range, especially if all members agree on it.

**Large spread.** A large ``epistemic_std`` means the members disagree
and the input is ambiguous or unfamiliar to them.

**Small spread is not confirmation.** Ensemble spread tends to be too
small on inputs unlike the training data (:ref:`uncertainty`). Members
can agree closely on a wrong answer.

**Disagreement with the classical estimate.** Compare ``k_min`` and
``k_max`` with ``--method fft-powerlaw`` (:ref:`spectral-estimators`).
The two measure related but not identical things, but a large
discrepancy deserves a look at the power spectrum.

What each parameter means on real data
--------------------------------------

:math:`k_\mathrm{min}`, :math:`k_\mathrm{max}`
   Band edges in cycles per *image side*. They convert to physical
   scales with the map's size: a wavenumber :math:`k` corresponds to a
   length :math:`L/k` on a map of side :math:`L`. On a real map they are
   likely to reflect the field of view (:math:`k_\mathrm{min}` near its
   lower bound) and the beam or noise (:math:`k_\mathrm{max}`) as well as
   any physical injection or dissipation scale. On an image of a size
   other than 128 pixels, structure is seen in pixel units, so the
   predicted :math:`k` values do not simply rescale with the image size.
   Prefer :math:`128 \times 128` inputs when the band edges matter.

:math:`k_\mathrm{max}` specifically
   In steep spectra the power near :math:`k_\mathrm{max}` is weak, so the
   upper edge is physically harder to detect than the lower one, and on
   real maps beam smoothing and noise act at exactly those scales. Treat
   :math:`k_\mathrm{max}` with particular caution; how well it is
   recovered on synthetic data is reported in the accompanying paper.

:math:`\beta`
   Defined as the 3D energy-spectrum index of the *linear* density
   (:ref:`notation`). For an integrated map, the projection-slice relation
   gives the linear map's power spectrum the same index as the 3D
   per-mode spectrum, so the prediction can be read as the 3D
   :math:`\beta` of the projected gas (:eq:`eq-projection`), under the
   assumption that the map's linear intensity behaves like the linear
   training density. For a thin slice or a narrow channel it cannot. A
   useful cross-check is ``--method fft-powerlaw``, whose ``alpha`` on the
   linear map corresponds to :math:`\beta - 2` under the same reading.

:math:`\sigma`
   The standard deviation of the linear density of a log-normal field
   with unit mean (:ref:`the four parameters <pyfc-parameters>`). On a real map it measures the
   contrast of the *log-intensity* map against what the model learned,
   because the real-data normalization keeps contrast and discards only
   the overall level. Turning it into a Mach number needs the
   assumptions of :eq:`eq-sigma-mach` plus a correction for projection,
   which `turbulens` does not make.

Good practice
-------------

- Use integrated or wide velocity ranges; avoid single narrow channels.
- Always use ``--real-data`` for real observations, and never for
  synthetic :math:`\log_{10}\rho` images.
- Check ``clipped_pixel_fraction_max`` and look for values pinned at a
  range bound.
- Cross-check the band edges with ``--method fft-powerlaw``.
- Report the ``model_revision`` with any published number
  (:doc:`outputs_provenance`).
- Treat predictions on real data as model-based estimates under the
  assumptions above, not as measurements. See :doc:`/limitations`.
