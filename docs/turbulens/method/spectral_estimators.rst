.. docs/turbulens/method/spectral_estimators.rst

.. _spectral-estimators:

Classical Spectral Estimators
=============================

Intuition
---------

The band edges :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}` can also
be read directly off an image's power spectrum, without any trained
model: find where the power starts and where it stops. `turbulens`
provides two such classical estimators. They are fast, transparent and
need no checkpoint, which makes them useful as a sanity check on the
neural-network predictions and as a baseline. They estimate only the
band edges (and, for real data, a slope), not :math:`\sigma`.

The two estimators target different kinds of spectra. Synthetic pyFC
images have power *only* inside the band, so their edges are sharp
cliffs. Real maps have a continuous power law that bends or flattens at
its ends, so their "edges" are where the spectrum departs from a power
law.

Formalism
---------

Both start from the azimuthally averaged power spectrum :math:`P(k)` of
the image, for integer :math:`k = 1, \dots, N/2 - 1` (:ref:`notation`),
computed by `turbulens.spectral.radial_power_spectrum`.

**Band-edge detection (synthetic images).** Normalize the spectrum by
its peak, :math:`p(k) = P(k)/\max_k P(k)`. Estimate a noise floor as the
3rd percentile :math:`p_3` of :math:`p` over all :math:`k`. Then

.. math::
   :label: eq-band-edge

   \hat k_\mathrm{min} = \min\{k : p(k) \ge f_\mathrm{lo}\}, \qquad
   \hat k_\mathrm{max} = \max\{k : p(k) > \max(f_\mathrm{hi},\ 3\,p_3)\},

with :math:`f_\mathrm{lo} = 0.1` and :math:`f_\mathrm{hi} = 10^{-12}` by
default. The upper threshold follows the measured floor, which is set by
the many empty bins of a band-limited spectrum. A percentile over the
whole spectrum does not depend on where the true edge lies, unlike a
fixed window at the highest wavenumbers. If an edge lands on the first
or last available bin, the true edge may lie beyond the measurable range
and a warning is issued. A spectrum with no bin above the thresholds
(pure noise, a constant image) returns ``nan`` for both edges.

**Inertial-range fit (real images).** Fit a straight line,
:math:`\log_{10} P = \alpha \log_{10} k + c`, by least squares over a
central range :math:`k \in [k_\mathrm{lo}, k_\mathrm{hi}]`, where
:math:`k_\mathrm{lo} = \max(2, \lfloor 0.05\,K \rfloor)` and
:math:`k_\mathrm{hi} = \lfloor 0.70\,K \rfloor` with :math:`K = N/2`.
This excludes the largest scales and the noise-dominated smallest
scales. The *inertial range* is then the contiguous run of wavenumbers,
containing the fit range, where :math:`\log_{10}P` stays within 0.5 dex
of the fitted line. Its ends are reported as :math:`\hat k_\mathrm{min}`
and :math:`\hat k_\mathrm{max}`: the scales where the spectrum departs
from a single power law, interpreted as the injection scale and the
dissipation, resolution or noise scale (:ref:`turbulence`).

The fit also returns the slope :math:`\alpha` and the coefficient of
determination :math:`r^2` of the line over the fit range. A small
:math:`r^2` means the spectrum is not a power law at all, and
:math:`\alpha` is then meaningless. On a *linear* map with averaged
power :math:`\propto k^{\beta-2}` the slope estimates :math:`\beta - 2`
(:ref:`notation`), and for a linear integrated-intensity map of 3D
turbulence the same relation links :math:`\alpha` to the 3D index
through :eq:`eq-projection`. The CLI applies the estimators to the
selected map as it is, in linear units, so this reading applies to
``--method fft-powerlaw`` on real data. On a log-transformed image, such
as a synthetic :math:`\log_{10}\rho` training image, :math:`\alpha` is not
:math:`\beta - 2` (:ref:`lognormal-fields`).

Two edge cases:

- If *no* wavenumber lies within 0.5 dex of the fitted line, the
  fit-range limits :math:`k_\mathrm{lo}, k_\mathrm{hi}` are returned with
  a warning: they are configuration choices, not measured scales.
- If the spectrum *never* departs by more than 0.5 dex, the run covers
  the whole grid and the estimator returns the grid limits
  (:math:`k = 1` and :math:`N/2 - 1`) **without** a warning. Treat edges
  at the grid limits as "no departure found", not as measured scales.

**When to use which.**

- Synthetic, band-limited pyFC images: band-edge detection
  (``--method fft-band``). The fit-based estimator would place its edges
  where the power law ends, which for band-limited images is the band
  edge only by coincidence.
- Real observations with continuous spectra: the inertial-range fit
  (``--method fft-powerlaw``). Band-edge detection assumes empty bins
  outside the band, which real data never have.

In turbulens
------------

- `turbulens.spectral.estimate_band_edges` and
  `turbulens.spectral.fit_inertial_range` implement the two estimators.
- `turbulens.spectral.SpectralEstimator` wraps either one behind the same
  ``.predict()`` interface as `turbulens.inference.Inferencer`. It
  returns a table with ``k_min`` and ``k_max`` (plus ``alpha`` and
  ``r_squared`` for the fit), with ``epistemic_std`` always ``nan``.
- From the command line: ``turbulens infer --method fft-band`` or
  ``--method fft-powerlaw``. Neither needs ``--model``.
