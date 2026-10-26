.. docs/turbulens/theory/notation.rst

.. _notation:

Notation and Conventions
========================

The same symbols are used on every page of this documentation. Where a
convention differs between references, the one listed here is the one the
code uses; each page that relies on it links back here.

Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Symbol
     - Meaning
   * - :math:`N`
     - Number of pixels along each side of a square image. The published
       models use :math:`N = 128`.
   * - :math:`\mathbf{k},\ k = |\mathbf{k}|`
     - Wavevector and wavenumber, in cycles per image side. :math:`k`
       takes integer values on the discrete Fourier grid; :math:`k = 1`
       is one full wave across the image.
   * - :math:`k_\mathrm{Nyq} = N/2`
     - Nyquist wavenumber, the largest representable :math:`k` along an
       axis (64 for :math:`N = 128`).
   * - :math:`\rho(\mathbf{x})`
     - Density field (dimensionless, in units of its mean).
   * - :math:`s = \ln \rho`
     - Logarithmic density. The images the models see are
       :math:`\log_{10}\rho = s / \ln 10`.
   * - :math:`\mu,\ \sigma`
     - Mean and standard deviation of the log-normal field :math:`\rho`
       itself (pyFC's ``mean`` and ``sigma``). The training data fix
       :math:`\mu = 1`.
   * - :math:`\mu_s,\ \sigma_s`
     - Mean and standard deviation of the Gaussian field :math:`s` from
       which :math:`\rho` is built; see :ref:`lognormal-fields`.
   * - :math:`D(\mathbf{k})`
     - Per-mode power spectrum: the squared modulus of the Fourier
       amplitude at :math:`\mathbf{k}`.
   * - :math:`P(k)`
     - Azimuthally (or spherically) *averaged* power: the mean of
       :math:`D(\mathbf{k})` over all modes with :math:`|\mathbf{k}|`
       in a unit-width bin around :math:`k`.
   * - :math:`E(k)`
     - Shell-integrated ("energy") spectrum: the *sum* of
       :math:`D(\mathbf{k})` over the shell, so :math:`E(k) = 2\pi k\,P(k)`
       in 2D and :math:`E(k) = 4\pi k^2 P(k)` in 3D.
   * - :math:`\beta`
     - Spectral index of the shell-integrated spectrum of a 3D field,
       :math:`E_\mathrm{3D}(k) \propto k^{\beta}`. Negative; Kolmogorov
       turbulence has :math:`\beta = -5/3`.
   * - :math:`k_\mathrm{min},\ k_\mathrm{max}`
     - Lower and upper wavenumber of the band in which the generated
       field has power. Outside :math:`[k_\mathrm{min}, k_\mathrm{max}]`
       the generating spectrum is zero.

Conventions
-----------

**Power-law spectra.** pyFC :cite:p:`pyfc` prescribes the per-mode
spectrum

.. math::
   :label: eq-pyfc-spectrum

   D(\mathbf{k}) \propto
   \begin{cases}
     |\mathbf{k}|^{\beta - 2}, & k_\mathrm{min} \le |\mathbf{k}| \le k_\mathrm{max},\\
     0, & \text{otherwise,}
   \end{cases}

whatever the dimensionality of the field. For a 3D field the
shell-integrated spectrum is then :math:`E(k) = 4\pi k^2 D \propto k^{\beta}`,
so :math:`\beta` is the familiar energy-spectrum index. For a 2D field
built with the same rule, the azimuthally averaged power is
:math:`P(k) \propto k^{\beta-2}` and the ring-integrated spectrum is
:math:`E_\mathrm{2D}(k) \propto k^{\beta - 1}`.

**What the training fields are.** The training fields are 2D fields
generated directly on a :math:`128 \times 128` grid (a pyFC "cube" one
cell deep), not slices cut from 3D cubes; see :ref:`synthetic-data`.
:ref:`projection-and-slices` explains how such 2D fields relate to
slices and projections of 3D turbulence.

**Which field the spectrum describes.** This distinction matters for
everything that follows:

- The *linear* density :math:`\rho` is the field whose spectrum pyFC
  tunes. Its averaged power follows :math:`P(k) \propto k^{\beta-2}`
  across the band, but it is not band-limited: exponentiation spreads
  some power to wavenumbers outside :math:`[k_\mathrm{min}, k_\mathrm{max}]`.
- The *images* the models see are :math:`\log_{10}\rho`. They are exactly
  band-limited (no power outside the band), but their in-band slope is
  **not** :math:`\beta - 2`. It is close to it only for weak contrast
  (small :math:`\sigma`) and becomes much steeper as :math:`\sigma` grows,
  because the Gaussian field must be shaped very differently for its
  exponential to follow the target spectrum.
- The labels :math:`\beta`, :math:`k_\mathrm{min}`, :math:`k_\mathrm{max}`
  and :math:`\sigma` always describe :math:`\rho`, not the image.

So :math:`\beta` cannot be read off a training image's spectrum
directly; the network has to learn the relation, which depends on
:math:`\sigma`. See :ref:`lognormal-fields`.

**Discrete Fourier transform.** All spectra are computed with the
unnormalized discrete Fourier transform of NumPy :cite:p:`harris2020`.
Absolute power levels therefore depend on :math:`N` and are never
used; only spectral *shapes* (slopes and band edges) carry meaning.
`turbulens.spectral.radial_power_spectrum` returns :math:`P(k)` for
integer :math:`k` from 1 to :math:`N/2 - 1`, averaging
:math:`D(\mathbf{k})` over pixels whose distance from the zero-frequency
pixel rounds down to :math:`k`.

**Logarithms.** :math:`\ln` is the natural logarithm and
:math:`\log_{10}` the decimal one. The log-normal construction uses
:math:`\ln`; the images and the classical spectral fits use
:math:`\log_{10}`.
