.. docs/turbulens/theory/turbulence.rst

.. _turbulence:

Turbulence and Power Spectra
============================

Intuition
---------

A turbulent flow carries structure on many scales at once. Energy is
injected at large scales, for example by supernova explosions,
expanding H II regions or galactic shear in the interstellar medium
(ISM). It is passed to ever smaller eddies, and finally dissipated by
viscosity, or in the compressible ISM also by shocks, at the smallest
scales :cite:p:`elmegreen2004,maclow2004`. Between the injection scale
and the dissipation scale lies the *inertial range*, where the flow has
no preferred scale and its statistics follow power laws.

A power spectrum measures how much fluctuation a field contains at each
spatial scale. For a turbulent field it has three regions: little power
at scales larger than the injection scale, a power law across the
inertial range, and a steep fall-off below the dissipation scale. The
four quantities `turbulens` predicts describe exactly this shape:
where the power-law band starts (:math:`k_\mathrm{min}`), where it ends
(:math:`k_\mathrm{max}`), how steeply it falls (:math:`\beta`), and how
strong the density contrast is overall (:math:`\sigma`, see
:ref:`lognormal-fields`).

Formalism
---------

**Power spectrum.** For a field :math:`f(\mathbf{x})` on a periodic
:math:`N \times N` grid with discrete Fourier transform
:math:`\hat f(\mathbf{k})`, the per-mode power is

.. math::
   :label: eq-power-spectrum

   D(\mathbf{k}) = \left|\hat f(\mathbf{k})\right|^2 .

For a statistically isotropic field :math:`D` depends only on
:math:`k = |\mathbf{k}|`. The azimuthal average :math:`P(k)` and the
ring- or shell-integrated spectrum :math:`E(k)` are defined in
:ref:`notation`. A power law :math:`E(k) \propto k^{\beta}` means that
every factor-of-two range of scales carries a fixed fraction of the
variance more (or less) than the next; the field is *self-similar*, or
fractal, across that range.

**Kolmogorov scaling.** For incompressible turbulence with a constant
energy flux :math:`\varepsilon` through the inertial range, dimensional
analysis gives :cite:p:`kolmogorov1991`

.. math::
   :label: eq-kolmogorov

   E(k) = C\,\varepsilon^{2/3} k^{-5/3},

with a dimensionless constant :math:`C`. This is the :math:`\beta = -5/3`
case, and pyFC's default index :cite:p:`pyfc`.

**Compressible, supersonic turbulence.** In the cold ISM the turbulent
velocities exceed the sound speed, so much of the kinetic energy is
dissipated in shocks rather than through a viscous cascade.
Shock-dominated flows have steeper velocity spectra, close to
:math:`E(k) \propto k^{-2}`, and the *density* field no longer follows
the velocity spectrum: compressions build dense, intermittent structures
whose spectrum depends on the Mach number and on how the turbulence is
driven :cite:p:`maclow2004,elmegreen2004,mckee2007,hennebelle2012`. A
model of ISM density structure therefore has to allow a range of
spectral indices, not just the Kolmogorov value.

**Observed density spectra.** Power-law spectra of atomic hydrogen are
measured over large ranges of scales, in the Magellanic Clouds
:cite:p:`stanimirovic1999` and in Galactic cirrus
:cite:p:`mivilledeschenes2003`. Interpreting them requires care because
an observed map is a projection along the line of sight and is mixed
with velocity structure :cite:p:`lazarian2000`; see
:ref:`projection-and-slices`.

**Band limits.** Writing the injection and dissipation length scales as
:math:`\ell_\mathrm{inj}` and :math:`\ell_\mathrm{diss}` in pixels, the
corresponding wavenumbers on an :math:`N`-pixel image are

.. math::
   :label: eq-band-edges

   k_\mathrm{min} = N / \ell_\mathrm{inj}, \qquad
   k_\mathrm{max} = N / \ell_\mathrm{diss}.

A larger :math:`k_\mathrm{min}` means the largest coherent structures are
smaller than the image; a larger :math:`k_\mathrm{max}` means structure
continues down to finer scales. No structure finer than two pixels can
be represented, so :math:`k_\mathrm{max} \le k_\mathrm{Nyq} = N/2`.

In turbulens
------------

- The synthetic fields the models learn from are generated from a
  band-limited power law, :eq:`eq-pyfc-spectrum`, with
  :math:`\beta \in [-3, -1]`. The range includes the Kolmogorov
  (:math:`-5/3`) and shock-dominated (:math:`-2`) indices. Those are
  velocity-spectrum values, often used as reference points for density
  too, although in compressible turbulence the density spectrum need not
  follow the velocity spectrum. See :ref:`synthetic-data` for the other
  ranges.
- `turbulens.spectral.radial_power_spectrum` computes :math:`P(k)` of an
  image as in :eq:`eq-power-spectrum`, averaged over integer-radius
  rings (:ref:`notation`).
- `turbulens.spectral.fit_inertial_range` fits a straight line to
  :math:`\log_{10} P(k)` against :math:`\log_{10} k` and reports its slope
  as ``alpha``. For a *linear* field whose averaged power falls as
  :math:`k^{\beta - 2}` (the convention of :eq:`eq-pyfc-spectrum`),
  ``alpha`` estimates :math:`\beta - 2`, not :math:`\beta` itself. On a
  log-transformed image, such as a synthetic training image, it does not
  (:ref:`notation`). See :doc:`/method/spectral_estimators`.
- Real interstellar spectra rarely have sharp band edges: the injection
  scale is usually larger than the field of view, and the dissipation
  scale is far below the telescope's resolution. What sets the apparent
  :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}` of an observed map is
  discussed in :ref:`projection-and-slices` and
  :doc:`/user_guide/interpreting_results`.
