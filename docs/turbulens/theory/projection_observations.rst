.. docs/turbulens/theory/projection_observations.rst

.. _projection-and-slices:

Projection and HI Observations
==============================

Intuition
---------

A telescope never sees a slice through a cloud. It sees everything
along the line of sight added up, and for a spectral line it sorts that
emission by line-of-sight velocity. The 2D images `turbulens` works on
are therefore one of two things: a sum over the whole line of sight (an
integrated map) or a narrow velocity range (a channel map). Each relates
differently to the 3D density field whose turbulence we want to
characterize, and the relation fixes what the predicted :math:`\beta` can
mean for a real map.

.. _observations:

Formalism
---------

**Slices and projections of a 3D field.** Let a 3D isotropic field have
per-mode spectrum :math:`D_\mathrm{3D}(k) \propto k^{\beta - 2}`, that is
:math:`E_\mathrm{3D} \propto k^{\beta}` (:ref:`notation`). By the
projection-slice theorem, the 2D Fourier transform of its projection
along :math:`z` equals the 3D transform on the plane :math:`k_z = 0`.
Integrating through a depth much larger than the structures of interest
therefore gives a map with

.. math::
   :label: eq-projection

   P_\mathrm{proj}(k) \propto D_\mathrm{3D}(k, k_z = 0) \propto k^{\beta - 2},

the *same* index as the 3D per-mode spectrum. A thin slice instead
averages the 3D spectrum over all :math:`k_z`, which for a steep spectrum
gives

.. math::
   :label: eq-slice

   P_\mathrm{slice}(k) \propto \int D_\mathrm{3D}\!\left(\sqrt{k^2 + k_z^2}\right) dk_z
   \propto k^{\beta - 1},

one power shallower. Maps of intermediate thickness change from one
behaviour to the other at the wavenumber matching the depth. These
relations, and their consequences for HI maps, are worked out in
:cite:t:`lazarian2000`.

The *linear* training fields :math:`\rho` obey
:math:`P(k) \propto k^{\beta - 2}` (:ref:`notation`). In spectral shape
they therefore resemble the *projection* of a 3D field with energy index
:math:`\beta`, which is what a linear integrated-intensity map measures,
rather than a thin slice through it. The comparison holds between
*linear* fields: a real map with linear-intensity power
:math:`\propto k^{\alpha}` corresponds, under this reading, to
:math:`\beta = \alpha + 2`. It does not hold for the log-transformed
images, whose spectra differ from the linear ones in a
:math:`\sigma`-dependent way (:ref:`lognormal-fields`); both training
fields and real maps are log-transformed before the network sees them.
Projection also narrows the one-point PDF (sums of many log-normal
values are less skewed), so a projected map is not itself a log-normal
field with the 3D :math:`\sigma`.

**The HI 21-cm line.** Neutral atomic hydrogen emits at 1420.4 MHz
(21 cm) through the hyperfine spin-flip transition. Radio surveys record
the brightness temperature :math:`T_B(l, b, v)` as a function of sky
position and line-of-sight velocity: a spectral cube. Where the line is
optically thin, the column density follows directly from the
velocity-integrated brightness temperature :cite:p:`dickey1990,kalberla2009`:

.. math::
   :label: eq-hi-column

   N_\mathrm{HI} = 1.823 \times 10^{18}\,\mathrm{cm^{-2}}
   \int \frac{T_B(v)}{\mathrm{K}}\,\frac{dv}{\mathrm{km\,s^{-1}}} .

An integrated-intensity map is then proportional to :math:`N_\mathrm{HI}`,
the projected density of :eq:`eq-projection`. Optical depth and
self-absorption break the proportionality in the densest, coldest gas.

**Channel maps and velocity structure.** A single narrow velocity
channel is not a density map. Emission at one velocity comes from all
gas moving at that velocity, wherever it sits along the line of sight,
so velocity fluctuations alone create structure in the map. For a
steep density spectrum, a channel narrow compared with the turbulent
velocity dispersion has a power spectrum dominated by velocity rather
than density statistics, and only a wide enough channel, or the whole
line integrated, is dominated by density. For a shallow density
spectrum, density can dominate even thin channels
:cite:p:`lazarian2000`. For density-structure
parameters such as those `turbulens` predicts, integrated or wide
velocity ranges are the physically appropriate input.

**Continuous spectra.** Observed HI power spectra are power laws over
the whole observed range :cite:p:`stanimirovic1999,mivilledeschenes2003,martin2015`.
They show no sharp band edges like the synthetic training fields
(:ref:`lognormal-fields`). The largest scales are cut by the field of
view; the smallest are smoothed by the telescope beam and buried in
noise. The "band" of a real map is therefore set at least partly by the
observation, not by the turbulence.

**GASS.** The Parkes Galactic All-Sky Survey (GASS) maps Galactic HI over
the southern sky (declinations :math:`\delta \le 1^\circ`, about
:math:`2\pi` sr) with the Parkes 64-m telescope, at about 16 arcmin
angular resolution, 1.0 km/s velocity resolution and 57 mK rms
brightness-temperature noise :cite:p:`mccluregriffiths2009`. Later
releases corrected stray radiation and other instrumental effects
:cite:p:`kalberla2010,kalberla2015`. GASS sub-cubes are the example real
data used throughout this documentation and the test suite.

In turbulens
------------

- `turbulens.io.cube.Cube.slice_channel` selects one channel: a channel
  map, dominated by velocity structure when the channel is narrow.
- `turbulens.io.cube.Cube.project` and
  `turbulens.io.cube.Cube.project_velocity` sum a channel range (by
  default; ``mode="mean"`` averages instead). Over a wide range this is
  proportional to the integrated intensity, and so to
  :math:`N_\mathrm{HI}` where the gas is optically thin. Constant factors
  such as the channel width do not matter, because the real-data
  normalization removes the overall level (:ref:`real-data normalization <real-data-normalization>`).
- The CLI exposes these as ``--channel``, ``--integrate`` and
  ``--velocity`` (:doc:`/user_guide/cli`).
- How beam smoothing, noise, finite fields of view and the
  slice-versus-projection difference affect predictions on real maps is
  covered in :doc:`/user_guide/interpreting_results`.
