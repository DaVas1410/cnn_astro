.. docs/turbulens/theory/lognormal_fields.rst

.. _lognormal-fields:

Log-normal Fractal Density Fields
=================================

Intuition
---------

Supersonic turbulence compresses and rarefies gas over and over. Each
compression multiplies the local density by a random factor, and a
product of many independent random factors has a logarithm that is a sum
of many random terms. The logarithm is therefore close to Gaussian, and
the density itself close to *log-normal*: most of the volume is
somewhat underdense, while a small fraction is packed into dense
filaments and clumps.

A *fractal* density field adds the spatial side of the picture: its
power spectrum is a power law, so it looks statistically similar at
every scale inside the band. The synthetic fields `turbulens` learns
from combine the two. Their one-point statistics are log-normal with a
chosen contrast :math:`\sigma`. Their two-point statistics are a
power-law spectrum of index :math:`\beta` restricted to the band
:math:`[k_\mathrm{min}, k_\mathrm{max}]`.

.. _lognormal-pdf:

Formalism
---------

**Log-normal density.** In isothermal supersonic turbulence the density
PDF is close to log-normal
:cite:p:`vazquezsemadeni1994,passot1998,padoan1997`. Write
:math:`s = \ln\rho` with :math:`s \sim \mathcal{N}(\mu_s, \sigma_s^2)`.
The mean :math:`\mu` and standard deviation :math:`\sigma` of
:math:`\rho` then follow from :math:`\mu_s, \sigma_s` and back:

.. math::
   :label: eq-lognormal-moments

   \mu = e^{\mu_s + \sigma_s^2/2}, \qquad
   \sigma^2 = \mu^2\left(e^{\sigma_s^2} - 1\right)
   \quad\Longleftrightarrow\quad
   \sigma_s^2 = \ln\!\left(1 + \frac{\sigma^2}{\mu^2}\right), \qquad
   \mu_s = \ln\mu - \frac{\sigma_s^2}{2}.

The width of the log-density distribution grows with the turbulent Mach
number :math:`\mathcal{M}`. Simulations give the relation
:cite:p:`padoan1997,federrath2008`

.. math::
   :label: eq-sigma-mach

   \sigma_s^2 = \ln\!\left(1 + b^2 \mathcal{M}^2\right),

where :math:`b` depends on how the turbulence is driven, from about
:math:`1/3` for purely solenoidal to about :math:`1` for purely
compressive forcing :cite:p:`federrath2008`. With :math:`\mu = 1`, the
density contrast in :eq:`eq-lognormal-moments` is
:math:`\sigma = b\mathcal{M}`. :math:`\sigma` is therefore a direct proxy
for how supersonic and how compressively driven the turbulence is.
`turbulens` predicts :math:`\sigma`, not :math:`\mathcal{M}` or
:math:`b`; turning one into the others needs these assumptions.

**Fractal fields and the Lewis & Austin construction.** A Gaussian
field with any power spectrum is easy to make: filter white noise in
Fourier space by the square root of the target spectrum. Exponentiating
it gives a log-normal field, but the exponentiation changes the
spectrum. :cite:t:`lewis2002` proposed an iterative correction so that
the *log-normal* field ends up with the target spectrum. pyFC
:cite:p:`pyfc` implements it as follows (``LogNormalFractalCube`` in the
bundled copy of pyFC):

1. Convert the requested :math:`(\mu, \sigma)` to :math:`(\mu_s, \sigma_s)`
   with :eq:`eq-lognormal-moments`, and draw Gaussian white noise with
   that mean and standard deviation.
2. Build the filter :math:`\sqrt{D(\mathbf{k})}` from the band-limited
   power law :eq:`eq-pyfc-spectrum`. Its normalization keeps the mean and
   variance of the noise unchanged.
3. Filter the noise in Fourier space, transform back, and exponentiate,
   giving a log-normal field :math:`\rho`.
4. Measure the isotropic spectrum of :math:`\rho` and, over the band, fit
   a quadratic to it in :math:`\log_{10} D`–:math:`\log_{10} k` space.
   Multiply the amplitude filter by :math:`10^{\eta\,\Delta}`, where
   :math:`\Delta` is the log10 difference between the target power law
   and the fit and :math:`\eta = 0.5`. Since the power is the square of
   the amplitude, this corrects the power filter by the full mismatch
   :math:`10^{\Delta}`. The filter is then renormalized.
5. Repeat steps 3–4 until the filter changes by less than 1% on average,
   or for at most about ten iterations. The final :math:`\rho` is the
   exponentiated, filtered field.

Fractal log-normal clouds of this kind are widely used to set up
realistic, inhomogeneous initial conditions in simulations of the
interstellar medium, for example by :cite:t:`sutherland2007`.

Two consequences matter for the models:

- The filter is zero outside :math:`[k_\mathrm{min}, k_\mathrm{max}]` at
  every iteration. The Gaussian field :math:`s = \ln\rho`, and hence the
  :math:`\log_{10}\rho` images the models see, therefore have **no power
  outside the band**: the band edges are sharp in the images. The linear
  field :math:`\rho` is not band-limited: exponentiation spreads some of
  its power outside the band, and only its in-band shape is fitted.
- Inside the band, the filter is tuned so that :math:`\rho`, not
  :math:`s`, follows :math:`k^{\beta-2}`. The more skewed the log-normal
  (the larger :math:`\sigma`), the more the Gaussian field must differ
  from the target to achieve this. As a result the in-band slope of the
  :math:`\log_{10}\rho` image is close to :math:`\beta - 2` only at weak
  contrast, and becomes much steeper as :math:`\sigma` grows. The image
  spectrum alone does not give :math:`\beta`; it gives a combination of
  :math:`\beta` and :math:`\sigma` that the network learns to separate.

.. _pyfc-parameters:

**The four predicted parameters.** For a generated field:

- :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}` are the band edges of
  :eq:`eq-pyfc-spectrum`, in cycles per image side. Both are real
  numbers; the band contains every Fourier mode with
  :math:`k_\mathrm{min} \le |\mathbf{k}| \le k_\mathrm{max}`.
- :math:`\beta` is the spectral index of :eq:`eq-pyfc-spectrum`, defined
  so that :math:`\beta = -5/3` is Kolmogorov in 3D (:ref:`notation`).
- :math:`\sigma` is the requested standard deviation of the linear
  density :math:`\rho` at :math:`\mu = 1`, *not* the standard deviation
  of :math:`\ln\rho` or of the :math:`\log_{10}\rho` image. Those follow
  from :eq:`eq-lognormal-moments` as
  :math:`\sigma_s = \sqrt{\ln(1 + \sigma^2)}`, and :math:`\sigma_s/\ln 10`
  for the image. The label is the generator's *requested* value. The
  standard deviation actually realized in one finite image scatters
  widely around it, by up to roughly a factor of two at high
  :math:`\sigma`, and its mean density scatters around 1. A strongly
  skewed field on a :math:`128 \times 128` grid has few large-scale modes,
  and its variance is dominated by rare dense peaks, so one realization
  is a noisy sample of the ensemble value. The log-space width
  :math:`\sigma_s` is realized much more tightly.

**The Nyquist limit.** pyFC caps :math:`k_\mathrm{max}` at the Nyquist
wavenumber :math:`N/2`. That is 64 for the :math:`128 \times 128`
training images, which is why the training range of
:math:`k_\mathrm{max}` ends at 64.

**The band-limit fix.** The original pyFC applied only the lower cutoff
:math:`k_\mathrm{min}`; the upper cutoff :math:`k_\mathrm{max}` was
stored as a label but never imposed, so the images carried power up to
the Nyquist limit regardless of the label. The copy of pyFC bundled with
the training data generator enforces both cutoffs, as in
:eq:`eq-pyfc-spectrum`. Data generated with an unmodified upstream pyFC
therefore do **not** reproduce the training data, and their
:math:`k_\mathrm{max}` labels do not correspond to a feature of the
image. See :ref:`synthetic-data` for where to get the bundled copy.

In turbulens
------------

- Every model was trained on :math:`\log_{10}\rho` images of fields made
  this way, with the ranges listed in :ref:`synthetic-data`. A
  prediction of :math:`\sigma` is a prediction of the *linear* density
  contrast in the sense above.
- Real observations are not log-normal fractal fields by construction.
  Whether their statistics resemble these closely enough for the
  predictions to be meaningful is the central question in
  :doc:`/user_guide/interpreting_results`.
