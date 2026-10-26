.. docs/turbulens/user_guide/inputs.rst

Input Data
==========

Supported formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Format
     - Notes
   * - FITS (``.fits``, ``.fits.gz``)
     - A spectral cube or a 2D image. ``--hdu`` selects the HDU (default
       0). The header's WCS and ``BUNIT`` are kept.
   * - ``.npy``
     - A single 2D array. Loaded without pickle support, so a ``.npy``
       file cannot execute code.
   * - PNG, TIFF
     - Needs ``pip install "turbulens[image]"``. Grayscale and float modes
       are read directly. Colour images are converted to grayscale, which
       discards information.
   * - ``.npz``
     - Rejected: an archive can hold several arrays and there is no safe
       way to guess which one is the image. Save the one you want as
       ``.npy``.

Every input becomes a `turbulens.io.cube.Cube` of shape
``(n_channels, H, W)``. A 2D image is a cube with one channel. FITS data
with more than three axes is reduced by dropping size-one axes, such as a
degenerate Stokes axis.

What the models expect
----------------------

The models were trained on :math:`128 \times 128` images of
:math:`\log_{10}` density (:ref:`synthetic-data`). Inputs are **not**
resized. The network accepts other sizes, but its notion of scale is
tied to training-image pixels, so :math:`k` predictions on a different
size need care (:doc:`interpreting_results`).

- **Synthetic** :math:`\log_{10}\rho` images, such as new pyFC fields,
  are already on the training scale: run without ``--real-data``.
- **Real observations** in linear physical units (kelvin, Jy/beam,
  column density) need ``--real-data`` (:ref:`real-data normalization <real-data-normalization>`).
  Non-positive and NaN pixels (for example blanked FITS borders) are
  raised to the image's smallest positive value before the logarithm;
  positive-infinite pixels are set to the lower training bound.

Choosing channels
-----------------

A multi-channel cube needs exactly one of:

``--channel N``
   One channel (0-indexed): a channel map.
``--integrate LOW:HIGH``
   The sum of channels ``LOW`` to ``HIGH - 1`` (half-open, like Python
   slices).
``--velocity LOW HIGH``
   The sum of all channels whose velocity lies in ``[LOW, HIGH]`` km/s.

A single-channel input takes none of them. For density-structure
parameters a wide velocity range is the physically appropriate choice;
narrow channels are dominated by velocity structure
(:ref:`projection-and-slices`).

Spectral axes
-------------

Velocity selection needs a spectral axis that `turbulens` can turn into
velocities:

- The spectral axis is found from ``CTYPE`` prefixes: ``VELO``, ``VOPT``
  and ``VRAD`` (velocity) or ``FREQ`` (frequency). This includes the
  older 9-character forms such as ``VELO-LSR`` or ``VELO-LSRK`` that
  radio data often use.
- Frequency axes are converted to radio-convention velocities with the
  header's rest frequency, or the HI 21-cm rest frequency
  (1420.405751 MHz) if none is given; a warning names the rest frequency
  used.
- Velocity units written with a capital ``S``, as in ``M/S``, which a
  strict reading would take as siemens, are interpreted as seconds, a
  common quirk of older radio FITS files. Astropy prints
  ``FITSFixedWarning`` messages when it repairs such headers; they are
  informational.

``turbulens inspect`` shows the velocity range it found. If it reports
none, use ``--integrate`` or ``--channel`` instead of ``--velocity``.
