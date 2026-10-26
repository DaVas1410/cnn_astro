.. docs/turbulens/method/synthetic_data.rst

.. _synthetic-data:

Synthetic Training Data
=======================

Intuition
---------

The models learn by example: images whose four parameters are known
exactly because they were chosen before each image was made. Every
image is a 2D log-normal fractal field generated with pyFC
(:ref:`lognormal-fields`) from its own randomly drawn
:math:`(k_\mathrm{min}, k_\mathrm{max}, \sigma, \beta)`. The ranges of
those draws define what the models can know: they have never seen a
field outside them, and their outputs are confined to them
(:ref:`output mapping <denormalization>`).

Formalism
---------

**Parameter sampling.** For each image, the four labels are drawn
independently and uniformly:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Range
     - Notes
   * - :math:`k_\mathrm{min}`
     - :math:`\mathcal{U}[1, 32]`
     - real-valued, cycles per image side
   * - :math:`k_\mathrm{max}`
     - :math:`\mathcal{U}[34, 64]`
     - real-valued; 64 is the Nyquist limit of a 128-pixel image
   * - :math:`\sigma`
     - :math:`\mathcal{U}[0.01, 5]`
     - standard deviation of the linear density
   * - :math:`\beta`
     - :math:`\mathcal{U}[-3, -1]`
     - spectral index, :eq:`eq-pyfc-spectrum`
   * - :math:`\mu`
     - fixed at 1
     - mean of the linear density, not a target

The :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}` ranges do not
overlap, so :math:`k_\mathrm{max} > k_\mathrm{min}` holds for every image
without rejection sampling, and both marginal distributions are exactly
flat. The price is that no training field has a narrow band with both
edges in the same part of the spectrum (for example
:math:`k_\mathrm{min} = 20,\ k_\mathrm{max} = 25`), and no field has
:math:`k_\mathrm{max} < 34` or :math:`k_\mathrm{min} > 32`.

**Images.** Each field is generated on a :math:`128 \times 128` grid as
a pyFC cube one cell deep, so it is a genuine 2D field rather than a
slice of a 3D cube. Each image stores :math:`\log_{10}\rho`. Every image
also has its own random seed, recorded alongside its labels, so any
single image can be regenerated.

**Splits.** Training, validation and test sets are generated separately,
each from its own master seed for the parameter and image-seed random
number generator (1000, 2000 and 3000), with 100,000, 20,000 and 50,000
images. Because the three sets come from different random streams, no
parameter draw is shared between them. The per-image noise seeds are
drawn from a large range independently in each split, so a handful
coincide by chance; those images share a white-noise realization but
have different parameters.

A second, smaller set (10,000 / 2,000 / 5,000 images, same generator
and ranges) was used only to screen the hyperparameter grid
(:doc:`/training_pipeline/hyperparameter_search`). It is not part of the
published dataset and was not used to train or evaluate the published
models.

**File layout.** Each split is one HDF5 file with:

- ``images``: float32, shape ``(n, 128, 128)``, the :math:`\log_{10}\rho`
  images;
- ``parameters/k_min``, ``parameters/k_max``, ``parameters/sigma``,
  ``parameters/beta``, ``parameters/mean``: float32 labels, shape
  ``(n,)``;
- ``parameters/ni``, ``parameters/nj``, ``parameters/nk`` (grid shape,
  here 128, 128, 1) and ``parameters/seed`` (per-image seed);
- file attributes recording the generator, the sampling scheme and
  ranges, the image dimensions and the master seed.

The training pipeline reads only ``images`` and the target labels; see
:doc:`/training_pipeline/overview`.

**Getting the data and the generator.** The three files are published
in the Hugging Face dataset repository
``Davas1410/turbulens-synthetic-data``. Its ``generation/`` folder holds
the exact generation scripts together with the bundled copy of pyFC that
enforces the :math:`k_\mathrm{max}` band limit. Upstream pyFC does not
reproduce these images (:ref:`lognormal-fields`). At the time of writing
the repository is private; access must be requested from the authors
until it is made public.

In turbulens
------------

- The ranges above are stored in every model's configuration as
  ``target_ranges`` (with :math:`\sigma` stored as :math:`[0, 5]`) and
  used to map network outputs back to physical units
  (:ref:`output mapping <denormalization>`).
- `turbulens` does not generate data. Regenerating or extending the
  dataset means running the generator from the dataset repository.
