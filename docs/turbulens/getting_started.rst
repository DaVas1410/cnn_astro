.. docs/turbulens/getting_started.rst

Getting Started
===============

This page takes a real HI observation from file to predictions in three
commands. It assumes Python 3.10 or newer.

Install
-------

From a checkout of the repository:

.. code-block:: bash

   pip install -e ".[image]"

The ``image`` extra adds PNG/TIFF support. FITS and ``.npy`` inputs need
no extras.

1. Look at the data
-------------------

.. code-block:: bash

   turbulens inspect --input turbulens/io/tests/data/gass_0_0_1774393774.fits.gz

``inspect`` prints the cube's shape, units, velocity range and pixel
statistics. This example is a GASS sub-cube (:ref:`projection-and-slices`):
1201 velocity channels of :math:`125 \times 125` pixels, in kelvin,
spanning about :math:`\pm 495` km/s. Knowing the channel count and
velocity range is what you need to choose a range in the next step.
``--json`` and ``--html`` write the same information to a file.

2. Predict with the pretrained ensemble
---------------------------------------

.. code-block:: bash

   turbulens infer --input turbulens/io/tests/data/gass_0_0_1774393774.fits.gz \
       --velocity -100 100 --real-data --model auto --out predictions.ecsv

- ``--velocity -100 100`` sums the channels between -100 and +100 km/s
  into one integrated map. A multi-channel input needs exactly one of
  ``--velocity``, ``--integrate LOW:HIGH`` (channel indices) or
  ``--channel N``; :ref:`projection-and-slices` explains why a wide
  velocity range is the physically appropriate choice.
- ``--real-data`` is required for any real observation. It
  log-transforms the map and moves it onto the training scale without
  changing its contrast (:ref:`real-data normalization <real-data-normalization>`).
  Leaving it out on real data gives confident-looking but meaningless
  numbers.
- ``--model auto`` uses the published multitask ensemble. The first run
  downloads it (about 450 MB) from Hugging Face and caches it; later
  runs use the cache (:doc:`user_guide/pretrained_models`). While the
  model repository is private, run ``hf auth login`` with an account that
  has access first.
- ``--out predictions.ecsv`` writes the table. Without ``--out`` it is
  printed.

3. Read the result
------------------

The table has one row per target:

``target``
   ``k_min``, ``k_max``, ``sigma`` or ``beta`` (:ref:`the four parameters <pyfc-parameters>`).
``value``
   The ensemble mean, in physical units.
``epistemic_std``
   The spread between the five ensemble members (:ref:`uncertainty`).
   It is not a calibrated error bar.

The file's header records how the result was produced: the input, the
channel selection, the model repository and the exact model commit,
and the fraction of pixels that fell outside the training range
(:doc:`user_guide/outputs_provenance`). Before relying on any number,
read :doc:`user_guide/interpreting_results`.

A model-free cross-check
------------------------

.. code-block:: bash

   turbulens infer --input turbulens/io/tests/data/gass_0_0_1774393774.fits.gz \
       --velocity -100 100 --method fft-powerlaw

This fits a power law to the map's spectrum and reports where it holds
(``k_min``, ``k_max``), its slope ``alpha`` and the fit quality
``r_squared``, with no network involved (:ref:`spectral-estimators`).
Agreement between the two methods is reassuring; disagreement is worth
investigating.

Next steps
----------

- :doc:`user_guide/cli` lists every option.
- :doc:`user_guide/python_api` does the same from Python.
- :doc:`theory/index` and :doc:`method/index` explain what the numbers
  mean and how they are produced.
