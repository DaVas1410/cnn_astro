.. docs/turbulens/index.rst

turbulens
=========

`turbulens` estimates the parameters of interstellar turbulence from a
single 2D map: the range of spatial scales the turbulent structure spans
(:math:`k_\mathrm{min}`, :math:`k_\mathrm{max}`), the slope of its power
spectrum (:math:`\beta`) and its density contrast (:math:`\sigma`). It
uses a deep ensemble of convolutional neural networks trained on
synthetic log-normal fractal fields, and it reads radio-astronomy FITS
cubes, such as HI 21-cm surveys, as well as plain images. Classical,
model-free spectral estimators are included as a cross-check.

Installation
------------

.. code-block:: bash

   pip install -e ".[image]"  # from a checkout of this repository

Quickstart
----------

.. code-block:: bash

   # Look at a cube: shape, units, velocity range
   turbulens inspect --input observation.fits

   # Predict from the HI integrated over -100..100 km/s; --model auto
   # downloads the pretrained ensemble on first use
   turbulens infer --input observation.fits --velocity -100 100 \
       --real-data --model auto --out predictions.ecsv

   # Model-free cross-check of the band edges
   turbulens infer --input observation.fits --velocity -100 100 \
       --method fft-powerlaw

:doc:`getting_started` walks through these commands on a real GASS cube.

Getting an ensemble
-------------------

``--model auto`` downloads the pretrained ensemble on first use and
caches it. See :doc:`user_guide/pretrained_models` for pre-downloading,
offline use, version pinning and your own ensembles.

Where to read next
------------------

**If you want to run it,** start with :doc:`getting_started`, then the
:doc:`user_guide/index`. Read :doc:`user_guide/interpreting_results` and
:doc:`limitations` before using any number in your work.

**If you are learning the subject,** read :doc:`theory/index` in order:
power spectra and turbulence, log-normal fractal fields, and how both
relate to HI observations. Each page starts with a plain-language
summary.

**If you are checking the method,** :doc:`method/index` gives every
transformation, the network, the loss, and what the uncertainty means,
each with the exact function that implements it.
:doc:`training_pipeline/index` documents how the published models were
selected and trained, including the leakage guards.

All equations and claims taken from the literature are cited; the
:doc:`references` list every source. The documentation explains the
method; results and their numerical values are reported in the
accompanying paper. To cite `turbulens`, see :doc:`citing`.

.. note::

   Equations are rendered with MathJax, which is loaded from a CDN. Viewed
   offline, they appear as raw LaTeX.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   theory/index
   method/index
   training_pipeline/index
   user_guide/index
   limitations
   api/index
   citing
   references
