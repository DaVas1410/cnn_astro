.. docs/turbulens/theory/index.rst

Theory
======

The physical background for the quantities `turbulens` predicts. Read
these pages in order if the subject is new to you; each one starts with
a plain-language summary before the equations.

- :doc:`notation` fixes the symbols and conventions used everywhere
  else, in particular what :math:`\beta` means for a 2D image.
- :doc:`turbulence` introduces turbulent cascades, power spectra and
  the band edges :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}`.
- :doc:`lognormal_fields` explains why interstellar density is close to
  log-normal, what :math:`\sigma` measures, and how pyFC builds the
  synthetic fields the models are trained on.
- :doc:`projection_observations` connects those fields to real HI
  21-cm observations such as the GASS survey.

.. toctree::
   :maxdepth: 1
   :hidden:

   notation
   turbulence
   lognormal_fields
   projection_observations
