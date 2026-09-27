.. docs/turbulens/references.rst

References
==========

External libraries
-------------------

- `astropy <https://www.astropy.org/>`_: FITS I/O, WCS, units, and
  table I/O (`Cube`, `write_results`).
- `PyTorch <https://pytorch.org/>`_ and `torchvision
  <https://pytorch.org/vision/stable/index.html>`_: the CNN backbone
  and training/inference runtime.
- `Pillow <https://pillow.readthedocs.io/>`_: PNG/TIFF image loading
  (optional, ``pip install turbulens[image]``).
- `pyFC <https://bitbucket.org/pandante/pyfc>`_: the log-normal
  fractal cube generator that defines the physical meaning of `k_min`,
  `k_max`, `sigma`, and `beta`. See :doc:`concepts`.

Repository documentation
--------------------------

- ``archive/docs/RESEARCH_JOURNEY.md``: the full narrative of how the current
  model architecture, normalization strategy, and ensembling approach
  were reached.
- ``archive/docs/bugs/pyfc-kmax-not-enforced.md``: the `k_max` spectral-cutoff
  bug referenced in :doc:`concepts`.
- ``docs/superpowers/specs/2026-08-18-turbulens-tool-design.md``: the
  original design spec for the `turbulens` package itself.
- ``outputs/comparison/local_v2/comparison_report.md``: the
  multitask-vs-single-target comparison referenced in :doc:`concepts`
  and in `turbulens.models.architecture`'s module docstring.
