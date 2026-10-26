.. docs/turbulens/references.rst

References
==========

Every reference below was checked against its DOI record, arXiv entry or
publisher page before it was added. Preprints are cited by the arXiv
version that was checked.

Bibliography
------------

.. bibliography::

Software
--------

`turbulens` is built on the following open-source packages:

- `astropy <https://www.astropy.org/>`_ :cite:p:`astropy2013,astropy2018,astropy2022`:
  FITS I/O, WCS, units, and table I/O (`Cube`, `write_results`).
- `PyTorch <https://pytorch.org/>`_ :cite:p:`paszke2019` and
  `torchvision <https://pytorch.org/vision/stable/index.html>`_: the CNN
  backbone and the training and inference runtime.
- `NumPy <https://numpy.org/>`_ :cite:p:`harris2020` and
  `SciPy <https://scipy.org/>`_ :cite:p:`virtanen2020`: array computing,
  Fourier transforms and fitting in the classical spectral estimators.
- `scikit-learn <https://scikit-learn.org/>`_ :cite:p:`pedregosa2011`:
  PCA and CCA in the training pipeline's interpretability stage.
- `huggingface_hub <https://huggingface.co/docs/huggingface_hub>`_:
  downloading and caching the pretrained ensembles.
- `Pillow <https://pillow.readthedocs.io/>`_: PNG/TIFF image loading
  (optional, ``pip install "turbulens[image]"``).
- `pyFC <https://bitbucket.org/pandante/pyfc>`_ :cite:p:`pyfc`: the
  log-normal fractal cube generator that defines the physical meaning of
  `k_min`, `k_max`, `sigma` and `beta`. The training data were generated
  with a bundled copy that enforces the `k_max` band limit; see
  :doc:`theory/lognormal_fields`.
