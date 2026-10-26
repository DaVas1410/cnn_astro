.. docs/turbulens/user_guide/outputs_provenance.rst

Output Tables and Provenance
============================

Columns
-------

``turbulens infer`` produces one `astropy.table.QTable` row per image
and target:

``image_index``
   Index of the image in the input (always 0 for the CLI).
``target``
   ``k_min``, ``k_max``, ``sigma`` or ``beta`` for the CNN; ``k_min``,
   ``k_max`` (and ``alpha``, ``r_squared`` for ``fft-powerlaw``) for the
   classical methods.
``value``
   The ensemble mean (:eq:`eq-ensemble`), or the classical estimate.
``epistemic_std``
   The ensemble spread (:ref:`uncertainty`); ``nan`` for the classical
   methods, which have no ensemble.

Metadata
--------

The table's metadata records how it was produced:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Meaning
   * - ``input_path``
     - The input file.
   * - ``method``
     - ``cnn``, ``fft-band`` or ``fft-powerlaw``.
   * - ``channel``, ``integrate``, ``velocity``
     - The channel selection used (unused ones are null).
   * - ``turbulens_version``
     - The installed `turbulens` version.
   * - ``model``
     - CNN only: the local ``ensemble/members`` directory used.
   * - ``model_repo``, ``model_revision``
     - CNN with a Hugging Face download only: the repository and the
       exact commit hash of the weights.
   * - ``real_data``
     - CNN only: whether the real-observation normalization was applied.
   * - ``n_members``
     - CNN only: the number of ensemble members.
   * - ``clipped_pixel_fraction_max``
     - CNN only: the largest fraction of pixels, over all members, that
       fell outside the training pixel range (:ref:`distribution-shift
       warning <ood-warning>`).

File formats
------------

``--out`` chooses the format from the suffix:

- ``.ecsv`` (recommended) keeps the metadata, so the file alone is
  enough to reproduce the result.
- ``.csv`` and ``.fits`` keep only the columns, and a warning says so.

From Python, `turbulens.io.results.write_results` writes a table the
same way; ``astropy.table.QTable.read("predictions.ecsv")`` reads it back
with metadata intact.

Reproducing a result
--------------------

With ``model_revision`` and the selection keys, a result can be
regenerated exactly:

.. code-block:: bash

   turbulens infer --input <input_path> --velocity <low> <high> \
       --real-data --model auto --revision <model_revision>

The ensemble mean is deterministic for given weights and input on the
same hardware, apart from floating-point differences between CPU and GPU
backends.
