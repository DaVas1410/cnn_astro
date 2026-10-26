.. docs/turbulens/user_guide/cli.rst

Command-Line Interface
=======================

`turbulens` installs a single console script, ``turbulens``, with three
subcommands.

``infer``
---------

Predict fractal turbulence parameters (`k_min`, `k_max`, `sigma`, and
`beta` when the loaded ensemble includes it) from a FITS cube or a
plain image.

.. argparse::
   :module: turbulens.cli
   :func: build_parser
   :prog: turbulens
   :path: infer

``inspect``
-----------

Summarize a cube's shape, WCS, spectral coverage, and pixel statistics
without running a model. Useful for sanity-checking an input before
choosing ``--channel``/``--integrate``/``--velocity`` for ``infer``.

.. argparse::
   :module: turbulens.cli
   :func: build_parser
   :prog: turbulens
   :path: inspect

``download``
------------

Download pretrained ensembles into the local Hugging Face cache ahead of
time. Not required: ``infer --model auto`` downloads what it needs on
first use. See :doc:`pretrained_models`.

.. argparse::
   :module: turbulens.cli
   :func: build_parser
   :prog: turbulens
   :path: download

Examples
--------

.. code-block:: bash

   # Inspect a FITS cube before inferring
   turbulens inspect --input observation.fits

   # Predict from a real telescope observation: --real-data is required here,
   # since observation.fits is on a real intensity scale, not the synthetic
   # log-density scale the model was trained on (omitting it silently gives
   # confident-looking but meaningless numbers -- see --real-data's --help).
   # --model auto downloads the pretrained multitask ensemble from the
   # Hugging Face Hub on first use and caches it.
   turbulens infer --input observation.fits --channel 0 --real-data \
       --model auto --out predictions.ecsv

   # Predict from a velocity-integrated range with the dedicated k_min
   # ensemble, pinned to a specific published revision
   turbulens infer --input observation.fits --velocity -20 20 --real-data \
       --model auto --target k_min --revision <tag-or-commit>

   # Pre-download every ensemble, e.g. before going offline
   turbulens download --all

   # Use an ensemble you trained yourself, or one member directory directly
   turbulens infer --input observation.fits --channel 0 --real-data \
       --model auto --output-root /path/to/pipeline/results
   turbulens infer --input observation.fits --channel 0 --real-data \
       --model /path/to/ensemble/members

   # Predict from a synthetic pyFC image (already on the training log-density
   # scale): no --real-data here.
   turbulens infer --input synthetic_slice.npy --model auto \
       --output-root /path/to/pipeline/results

   # Same options from a JSON file; an explicit flag still overrides
   # the matching JSON key (e.g. --out here)
   turbulens infer --config run.json --out different.ecsv
