.. docs/turbulens/index.rst

turbulens
=========

`turbulens` predicts fractal turbulence parameters (`k_min`, `k_max`,
`sigma`, and `beta` where available) from radio-astronomy FITS cubes
or plain images, using a pretrained deep-ensemble CNN. See
:doc:`concepts` for the physical background, :doc:`cli` for
command-line usage, and :doc:`api/index` for the full API reference.

Installation
------------

.. code-block:: bash

   pip install -e .[image]  # from a checkout of this repository

.. _getting-an-ensemble:

Getting an ensemble
-------------------

**No model checkpoints ship with** `turbulens`. The package is inference
code only; every command that predicts requires you to point ``--model``
at an ensemble you already have.

Ensembles are produced by this repository's separate training pipeline,
``turbulens/training``, which writes each run under::

   <output_root>/<multitask|single_<target>>/<version>/ensemble/members/
       member_00/
           COMPLETED.json
           ...
       member_01/
           COMPLETED.json
           ...

`turbulens.models.registry.EnsembleRegistry` is the only place this
layout is encoded. ``EnsembleRegistry.resolve`` walks from an
``output_root`` down to the ``ensemble/members`` directory for a given
mode (joint ``multitask``, or ``single_<target>`` for a single-target
run) and version (a directory name, or ``"latest"`` to pick the most
recently modified one). ``EnsembleRegistry.discover_members`` then lists
the member subdirectories, counting only those with a ``COMPLETED.json``
marker so that a half-written member from a crashed training job is
never loaded.

The CLI's ``--model`` argument takes the ``ensemble/members`` directory
itself, or the literal value ``auto``, in which case ``--output-root``
(and ``--target`` for a single-target run) drive the resolution
described above. `turbulens.inference.Inferencer` requires at least two
completed members, since it reports member-to-member spread as its
epistemic uncertainty.

Quickstart
----------

.. code-block:: bash

   # Inspect an input before running inference
   turbulens inspect --input observation.fits

   # Predict physical targets and write results to an ECSV table.
   # --model points at an ensemble/members directory you supply; see
   # "Getting an ensemble" above. --real-data is required for a real
   # telescope observation like this one -- see :doc:`cli` for why.
   turbulens infer --input observation.fits --channel 0 --real-data \
       --model /path/to/ensemble/members --out predictions.ecsv

See :doc:`cli` for every available flag, and the `Cube`, `Inferencer`,
and `EnsembleRegistry` API pages under :doc:`api/index` for using
`turbulens` as a library rather than a CLI.

.. toctree::
   :maxdepth: 2
   :hidden:

   concepts
   cli
   api/index
   references
