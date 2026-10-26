.. docs/turbulens/user_guide/pretrained_models.rst

.. _getting-an-ensemble:

Pretrained Models
=================

The published ensembles
-----------------------

The pretrained deep ensembles are published in the Hugging Face model
repository ``Davas1410/turbulens-ensembles``. They are not bundled with
the package. It holds five ensembles of five members each:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Mode
     - Selected with
     - Predicts
   * - ``multitask``
     - ``--model auto``
     - all four targets from one network
   * - ``single_k_min``
     - ``--model auto --target k_min``
     - :math:`k_\mathrm{min}` only
   * - ``single_k_max``
     - ``--model auto --target k_max``
     - :math:`k_\mathrm{max}` only
   * - ``single_sigma``
     - ``--model auto --target sigma``
     - :math:`\sigma` only
   * - ``single_beta``
     - ``--model auto --target beta``
     - :math:`\beta` only

All were trained by the same pipeline on the same data
(:doc:`/training_pipeline/overview`). Each single-target ensemble went
through its own hyperparameter search.

At the time of writing the repository is private. Downloading needs a
Hugging Face account with access, logged in with ``hf auth login``.

Download on demand
------------------

``turbulens infer --model auto`` without ``--output-root`` downloads
**only the ensemble the command needs**, about 450 MB, the first time it
is needed. It goes into the standard Hugging Face cache (``HF_HOME``, by
default ``~/.cache/huggingface``) and is reused on later runs. Files
already in the cache are never downloaded again.

To download ahead of time, for example before working on a machine
without internet access:

.. code-block:: bash

   turbulens download                  # the multitask ensemble
   turbulens download --target sigma   # one single-target ensemble
   turbulens download --all            # all five

Each line printed gives the local ``ensemble/members`` directory and the
commit it came from. Cached ensembles work offline; set
``HF_HUB_OFFLINE=1`` to skip the network check entirely.

Pinning a version
-----------------

``--revision`` selects a branch, tag or commit of the model repository,
for ``infer`` and ``download`` alike:

.. code-block:: bash

   turbulens infer --input cube.fits --velocity -100 100 --real-data \
       --model auto --revision <tag-or-commit>

Whether or not a revision is pinned, the exact commit used is recorded
in the output table as ``model_revision``, next to ``model_repo``
(:doc:`outputs_provenance`). Any result can therefore be reproduced with
exactly the same weights.

Using your own ensembles
------------------------

Ensembles trained with the training pipeline can be used the same way:

.. code-block:: bash

   # pick the newest version under a pipeline output root
   turbulens infer --input cube.fits --velocity -100 100 --real-data \
       --model auto --output-root /path/to/pipeline/outputs

   # or point at one ensemble directly
   turbulens infer --input cube.fits --velocity -100 100 --real-data \
       --model /path/to/<mode>/<version>/ensemble/members

With ``--output-root``, ``auto`` looks in
``<output_root>/<multitask|single_<target>>/`` and takes the most
recently modified version directory (by modification time, not by
name). An explicit ``--model`` must be the ``ensemble/members``
directory itself, not the version or ``ensemble`` directory above it.

What a member must contain
--------------------------

Only members with a ``COMPLETED.json`` marker are loaded, so a member
left half-written by a crashed job is ignored. Each member needs:

- ``configuration.json``: architecture, targets and target ranges;
- ``checkpoints/best_checkpoint.pt``: the weights and the pixel
  normalization statistics;
- ``COMPLETED.json``.

At least two completed members are required, since the spread between
members is the reported uncertainty (:ref:`uncertainty`).

From Python
-----------

`turbulens.models.registry.EnsembleRegistry` does the same resolution:
``fetch`` downloads from the Hub, ``resolve`` searches a pipeline output
root, and ``discover_members`` lists the completed members of a
directory. See :doc:`python_api`.
