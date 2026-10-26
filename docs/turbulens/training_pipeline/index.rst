.. docs/turbulens/training_pipeline/index.rst

Training Pipeline
=================

How the published ensembles were produced, and how to train your own.
You do not need any of this to *use* `turbulens`; it is here so the
models can be checked and reproduced.

- :doc:`overview`: the nine stages, the two data scales and the flow of
  information between splits.
- :doc:`configuration`: the YAML configuration and experiment versions.
- :doc:`hyperparameter_search`: the search space, why it has that shape,
  and how the final configuration is selected.
- :doc:`scientific_invariants`: leakage guards, determinism and the
  release checklist.
- :doc:`running`: commands to run, monitor, stop and resume.
- :doc:`outputs`: the result tree and what the published models contain.

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   configuration
   hyperparameter_search
   scientific_invariants
   running
   outputs
