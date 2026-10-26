.. docs/turbulens/training_pipeline/overview.rst

.. _pipeline-stages:

Pipeline Overview
=================

The ensembles that `turbulens` loads are produced by a separate,
resumable training pipeline in ``turbulens/training/``. It is not part
of the installed package: it lives in the repository, has its own
dependencies (``pip install -e ".[training]"``), and is driven by one
YAML file per experiment. A single command takes a configuration from
raw data to a finished, evaluated deep ensemble, and the same command
resumes it after an interruption.

What the pipeline does
----------------------

One experiment trains either the **multitask** model (all four targets,
one network) or a **single-target** model (one target). The full study
runs the multitask experiment and all four single-target experiments,
then compares them (:ref:`evaluation-metrics`).

Each experiment runs nine stages in order:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Stage
     - Depends on
     - What it does
   * - ``validate_data``
     - —
     - Checks every HDF5 file: required datasets, image geometry,
       finite labels within the configured ranges, non-empty splits,
       and no file used in more than one split. Writes a manifest with a
       dataset fingerprint per data scale.
   * - ``grid_search``
     - ``validate_data``
     - Trains every configuration of the hyperparameter grid once on the
       small screening dataset.
   * - ``select_candidates``
     - ``grid_search``
     - Ranks grid runs by validation score and keeps the best
       configurations per backbone.
   * - ``retrain_candidates``
     - ``select_candidates``, ``validate_data``
     - Retrains each candidate on the large dataset with two seeds.
   * - ``select_final``
     - ``retrain_candidates``
     - Picks the single best candidate configuration by its
       seed-averaged validation score.
   * - ``train_ensemble``
     - ``select_final``
     - Retrains that configuration with five different seeds: the deep
       ensemble.
   * - ``evaluate_ensemble``
     - ``train_ensemble``
     - Aggregates the members and computes test metrics, coverage and
       bootstrap intervals.
   * - ``interpretability``
     - ``train_ensemble``
     - PCA, CCA and optional UMAP of the members' embeddings.
   * - ``report``
     - ``evaluate_ensemble``
     - Writes a Markdown/JSON report and per-stage summaries.

Selection details are in :doc:`hyperparameter_search`.

Two data scales
---------------

Screening a large grid on the full dataset would be expensive, so the
grid stage uses a **small** screening dataset: 10,000 training, 2,000
validation and 5,000 test images from the same generator and parameter
ranges as the main set. Candidate retraining, the final ensemble and all
reported evaluation use the **large** dataset (100,000 / 20,000 / 50,000
images; :ref:`synthetic-data`). Only the large dataset is published.

Information flow
----------------

Every decision is made on data that the final evaluation never touches:

.. code-block:: text

   training split    ->  weights, pixel-normalization percentiles
   validation split  ->  early stopping, learning-rate schedule,
                         grid ranking, candidate ranking, final selection,
                         fitting PCA/CCA/UMAP
   test split        ->  final ensemble metrics, interpretability
                         evaluation, multitask vs single-target comparison

See :doc:`scientific_invariants` for how this is enforced.

Resumability and provenance
---------------------------

- Each stage writes a completion marker under ``pipeline_state/``, and
  each training run writes ``COMPLETED.json`` when it finishes. Rerunning
  the same command skips completed work and resumes an interrupted run
  from its latest checkpoint.
- The fully resolved configuration is saved, together with its SHA-256
  hash, when an experiment version is first run. A later run whose
  scientific settings hash differently is refused unless the version is
  changed (:doc:`configuration`). One version name therefore always means
  one scientific configuration.
- An exclusive file lock prevents two pipeline processes from writing to
  the same experiment at once.
- Data manifests, split indices, seeds, normalization statistics,
  training histories and the software environment are recorded for every
  run (:doc:`outputs`).

Entry points
------------

``pipeline.py`` is the orchestrator, launched through the shell scripts
described in :doc:`running`. Every training run, whether a grid point, a
candidate or an ensemble member, executes the same single-model script,
``train_regression.py`` (:ref:`training`). ``evaluate_ensemble.py``,
``interpret_embeddings.py``, ``compare_multitask_single.py`` and
``analyze_hyperparameter_importance.py`` are the analysis scripts.
