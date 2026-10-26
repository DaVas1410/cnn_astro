.. docs/turbulens/training_pipeline/running.rst

Running the Pipeline
====================

All commands run from ``turbulens/training/`` on a Linux machine with
one or more CUDA GPUs. Install the training dependencies first, from the
repository root: ``pip install -e ".[training]"``.

1. Preflight
------------

.. code-block:: bash

   python -u check_environment.py --config configs/config_multitask.yaml

This checks packages, paths, data and GPU visibility, and prints the
planned run counts: 108 grid runs, 24 candidates, 48 candidate runs and
5 ensemble members for the canonical configuration.

2. Dry run
----------

.. code-block:: bash

   bash run_pipeline.sh configs/config_multitask.yaml --dry-run --print-plan

Prints the stages and runs that would execute, without training
anything.

3. Search, then review
----------------------

Running up to the final selection first leaves room to review the
search before spending GPU time on the ensemble:

.. code-block:: bash

   nohup bash run_pipeline.sh configs/config_multitask.yaml \
       --to-stage select_final > multitask_search.log 2>&1 &

   python -u analyze_hyperparameter_importance.py \
       --run-root <output_root>/multitask/local_v2 --overwrite

The analysis writes ``hyperparameter_analysis/ANALYSIS_REPORT.md`` and
the supporting tables (:doc:`hyperparameter_search`).

4. Final ensemble
-----------------

.. code-block:: bash

   nohup bash run_pipeline.sh configs/config_multitask.yaml \
       --from-stage train_ensemble > multitask_final.log 2>&1 &

This continues the same version with the same configuration hash.
``start_pipeline.sh`` is a shorthand that launches ``run_pipeline.sh``
in the background with ``nohup``.

Monitoring, stopping and resuming
---------------------------------

.. code-block:: bash

   bash status_pipeline.sh configs/config_multitask.yaml
   bash stop_pipeline.sh configs/config_multitask.yaml

``stop_pipeline.sh`` sends ``SIGTERM`` so runs can clean up and
checkpoint; avoid ``kill -9``. To resume, run the same command again:
completed stages and runs are skipped, and an interrupted run continues
from its latest checkpoint. The launcher log is
``<run root>/logs/pipeline_launcher.log``.

Useful options of ``run_pipeline.sh`` (passed on to ``pipeline.py``):
``--stages a,b`` runs only the named stages, ``--from-stage`` and
``--to-stage`` bound the run, ``--force-stage`` reruns a completed stage,
and ``--no-auto-dependencies`` stops missing prerequisites from being
added automatically.

The full study
--------------

.. code-block:: bash

   bash run_pipeline.sh configs/config_single_k_min.yaml   # one target
   nohup bash run_all_experiments.sh > all_experiments.log 2>&1 &

``run_all_experiments.sh`` runs the multitask experiment (skipped if
already complete), the four single-target experiments, then the
comparison. Its own logs go under ``<output_root>/sequence_logs/``. The
comparison alone is

.. code-block:: bash

   python -u compare_multitask_single.py --config configs/comparison.yaml --overwrite

Environment variables
---------------------

``PIPELINE_GPUS``
   Comma-separated GPU indices to use, overriding the YAML, for example
   ``PIPELINE_GPUS="1"``. Prefer this to setting ``CUDA_VISIBLE_DEVICES``
   yourself: the pipeline assigns one visible GPU to each training
   process.

``PIPELINE_PYTHON`` / ``CONDA_ENV``
   The Python interpreter to use. ``run_pipeline.sh`` does not activate
   environments itself, so it is safe under ``nohup``; it uses
   ``PIPELINE_PYTHON``, then ``$CONDA_ENV/bin/python``, then ``python`` on
   ``PATH``.

The GPU guard waits until the configured GPUs are free before starting
each training process. It never changes the batch size or any other
scientific setting.
