.. docs/turbulens/training_pipeline/outputs.rst

.. _pipeline-outputs:

Pipeline Outputs
================

Each experiment writes everything under one run root,
``<output_root>/<multitask|single_<target>>/<version>/``:

.. code-block:: text

   <run root>/
   ├── resolved_config.yaml / .json     full resolved configuration
   ├── resolved_config_hash.txt         SHA-256 of the scientific settings
   ├── environment.json                 software and hardware environment
   ├── source_provenance.json
   ├── pipeline_state/                  one completion marker per stage
   ├── logs/                            launcher and per-stage logs
   ├── data_validation/                 small_manifest.json, large_manifest.json
   ├── grid_search/
   │   ├── experiments/<run id>/        one directory per grid run
   │   ├── experiment_status.csv
   │   └── selection/                   ranked runs and configurations,
   │                                    selected_configurations.json
   ├── candidate_retraining/
   │   ├── experiments/<run id>/
   │   └── selection/
   ├── ensemble/
   │   ├── final_configuration.json
   │   ├── members/member_<NN>_seed<seed>/
   │   ├── evaluation/                  FINAL_ENSEMBLE_EVALUATION.json,
   │   │                                predictions, metrics, plots
   │   └── interpretability/<layer>/    PCA, CCA and UMAP outputs
   ├── hyperparameter_analysis/         only after the analysis script runs
   └── report/                          pipeline_report.md/.json,
                                        stage_summary.csv, storage_summary.csv

What each run keeps
-------------------

Grid and candidate runs are kept compact: the requested and resolved
configuration, the data manifest, the split indices, the normalization
statistics, the training history, validation metrics, a summary,
``COMPLETED.json`` and logs. Large prediction arrays, plots and
checkpoints are not kept for finished screening runs; an unfinished run
keeps the checkpoint it needs to resume.

Final ensemble members keep everything needed for evaluation and
inference:

.. code-block:: text

   member_01_seed42/
   ├── configuration.json            full resolved configuration
   ├── COMPLETED.json                written last; marks the member usable
   ├── checkpoints/best_checkpoint.pt
   │                                 weights from the best validation epoch,
   │                                 with the normalization statistics
   ├── run_config.requested.json     configuration as requested by the stage
   ├── normalization.json
   ├── data_manifest.json, split_indices.npz
   ├── environment.json
   ├── history/                      per-epoch losses and learning rates
   ├── metrics/, predictions/        validation and test results
   ├── summary.json, summary.csv
   ├── plots/
   └── training.log, launcher.log

Relation to the published models
--------------------------------

`turbulens` needs only three files per member: ``configuration.json``,
``COMPLETED.json`` and ``checkpoints/best_checkpoint.pt``
(`turbulens.models.checkpoint.load_member`). The Hugging Face model
repository ``Davas1410/turbulens-ensembles`` contains exactly those files,
under the same ``<mode>/<version>/ensemble/members/`` layout, for the
multitask ensemble and the four single-target ensembles. A downloaded
snapshot is therefore itself a valid output root
(:doc:`/user_guide/pretrained_models`).
