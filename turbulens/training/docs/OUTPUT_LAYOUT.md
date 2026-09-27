# Output Layout

For the canonical multitask `local_v2` run:

```text
<output_root>/multitask/local_v2/
├── resolved_config.yaml
├── resolved_config.json
├── resolved_config_hash.txt
├── environment.json
├── source_provenance.json
├── pipeline_state/
│   ├── pipeline.json
│   ├── validate_data.json
│   ├── grid_search.json
│   ├── select_candidates.json
│   ├── retrain_candidates.json
│   ├── select_final.json
│   ├── train_ensemble.json
│   ├── evaluate_ensemble.json
│   ├── interpretability.json
│   └── report.json
├── logs/
├── data_validation/
│   ├── small_manifest.json
│   └── large_manifest.json
├── grid_search/
│   ├── experiments/
│   ├── experiment_status.csv
│   ├── selection_settings.json
│   ├── selection/
│   │   ├── ranked_runs.csv
│   │   ├── ranked_configurations.csv
│   │   ├── top_configurations.csv
│   │   ├── selected_configurations.json
│   │   └── COMPLETED.json
│   └── COMPLETED.json
├── candidate_retraining/
│   ├── experiments/
│   ├── experiment_status.csv
│   ├── selection/
│   │   ├── ranked_configurations.csv
│   │   ├── top_configurations.csv
│   │   ├── selected_configurations.json
│   │   └── COMPLETED.json
│   └── COMPLETED.json
├── ensemble/
│   ├── final_configuration.json
│   ├── members/
│   ├── evaluation/
│   └── interpretability/
├── hyperparameter_analysis/        # created only when analysis script is run
└── report/
    ├── pipeline_report.md
    ├── pipeline_report.json
    ├── stage_summary.csv
    └── storage_summary.csv
```

## Screening run retention

A completed grid/candidate run keeps compact reproducibility and validation information such as:

```text
configuration.json
run_config.requested.json
data_manifest.json
split_indices.npz
normalization.json
history/history.json
history/history.csv
metrics/validation_metrics.json
summary.json
summary.csv
COMPLETED.json
training.log
launcher.log
```

By default successful screening runs do not retain large prediction arrays, plots, or model checkpoints. An incomplete run retains the checkpoint required for resume.

## Final ensemble retention

Final ensemble members retain inference checkpoints, predictions, plots, environment metadata, and histories needed for publication-quality evaluation and reproducibility.

## Sequence logs

`run_all_experiments.sh` writes orchestration logs under:

```text
<output_root>/sequence_logs/local_v2/
```

rather than modifying the source repository.
