# Runbook

This is the operational guide for a clean `local_v2` study.

## 1. Extract the release

```bash
cd /path/to/project/code
tar -xzf fractal-regression-pipeline-v2.0.0.tar.gz
cd fractal-regression-pipeline-v2.0.0
```

The default relative-path configuration assumes this repository is one directory below `<project>/code/`.

## 2. Activate Python

```bash
conda activate /path/to/environment
```

Check:

```bash
which python
python --version
```

Python 3.10 or newer is required. Keep the release archive and its SHA-256 checksum with the final study provenance.

## 3. Preflight

```bash
python -u check_environment.py --config configs/config_multitask.yaml
```

Expected key counts:

```text
Grid runs        : 108
Grid candidates  : 24
Candidate runs   : 48
Ensemble members : 5
```

## 4. Dry run

```bash
bash run_pipeline.sh configs/config_multitask.yaml --dry-run --print-plan
```

No model training occurs.

## 5. Recommended first production phase

Run through final hyperparameter selection, but stop before the final ensemble:

```bash
nohup bash run_pipeline.sh configs/config_multitask.yaml \
  --to-stage select_final \
  > multitask_local_v2_search.log 2>&1 &
```

Record the shell PID if desired:

```bash
echo $!
```

Monitor:

```bash
tail -f multitask_local_v2_search.log
```

The canonical launcher log is also written under:

```text
<project>/turbulens/training/outputs/multitask/local_v2/logs/pipeline_launcher.log
```

## 6. Status and safe stop

```bash
bash status_pipeline.sh configs/config_multitask.yaml
```

Safe stop:

```bash
bash stop_pipeline.sh configs/config_multitask.yaml
```

Do not use `kill -9` unless graceful termination fails, because `SIGKILL` prevents checkpoint cleanup.

## 7. Resume after interruption

Run the same command again:

```bash
bash run_pipeline.sh configs/config_multitask.yaml --to-stage select_final
```

Completed stages/runs are skipped. An incomplete training run resumes from its latest compatible checkpoint.

## 8. Analyze the refined search

After `select_final` completes:

```bash
python -u analyze_hyperparameter_importance.py \
  --run-root /path/to/project/turbulens/training/outputs/multitask/local_v2 \
  --overwrite
```

Inspect first:

```bash
less /path/to/project/turbulens/training/outputs/multitask/local_v2/hyperparameter_analysis/ANALYSIS_REPORT.md
```

Then review:

```text
grid_search/parameter_level_summary.csv
grid_search/marginal_importance_summary.csv
grid_search/rf_permutation_importance.csv
grid_search/interaction_importance_summary.csv
grid_search/conditional_best_regime/parameter_level_summary.csv
cross_stage/grid_candidate_comparison.csv
cross_stage/grid_candidate_correlations.json
```

## 9. Continue final multitask ensemble

If the refined search is accepted:

```bash
nohup bash run_pipeline.sh configs/config_multitask.yaml \
  --from-stage train_ensemble \
  > multitask_local_v2_final.log 2>&1 &
```

This continues the same `local_v2` configuration and therefore does not create a configuration-hash conflict.

## 10. Run single-target experiments

Individually:

```bash
bash run_pipeline.sh configs/config_single_k_min.yaml
bash run_pipeline.sh configs/config_single_k_max.yaml
bash run_pipeline.sh configs/config_single_sigma.yaml
bash run_pipeline.sh configs/config_single_beta.yaml
```

Or run the complete sequential suite in the background:

```bash
nohup bash run_all_experiments.sh > all_experiments.log 2>&1 &
```

The suite first invokes multitask. If multitask is already complete it is skipped safely, then the four single-target pipelines are executed, followed by the final comparison.

## 11. Monitor GPU use

```bash
watch -n 5 nvidia-smi
```

The pipeline's GPU availability guard waits for configured GPUs to be stably free before launching a training subprocess. It does not change batch size or other scientific parameters.

## 12. Restrict to one GPU temporarily

```bash
PIPELINE_GPUS="1" \
bash run_pipeline.sh configs/config_multitask.yaml --to-stage select_final
```

`PIPELINE_GPUS` should be preferred to manually setting a conflicting `CUDA_VISIBLE_DEVICES` because the pipeline explicitly assigns one visible GPU per training subprocess.

## 13. Use a specific Python executable

```bash
PIPELINE_PYTHON=/path/to/env/bin/python \
bash run_pipeline.sh configs/config_multitask.yaml --to-stage select_final
```

## 14. Do not overwrite an experiment version with changed science

If you change any scientific parameter, change:

```yaml
experiment:
  version: local_v3
```

Do not set `allow_config_change: true` merely to bypass a hash mismatch for a scientific rerun.

## 15. Final comparison only

When all five final ensembles exist:

```bash
python -u compare_multitask_single.py \
  --config configs/comparison.yaml \
  --overwrite
```

The default comparison configuration points to the `local_v2` result roots.

## 16. Manuscript-quality reproducibility QC

Before freezing results for publication, complete [`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md). It covers environment/data validation, test isolation, search QC, final ensemble checks, paired multitask-vs-single-target statistics, and archival provenance.
