# Validation Report: 2.0.0-local

This report records the release checks performed on the clean `2.0.0-local` package after promoting the evidence-based `local_v2` methodology to the canonical configuration.

## Scope

The validation focuses on configuration correctness, scientific guardrails, orchestration, resumability, source portability, and a synthetic end-to-end execution. It is not a substitute for the intended production GPU rerun on the full scientific datasets.

## Configuration and scientific checks

- All five task configurations resolve through the same canonical `configs/config_base.yaml` and use `experiment.version: local_v2`.
- Multitask and each single-target task enumerate exactly **108** grid runs.
- Grid selection retains **8 configurations per architecture**, giving **24** large-data candidate configurations.
- Candidate retraining uses **2 independent seeds per configuration**, giving **48** candidate runs.
- Final ensemble size is **5** members with unique configured seeds.
- Refined learning rates are exactly `[5e-4, 1e-3, 2e-3]`.
- Refined training batch sizes are exactly `[64, 128]` and are passed as search values. There is no automatic batch-size probing, fallback, clamping, or replacement logic.
- Pretraining is fixed to `true` in `local_v2` and exact torchvision recipes are pinned: ResNet18/34 `IMAGENET1K_V1`, ResNet50 `IMAGENET1K_V2`.
- The production loss pathway is ordinary `smooth_l1` only. Historical target-specific `loss_weights` and `weighted_smooth_l1` behavior are not part of the production implementation.
- Grid and candidate `evaluate_test` settings are required to be `false` by configuration validation; final ensemble `evaluate_test` is required to be `true`.
- Integer configuration fields are validated without silent float truncation.
- Child YAML configurations can narrow the architecture list; inherited pretrained-weight entries for inactive architectures are pruned from the resolved configuration.
- Scientific Python defaults do not contain a hidden duplicate of the hyperparameter grid.

## Refined-grid evidence checks

The interaction diagnostic was exercised against the uploaded completed `local_v1` analysis table. For the validation selection score, the leading descriptive two-factor interactions were reproduced as:

```text
pretrained x learning_rate      eta^2 ~ 0.1150
architecture x learning_rate    eta^2 ~ 0.0188
batch_size x learning_rate      eta^2 ~ 0.0138
pretrained x batch_size         eta^2 ~ 0.0063
```

This confirms that the revised analyzer detects the interaction structure that motivated retaining batch sizes 64/128 rather than relying on the misleading unconditional marginal batch-size ranking.

## Automated tests

`python -m pytest -q` passes:

```text
14 passed
```

The tests cover:

- exact refined grid values/count;
- shared refined search across all single-target configurations;
- held-out-test isolation settings;
- 24-candidate / 48-run candidate design;
- explicit dropout policy propagation;
- strict batch-size integer validation;
- rejection of the removed weighted-loss mode;
- enforced grid/candidate test isolation;
- architecture narrowing under inherited YAML configuration;
- configurable model forward pass;
- validation-score ordering;
- explicit pretrained-weight resolution;
- mean-of-bounded-member ensemble aggregation;
- Holm multiple-testing adjustment behavior.

## Static checks

- All Python files compile with `python -m compileall`.
- All top-level shell launchers pass `bash -n`.
- No executable Python, shell, or YAML file contains user-specific `/home/...` or `/mnt/...` paths.
- No executable/configuration file contains the removed automatic batch-size probe controls.
- No production Python module relies on `assert` for runtime validation.
- Relative Markdown links in README/docs resolve inside the package.
- `.gitignore` excludes caches, logs, PIDs, virtual environments, and result trees.

The optional `ruff` development check is declared in `requirements-dev.txt`; it was not executed in the packaging environment because `ruff` was not installed there.

## Synthetic CPU end-to-end smoke test

A fresh synthetic HDF5 project was generated with authoritative `train`, `val`, and `test` folders for both small and large datasets. Images were 32x32 one-channel arrays and all four target datasets were created within their configured physical ranges.

A reduced smoke configuration was used only for release validation:

```text
architecture              resnet18
pretrained                false
input channels            1
grid configurations       1
candidate configurations  1
candidate repetitions     1
ensemble members          2
epochs                    1
device                    cpu
interpretability          disabled
```

The following complete stage chain executed successfully from an empty result directory:

```text
validate_data
-> grid_search
-> select_candidates
-> retrain_candidates
-> select_final
-> train_ensemble
-> evaluate_ensemble
-> report
```

The fresh run completed successfully in approximately 30 seconds in the packaging environment. It produced data manifests, selection tables, candidate outputs, final member checkpoints/predictions, ensemble metrics/plots/bootstrap summaries, stage markers, resolved configuration/provenance, and the pipeline report.

Grid/candidate summaries contained `test: null` and no test prediction evaluation, while final ensemble members produced validation and test predictions as intended.

## Resume behavior

The exact same synthetic command was run again without deleting results. All valid stages were detected and skipped:

```text
SKIP validate_data
SKIP grid_search
SKIP select_candidates
SKIP retrain_candidates
SKIP select_final
SKIP train_ensemble
SKIP evaluate_ensemble
SKIP report
```

The resumed invocation completed in effectively zero pipeline time, confirming stage-level completion-marker reuse for a matching configuration hash.

The shell launcher `run_pipeline.sh` was also exercised against the completed smoke configuration and selected the active interpreter correctly.

## Preflight validation

`check_environment.py` was run against the synthetic CPU configuration. It successfully validated imports, resolved configuration counts, both HDF5 dataset roots, sample/file counts, image layout, and dataset fingerprints, and returned `PREFLIGHT PASSED`.

## Production run addendum (2026-09-16)

The production `local_v2` GPU grid has since been executed for real: the multitask configuration and all four single-target configurations (`k_min`, `k_max`, `sigma`, `beta`), each through the full `validate_data -> ... -> evaluate_ensemble -> interpretability -> report` chain, plus the `compare_multitask_single.py` comparison stage. Results live under `outputs/<multitask|single_<target>>/local_v2/` and `outputs/comparison/local_v2/`.

Headline test-set results (multitask ensemble, `outputs/multitask/local_v2/ensemble/evaluation/metrics/test_ensemble_metrics.json`):

| target | R2 | MAE |
| --- | --- | --- |
| k_min | 0.9997 | 0.119 |
| k_max | 0.9795 | 0.602 |
| sigma | 0.9990 | 0.034 |
| beta | 0.9203 | 0.111 |

The multitask-vs-single-target comparison (`outputs/comparison/local_v2/comparison_report.md`) found single-target models statistically significantly but only marginally ahead of multitask on `k_min`, `k_max`, and `sigma`, multitask clearly ahead on `beta`, and the single-target system costing roughly 5x the total training time and 4x the parameters of one shared multitask model. See `turbulens/models/architecture.py`'s module docstring for how this is reflected in the default architecture choice.

## Release caveats

- GPU availability/wait behavior cannot be fully exercised in this CPU-only packaging environment.
- Optional UMAP execution depends on `umap-learn`; the default policy is `optional`.
- Bitwise reproducibility is not guaranteed across different CUDA/cuDNN/GPU stacks when `training.deterministic: false`; seeds and environment provenance are still recorded.

These limitations are operational rather than known scientific-logic failures. Production results should be frozen only after completing `docs/REPRODUCIBILITY_CHECKLIST.md` on the target server.

## Archive validation

The release tree was cleaned of Python/pytest cache artifacts, `MANIFEST.sha256` was regenerated for all packaged files, and every manifest entry verified successfully before archive creation. Both TAR.GZ and ZIP archives were created with a single top-level `fractal-regression-pipeline-v2.0.0/` directory. ZIP integrity testing passed. The TAR.GZ was extracted into a fresh directory, its internal SHA-256 manifest was revalidated, the 14-test pytest suite was rerun successfully from the extracted copy, and all shell launchers again passed syntax checking.
