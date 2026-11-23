# Training pipeline

Trains the deep ensembles that `turbulens infer --method cnn` uses: one multitask model for
`k_min`, `k_max`, `sigma` and `beta`, or one single-target model per parameter.

Each run goes grid search → candidate retraining → best configuration → 5-member ensemble →
test evaluation → report. Model selection uses validation data only; the test set is used only
for the final report.

## Install

```bash
uv sync --extra training
```

## Data

One HDF5 file per split (train / validation / test), each with `/images` and
`/parameters/{k_min,k_max,sigma,beta}`. Set their location in `configs/config_base.yaml`.

## Run

From `turbulens/training/`:

```bash
python -u check_environment.py --config configs/config_multitask.yaml   # check setup
bash run_pipeline.sh configs/config_multitask.yaml                       # run (rerun to resume)
bash status_pipeline.sh configs/config_multitask.yaml                    # progress
bash stop_pipeline.sh configs/config_multitask.yaml                      # stop safely
```

Single-target models use `configs/config_single_<target>.yaml`. To run everything (multitask,
the four single-target models and their comparison) in one go:

```bash
bash run_all_experiments.sh
```

Results go to `<output_root>/<multitask|single_<target>>/<version>/`. To change a setting,
also change `experiment.version` in the config: a run refuses to continue with a different
configuration under the same version.

## More

Full details (configuration, search grid, outputs, scientific safeguards) are in the docs:
[`docs/turbulens/training_pipeline/`](../../docs/turbulens/training_pipeline/index.rst).
