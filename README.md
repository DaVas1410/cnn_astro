# CNN Astro - Fractal Cloud Parameter Prediction

Deep learning research repository for predicting turbulence-related fractal parameters from synthetic cloud images.

This repo documents an end-to-end workflow you have built over time:

- synthetic data generation with pyFC,
- scalable local/HPC dataset production,
- multiple model families for k_min and k_max prediction,
- iterative notebook-based experiments,
- output organization for comparisons and reproducibility.

## Project Status (Snapshot)

Current state of the repository reflects active research and engineering work in parallel tracks:

- Core generator pipeline is implemented and tested.
- Flexible per-image random parameter generation is implemented.
- Two modular model packages exist:
  - `fractal_classifier_01` (classification-first pipeline, extended analysis tooling).
  - `fractal_classifier_02` (lightweight regression-first pipeline).
- Multiple experiment notebooks are organized by task (`kmin`, `kmax`, `comparison`, `sigma`, `data_generation`).
- Output directories contain training artifacts for several model variants and comparisons.
- Some folders are placeholders for future consolidation (`docs/`, `experiments/kmin`, `experiments/kmax`, `experiments/dual_output`).

## What Has Been Built So Far

## 1. Synthetic Dataset Generation Framework

### `src/dataset_generator.py`

You implemented a configurable `DatasetGen` pipeline around pyFC with:

- INI-driven configuration loading (`configs/dataset_config.ini`, `configs/test_config.ini`),
- safe expression parsing for numeric config fields (for values like `np.sqrt(5.)`),
- support for paired `k_values` and `k_max_values` (including `auto`/Nyquist mode),
- support for parameter combinations (`k_min`, `k_max`, `beta`, `mean`, `sigma`),
- multiprocessing worker execution for image generation,
- HDF5 dataset creation and metadata storage,
- validation and memory-management hooks,
- helper functionality for merging and summarizing generated datasets.

### Performance and Scalability Work

- Worker compatibility wrapper handles pyFC API variants for seeding (`random_seed`, `seed`, or fallback behavior).
- SLURM-ready generation scripts are integrated.
- Performance JSON outputs and plotting support exist via `src/plot_performance.py`.

## 2. Evolution of Data Generation Strategy

You now have a clear evolution path from fixed/discrete generation to fully flexible random generation:

- `scripts/run_generation.py`:
  - job-array style generation across `k_min` values,
  - per-resolution generation (`128x128`, `256x256`, `512x512`),
  - grouped merge of per-dimension files.

- Legacy random scripts:
  - `scripts/run_generation_random_kmin.py` (random `k_min`, `k_max=auto`),
  - `scripts/run_generation_random_kmax.py` (fixed `k_min`, random `k_max`),
  - `scripts/run_generation_random_both.py` (random `k_min` and `k_max`, with optional balanced `k_max` labels).

- New flexible generator:
  - `scripts/run_generation_flexible.py` + `src/param_sampler.py`,
  - per-image random sampling for `k_min`, `k_max`, `sigma`, and `mean`,
  - parameter constraints encoded (Nyquist limit, ordering, positivity),
  - flat HDF5 layout with per-image parameter arrays.

This progression is a major repo milestone: from scripted/discrete sweeps to general continuous-space sampling.

## 3. Model Development Tracks

### `src/fractal_classifier_01` (v0.1)

Classification-oriented framework (with centroid-style interpolation logic) featuring:

- modular package structure (`config`, `data`, `models`, `training`, `evaluation`, `visualization`, `scripts`),
- YAML-driven configuration,
- CLI workflows for training/evaluation/analysis,
- GPU/mixed precision support,
- rich visualization support.

Major enhancement work captured in `ENHANCEMENTS.md` and `KERNEL_ANALYSIS_GUIDE.md` includes:

- kernel export callbacks,
- checkpoint-wise kernel statistics,
- convergence trajectory analysis,
- reference kernel comparison,
- TensorBoard integration,
- expanded regularization/optimizer/schedule options.

### `src/fractal_classifier_02` (v0.2)

Regression-oriented lightweight framework focused on `k_min` prediction:

- depthwise-separable CNN (~180K params) versus larger v0.1 model,
- regression head with Huber loss,
- evaluation utilities for MAE/RMSE/R²,
- CLI scripts for train/evaluate,
- explicit support for multi-resolution dataset organization.

Together, v0.1 and v0.2 represent two complementary lines of work:

- robust classification + analysis ecosystem,
- efficient regression architecture for continuous targets.

## 4. Notebook Research Portfolio

You have built an extensive notebook suite for experiments and iteration:

- Top-level investigations:
  - `notebooks/flexible_dataset_generation.ipynb`
  - `notebooks/multi_parameter_regression.ipynb`
  - `notebooks/sigma_regression.ipynb`
  - `notebooks/sigma_research.ipynb`

- Data generation notebooks:
  - `notebooks/data_generation/dataset_gen.ipynb`
  - `notebooks/data_generation/kmax_dataset.ipynb`
  - `notebooks/data_generation/k_mas_dataset.ipynb`
  - `notebooks/data_generation/random_kmin_dataset_local.ipynb`

- k_min experiments:
  - `notebooks/kmin_experiments/fractal_classifier_v02.ipynb`
  - `notebooks/kmin_experiments/kmin_regression.ipynb`
  - `notebooks/kmin_experiments/pytorch_resnet.ipynb`
  - `notebooks/kmin_experiments/resnet_nopatchs.ipynb`
  - `notebooks/kmin_experiments/resnet_nopatchs _discrete.ipynb`

- k_max experiments:
  - `notebooks/kmax_experiments/kmax_binned_classification.ipynb`
  - `notebooks/kmax_experiments/kmax_regression.ipynb`
  - `notebooks/kmax_experiments/kmax_regression_flexible.ipynb`
  - `notebooks/kmax_experiments/resnet_kmax.ipynb`

- Comparison notebooks:
  - `notebooks/comparison/dual_output_regression.ipynb`
  - `notebooks/comparison/model_comparison.ipynb`

This notebook structure shows broad experimentation across:

- single-parameter tasks (`k_min`, `k_max`),
- multi-parameter regression,
- model-family comparisons,
- data-generation method comparisons.

## 5. Experiment Outputs and Artifacts

The `outputs/` tree is already organized by research line and model family:

- `outputs/kmin/`:
  - `pytorch_resnet/`, `regression/`, `resnet50_patches/`, `v1/`, `v02/`, `v03/`
- `outputs/kmax/`:
  - `binned_classification/`, `regression/`
- `outputs/comparison/`:
  - `dual_output/`, `figures/`
- `outputs/sigma/regression/`
- `outputs/pytorch_resnet/kmax_classification/`
- additional output files including `performance_report.png` and `output_data_cuda_small.npy`.

This indicates you have run and stored results for multiple independent tracks rather than a single linear experiment.

## 6. Testing and Reliability Work

Testing support currently centers around dataset generation reliability:

- `tests/test_dataset_generator.py` includes broad unit coverage for:
  - config parsing,
  - parameter combinations,
  - worker behavior,
  - HDF5 operations,
  - dataset population,
  - utility/memory and merge behaviors.

- Test utilities:
  - `tests/run_tests.py` (suite/class-level runner),
  - `tests/check_dependencies.py` (environment sanity check),
  - `tests/fix_tests.py` (maintenance helper).

## 7. HPC and Operational Integration

SLURM integration is established for both legacy and flexible generation:

- `scripts/job.sh`: array-based multi-task generation (legacy/discrete workflow).
- `scripts/job_flexible.sh`: single-job flexible workflow with environment-variable overrides.

The scripts include environment setup, logging paths, resource directives, and runtime configuration echoing.

## 8. Environment and Dependency Management

- `environment.yml` defines a Conda environment (`py39`) with core scientific stack.
- `requirements_py311.txt` provides a Python 3.11 pip stack.
- `requirements_py39.txt.txt` provides a Python 3.9 pip stack.

Note: the Python 3.9 requirements filename currently includes a double extension (`.txt.txt`).

## 9. Data Footprint in Repository

Current repository includes generated dataset files under `data/raw/`, including examples from:

- flexible generation runs,
- random `k_min` runs,
- random `k_min` + `k_max` runs (including balanced `k_max` variant).

These files serve as concrete evidence of completed generation workflows and parameter-sampling experiments.

## 10. Known Metrics and Outcomes (So Far)

Based on your tracked notes/results in this repository:

| Area | Status |
|---|---|
| k_min classification (ResNet-style) | ~99% accuracy reported |
| k_max high-class-count classification baseline | ~5% accuracy reported |
| k_max regression | active experimentation (ongoing) |
| dual-output regression | active experimentation (ongoing) |

## Quick Start

### 1. Environment setup

```bash
# Conda
conda env create -f environment.yml
conda activate py39

# Or pip (Python 3.11 environment)
pip install -r requirements_py311.txt
```

### 2. Generate data (flexible workflow)

```bash
# Example local run
NUM_IMAGES=1000 KMIN_LOW=1 KMIN_HIGH=32 KMAX_MODE=auto SIGMA_LOW=2.0 SIGMA_HIGH=2.5 python scripts/run_generation_flexible.py
```

### 3. Generate data on SLURM

```bash
sbatch scripts/job_flexible.sh
```

### 4. Run dataset generator tests

```bash
python tests/run_tests.py -v
```

## Compact Repository Map

```text
cnn_astro/
├── src/
│   ├── dataset_generator.py
│   ├── param_sampler.py
│   ├── plot_performance.py
│   ├── fractal_classifier_01/
│   ├── fractal_classifier_02/
│   └── pyFC_lib/
├── scripts/
├── notebooks/
├── data/
├── outputs/
├── configs/
├── tests/
├── docs/                  # currently empty
└── experiments/           # archive + placeholders for organization
```

## Citation

```bibtex
@thesis{fractal_cnn_2026,
  title={Deep Learning for Fractal Cloud Parameter Prediction},
  author={J.D.V.V},
  year={2026}
}
```

## License

MIT License
