# Migration from Slurm

The local package replaces scheduler submission/dependency management with direct process orchestration.

Removed scheduler-specific behavior includes:

- `sbatch`, `squeue`, and `scancel` submission logic;
- partitions/accounts/GRES requests;
- Slurm dependency chains;
- Slurm-specific environment wrappers.

Retained scientific behavior includes:

- dataset handling and normalization;
- model definitions;
- hyperparameter enumeration;
- training and validation;
- candidate/final selection;
- deep ensemble evaluation;
- interpretability;
- result manifests and checkpoints.

The canonical local experiment version in this release is `local_v2`.

The launcher resolves Python from, in order:

1. `PIPELINE_PYTHON` if explicitly set;
2. `CONDA_ENV/bin/python` if `CONDA_ENV` is explicitly set;
3. active `python`;
4. active `python3`.

It does not source a system `conda.sh`, avoiding the non-interactive `PS1`/`set -u` issue encountered in earlier scheduler wrappers.
