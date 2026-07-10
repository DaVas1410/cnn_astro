# Local run (single Linux PC + GPU, no SLURM)

Train the 5-member uncertainty ensemble and evaluate it on **one machine**
(a Linux workstation/server with an NVIDIA GPU such as an A100). Same model,
same config as `scripts/hpc_ensemble/` — just run sequentially, no scheduler.

**Three commands total:** `setup.sh` (once) → edit two lines in the config →
`run.sh`. Details below.

---

## Requirements

- **Linux** with an **NVIDIA GPU** and drivers installed (`nvidia-smi` works).
  These scripts are bash + a Python venv; they will not run on Windows/macOS.
- **Python 3.9–3.11** (`python3 --version`).
- The **dataset file** (~5.7 GB, see step 0). It is *not* included in the code
  bundle — it must be copied over separately.
- Free disk: ~7 GB for the dataset + a few hundred MB for outputs.

Expect roughly a few hours on a single A100 for all 5 members at the default
80 epochs (scales with GPU and `training.epochs`).

---

## 0. Get the code and the dataset in place

Put the whole `cnn_astro` project somewhere you can write to, then place the
dataset under `data/raw/`:

```text
cnn_astro/
├── data/
│   └── raw/
│       └── balanced_4param_128x128_100000_kmaxfix.h5   ← the 5.7 GB dataset
├── scripts/
│   ├── hpc_ensemble/     ← model + config live here
│   └── local_run/        ← you are here
└── outputs/              ← results land here (created automatically)
```

If you received the dataset under a different name or path, that's fine — just
point the config to it in step 2.

Run every command below **from the project root** (`cnn_astro/`).

---

## 1. Install dependencies (once)

```bash
bash scripts/local_run/setup.sh
```

This creates a virtual environment at `.venv/` and installs everything from
`scripts/hpc_ensemble/requirements.txt` (PyTorch, h5py, numpy, …). It finishes
by printing whether CUDA is available and the GPU name — **check that line says
`CUDA available: True`** before continuing.

The default PyPI `torch` wheels are CUDA builds and work on an A100. If you need
a specific CUDA version, install torch first, then run setup:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
bash scripts/local_run/setup.sh
```

---

## 2. Configure (edit two lines)

Open `scripts/hpc_ensemble/config.yaml` and set these **two** fields:

```yaml
data:
  file: "data/raw/balanced_4param_128x128_100000_kmaxfix.h5"   # ← the dataset from step 0

output:
  dir: "outputs/hpc_ensemble"   # ← where results are written (relative = under the project)
```

- `data.file` — must point to the dataset you placed in step 0.
- `output.dir` — use the relative path above and results land in
  `outputs/hpc_ensemble/` inside the project. (An absolute path also works if
  you want results elsewhere, e.g. `/scratch/<you>/cnn_out`.)

**Everything else can stay at its defaults.** In particular:

- The `hpc:` block (partition, conda_env, `project_dir`, …) is only used by the
  SLURM path — **ignore it here.** `run.sh` fills in `project_dir` for this
  machine automatically (it writes a throwaway `logs/config_local.yaml` and
  never touches your `config.yaml`).
- The `data:` normalization constants (`img_p1`, `kmin_hi`, …) are tuned for the
  dataset above — leave them as-is unless you know you're using a different one.

Optional knobs: `training.epochs`, `batch_size`, `lr`, and
`ensemble.n_members` (default 5).

---

## 3. Train + evaluate

```bash
bash scripts/local_run/run.sh
```

This trains each ensemble member in turn, then evaluates the combined ensemble.
It prints the project path, config, member count, and the GPU it will use before
starting. You can safely leave it running.

Optional:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/local_run/run.sh    # pick which GPU to use
CONFIG=/path/to/my_config.yaml bash scripts/local_run/run.sh   # use a different config
```

---

## Outputs

All under `outputs/hpc_ensemble/` (or wherever you pointed `output.dir`):

- `member_{0..4}/best_model.pt`, `history.json` — per-member weights + training curves
- `results.json`, `results_summary.txt` — ensemble metrics (R², errors per parameter)
- `scatter.png`, `residuals.png`, `uncertainty_calibration.png` — diagnostic plots
- Per-step logs: `logs/member_*.log` and `logs/evaluate.log`

Start with `results_summary.txt` and the three PNGs for a quick read of how the
ensemble did.

---

## Troubleshooting

- **`venv not found … run setup.sh first`** — you skipped step 1, or you're not
  in the project root. `cd` into `cnn_astro/` and run `bash scripts/local_run/setup.sh`.
- **`CUDA available: False` / `CPU (no CUDA!)`** — the GPU/driver isn't visible.
  Check `nvidia-smi`, and reinstall torch with the matching CUDA index URL (step 1).
- **`No such file … .h5` / dataset errors** — `data.file` in the config doesn't
  match where the dataset actually is (step 0 / step 2).
- **Out of GPU memory** — lower `training.batch_size` in the config (e.g. 32 or 16).
