# Loss Function Benchmark & Sigma Model Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare MAE, RMSE, and 0.5·RMSE+0.5·MAE loss functions for predicting k_min, k_max, and sigma from 128×128 fractal cloud images, and fix a broken sigma regression model.

**Architecture:** A new generation script produces a 100k-image dataset where all three parameters vary independently. A single benchmark notebook trains ResNet50 regressors for each (target × loss) combination with skip-if-exists resume, then renders comparison plots inline. Sigma improvements (global image normalization + log-scale target) are applied in a dedicated notebook section.

**Tech Stack:** Python 3.11, PyTorch, torchvision (ResNet50), h5py, scikit-learn, matplotlib, numpy, pyFC (fractal image generation)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/run_generation_flexible_kmax.py` | Generate 100k images with varying k_min, k_max, sigma |
| Create | `notebooks/comparison/loss_benchmark.ipynb` | Full benchmark notebook (all phases) |

Outputs written at runtime (not tracked in git):
- `data/raw/flexible_kmax_{timestamp}_128x128_100000.h5`
- `outputs/comparison/loss_benchmark/{target}/{loss}/best_model.pt`
- `outputs/comparison/loss_benchmark/{target}/{loss}/results.json`

---

## Task 1: Dataset generation script

**Files:**
- Create: `scripts/run_generation_flexible_kmax.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""
Flexible Dataset Generation — varying k_min, k_max, and sigma.

Each image gets unique random parameters:
  k_min  ~ Uniform[KMIN_LOW, KMIN_HIGH]
  k_max  ~ Uniform[KMAX_LOW, KMAX_HIGH], resampled until k_max > k_min
  sigma  ~ Uniform[SIGMA_LOW, SIGMA_HIGH]
  mean   = 1.0 (fixed)
  beta   = -5/3 (fixed)

Environment variables (all optional, shown with defaults):
  IMAGE_WIDTH=128   IMAGE_HEIGHT=128   IMAGE_DEPTH=1
  NUM_IMAGES=100000   BATCH_SIZE=50   NUM_WORKERS=1
  KMIN_LOW=1.0   KMIN_HIGH=32.0
  KMAX_LOW=1.0   KMAX_HIGH=64.0
  SIGMA_LOW=0.01  SIGMA_HIGH=5.0
  OUTPUT_FILE=''   (auto-generates name if empty)
"""

import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'pyFC_lib'))

import h5py
import numpy as np

from dataset_generator import _generate_image_worker


def _env_float(name, default): return float(os.environ.get(name, str(default)))
def _env_int(name, default):   return int(os.environ.get(name, str(default)))
def _env_str(name, default):   return os.environ.get(name, default)


OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'data', 'raw')
PERF_DIR     = os.path.join(OUTPUT_DIR, 'perf')

NUM_IMAGES   = _env_int('NUM_IMAGES',   100000)
IMAGE_WIDTH  = _env_int('IMAGE_WIDTH',  128)
IMAGE_HEIGHT = _env_int('IMAGE_HEIGHT', 128)
IMAGE_DEPTH  = _env_int('IMAGE_DEPTH',  1)
BATCH_SIZE   = _env_int('BATCH_SIZE',   50)
NUM_WORKERS  = _env_int('NUM_WORKERS',  1)
if NUM_WORKERS == 1:
    NUM_WORKERS = _env_int('SLURM_CPUS_PER_TASK', 1)

KMIN_LOW   = _env_float('KMIN_LOW',   1.0)
KMIN_HIGH  = _env_float('KMIN_HIGH',  32.0)
KMAX_LOW   = _env_float('KMAX_LOW',   1.0)
KMAX_HIGH  = _env_float('KMAX_HIGH',  64.0)
SIGMA_LOW  = _env_float('SIGMA_LOW',  0.01)
SIGMA_HIGH = _env_float('SIGMA_HIGH', 5.0)
MEAN_LOW   = _env_float('MEAN_LOW',   1.0)
MEAN_HIGH  = _env_float('MEAN_HIGH',  1.0)
BETA       = _env_float('BETA',       -5.0 / 3.0)
OUTPUT_FILE = _env_str('OUTPUT_FILE', '')

COMPRESSION      = 'gzip'
COMPRESSION_OPTS = 1


def _sample_kmax_valid(k_min_val: float, low: float, high: float,
                       max_attempts: int = 50) -> float:
    """Sample k_max uniformly from [low, high] ensuring k_max > k_min."""
    for _ in range(max_attempts):
        k_max = np.random.uniform(low, high)
        if k_max > k_min_val:
            return float(k_max)
    # Fallback: place k_max just above k_min, capped at high
    return float(min(k_min_val + 1.0, high))


def _create_hdf5(output_file: str, total_images: int) -> None:
    with h5py.File(output_file, 'w') as f:
        f.create_dataset(
            'images',
            shape=(total_images, IMAGE_HEIGHT, IMAGE_WIDTH),
            maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=np.float32,
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
            chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        f['images'].attrs['description'] = 'Log10-transformed fractal density slices'
        f['images'].attrs['transform']   = 'log10(density_slice)'

        pg = f.create_group('parameters')
        pg.attrs['description'] = 'Per-image generation parameters'

        for name in ['k_min', 'k_max', 'sigma', 'mean', 'beta']:
            pg.create_dataset(name, shape=(total_images,), maxshape=(None,),
                              dtype=np.float32, compression=COMPRESSION,
                              compression_opts=COMPRESSION_OPTS, chunks=True)
        for name in ['ni', 'nj', 'nk']:
            pg.create_dataset(name, shape=(total_images,), maxshape=(None,),
                              dtype=np.int32, compression=COMPRESSION,
                              compression_opts=COMPRESSION_OPTS, chunks=True)
        pg.create_dataset('seed', shape=(total_images,), maxshape=(None,),
                          dtype=np.int64, compression=COMPRESSION,
                          compression_opts=COMPRESSION_OPTS, chunks=True)

        f.attrs['method']           = 'lognormal_flexible_kmax'
        f.attrs['image_dimensions'] = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]
        f.attrs['batch_size']       = BATCH_SIZE
        f.attrs['total_images']     = total_images
        f.attrs['creation_date']    = datetime.now().isoformat()
        f.attrs['generator']        = 'run_generation_flexible_kmax.py'
        f.attrs['num_workers']      = NUM_WORKERS
        f.attrs['kmin_range']       = [KMIN_LOW, KMIN_HIGH]
        f.attrs['kmax_range']       = [KMAX_LOW, KMAX_HIGH]
        f.attrs['sigma_range']      = [SIGMA_LOW, SIGMA_HIGH]


def main() -> None:
    if NUM_IMAGES <= 0:
        raise ValueError(f'NUM_IMAGES must be > 0, got {NUM_IMAGES}')
    if KMIN_HIGH < KMIN_LOW:
        raise ValueError(f'KMIN_HIGH ({KMIN_HIGH}) < KMIN_LOW ({KMIN_LOW})')
    if KMAX_HIGH <= KMIN_LOW:
        raise ValueError(
            f'KMAX_HIGH ({KMAX_HIGH}) <= KMIN_LOW ({KMIN_LOW}): no valid k_max possible')
    if SIGMA_LOW <= 0:
        raise ValueError(f'SIGMA_LOW must be > 0, got {SIGMA_LOW}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERF_DIR,   exist_ok=True)

    if OUTPUT_FILE:
        output_path = (OUTPUT_FILE if os.path.isabs(OUTPUT_FILE)
                       else os.path.join(OUTPUT_DIR, OUTPUT_FILE))
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(
            OUTPUT_DIR,
            f'flexible_kmax_{ts}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}.h5'
        )

    print('=' * 70)
    print('FLEXIBLE DATASET GENERATION — varying k_min, k_max, sigma')
    print('=' * 70)
    print(f'Output:     {output_path}')
    print(f'Images:     {NUM_IMAGES}')
    print(f'Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_DEPTH}')
    print(f'Workers:    {NUM_WORKERS}   Batch: {BATCH_SIZE}')
    print(f'k_min:      [{KMIN_LOW}, {KMIN_HIGH}] uniform')
    print(f'k_max:      [{KMAX_LOW}, {KMAX_HIGH}] uniform, k_max > k_min enforced')
    print(f'sigma:      [{SIGMA_LOW}, {SIGMA_HIGH}] uniform')
    print('=' * 70)

    _create_hdf5(output_path, NUM_IMAGES)

    batch_times  = []
    start_time   = time.time()
    global_index = 0

    with h5py.File(output_path, 'a') as h5f:
        images_ds = h5f['images']
        pg        = h5f['parameters']

        with mp.Pool(processes=NUM_WORKERS) as pool:
            num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in range(num_batches):
                t0 = time.time()
                cur = min(BATCH_SIZE, NUM_IMAGES - global_index)

                k_mins = np.random.uniform(KMIN_LOW, KMIN_HIGH, size=cur)
                k_maxs = np.array([_sample_kmax_valid(float(km), KMAX_LOW, KMAX_HIGH)
                                   for km in k_mins])
                sigmas = np.random.uniform(SIGMA_LOW, SIGMA_HIGH, size=cur)
                means  = np.random.uniform(MEAN_LOW,  MEAN_HIGH,  size=cur)
                seeds  = np.random.randint(0, 2**31,  size=cur, dtype=np.int64)

                work_args = [
                    (float(k_mins[i]), float(k_maxs[i]),
                     IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH,
                     float(means[i]), float(sigmas[i]), float(BETA),
                     False, int(seeds[i]))
                    for i in range(cur)
                ]
                results = pool.starmap(_generate_image_worker, work_args)

                end_index = global_index + cur
                images_ds[global_index:end_index] = np.array(
                    [img for img, _ in results], dtype=np.float32)

                bparams = {k: [] for k in
                           ['k_min', 'k_max', 'sigma', 'mean', 'beta',
                            'ni', 'nj', 'nk', 'seed']}
                for _, p in results:
                    for key in bparams:
                        bparams[key].append(p[key])
                for key, vals in bparams.items():
                    pg[key][global_index:end_index] = vals

                global_index = end_index
                dt      = time.time() - t0
                elapsed = time.time() - start_time
                rate    = global_index / elapsed if elapsed > 0 else 0
                eta     = (NUM_IMAGES - global_index) / rate if rate > 0 else 0
                batch_times.append(dt)
                print(f'Batch {batch_idx+1:3d}/{num_batches} | '
                      f'{global_index:6d}/{NUM_IMAGES} '
                      f'({100*global_index/NUM_IMAGES:5.1f}%) | '
                      f'{rate:7.2f} img/s | ETA {eta/60:6.1f} min')

    total_time   = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)

    perf = {
        'run_type':         'flexible_varying_kmax',
        'output_file':      os.path.basename(output_path),
        'num_images':       NUM_IMAGES,
        'image_size':       [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
        'kmin_range':       [KMIN_LOW, KMIN_HIGH],
        'kmax_range':       [KMAX_LOW, KMAX_HIGH],
        'sigma_range':      [SIGMA_LOW, SIGMA_HIGH],
        'batch_size':       BATCH_SIZE,
        'num_workers':      NUM_WORKERS,
        'time_seconds':     round(total_time, 2),
        'time_minutes':     round(total_time / 60.0, 2),
        'images_per_second': round(NUM_IMAGES / total_time, 2) if total_time > 0 else 0,
        'file_size_mb':     round(file_size_mb, 2),
        'created_at':       datetime.now().isoformat(),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    perf_path = os.path.join(
        PERF_DIR,
        f'flexible_kmax_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}_{ts}_perf.json')
    with open(perf_path, 'w') as fp:
        json.dump(perf, fp, indent=2)

    print('=' * 70)
    print(f'Complete in {total_time / 60:.2f} min | {file_size_mb:.1f} MB')
    print(f'Output: {output_path}')
    print(f'Perf:   {perf_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify script runs with a small test (200 images)**

```bash
cd /home/davas/Documents/cnn_astro
NUM_IMAGES=200 BATCH_SIZE=20 OUTPUT_FILE=test_kmax_200.h5 python scripts/run_generation_flexible_kmax.py
```

Expected: prints batch progress, completes without error, creates `data/raw/test_kmax_200.h5`.

- [ ] **Step 3: Inspect generated dataset — verify k_max varies and constraint holds**

```python
import h5py, numpy as np
with h5py.File('data/raw/test_kmax_200.h5', 'r') as f:
    k_mins = f['parameters/k_min'][:]
    k_maxs = f['parameters/k_max'][:]
    sigmas = f['parameters/sigma'][:]
    print(f"k_min: {k_mins.min():.2f} – {k_mins.max():.2f}")
    print(f"k_max: {k_maxs.min():.2f} – {k_maxs.max():.2f}  (should NOT be constant)")
    print(f"sigma: {sigmas.min():.3f} – {sigmas.max():.3f}")
    violations = (k_maxs <= k_mins).sum()
    print(f"k_max <= k_min violations: {violations}  (must be 0)")
    assert violations == 0, "Constraint enforcement failed"
    assert k_maxs.std() > 1.0, "k_max appears constant — constraint enforcement broken"
print("OK")
```

Expected output:
```
k_min: ~1.x – ~31.x
k_max: ~2.x – ~63.x   (varies, NOT constant at 64)
sigma: ~0.01 – ~4.9x
k_max <= k_min violations: 0  (must be 0)
OK
```

- [ ] **Step 4: Delete test file**

```bash
rm data/raw/test_kmax_200.h5
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_generation_flexible_kmax.py
git commit -m "feat: add flexible dataset generator with varying k_max constraint enforcement"
```

---

## Task 2: Generate full 100k dataset (long-running ~2–4 hours)

**Files:**
- Produces: `data/raw/flexible_kmax_{timestamp}_128x128_100000.h5`

> **Note:** This runs for several hours locally. Use multiple workers if possible (set `NUM_WORKERS` to your CPU count). Run in background or overnight.

- [ ] **Step 1: Run full generation**

```bash
cd /home/davas/Documents/cnn_astro
NUM_IMAGES=100000 BATCH_SIZE=50 NUM_WORKERS=4 python scripts/run_generation_flexible_kmax.py
```

Expected: steady progress output ending with `Complete in X.XX min`.

- [ ] **Step 2: Verify the full dataset**

Replace `<filename>` with the actual generated filename shown in the script output.

```python
import h5py, numpy as np
with h5py.File('data/raw/<filename>', 'r') as f:
    k_mins = f['parameters/k_min'][:]
    k_maxs = f['parameters/k_max'][:]
    sigmas = f['parameters/sigma'][:]
    n      = f['images'].shape[0]
    print(f"Total images: {n}  (expected 100000)")
    print(f"k_min: {k_mins.min():.2f} – {k_mins.max():.2f}")
    print(f"k_max: {k_maxs.min():.2f} – {k_maxs.max():.2f}  std={k_maxs.std():.2f}")
    print(f"sigma: {sigmas.min():.3f} – {sigmas.max():.3f}")
    assert n == 100000
    assert (k_maxs <= k_mins).sum() == 0
    assert k_maxs.std() > 5.0
print("Dataset OK")
```

---

## Task 3: Notebook — Config and shared code cells

**Files:**
- Create: `notebooks/comparison/loss_benchmark.ipynb`

The notebook directory already exists. Create a new notebook with the cells below.

- [ ] **Step 1: Create the notebook with Cell 1 — Imports and config**

Create `notebooks/comparison/loss_benchmark.ipynb` and add this as the first code cell:

```python
# Cell 1 — Imports and config
from pathlib import Path
import json, time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd().parent.parent
# Running from notebooks/comparison/ → go up two levels to project root
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Set DATA_FILE to the actual filename produced in Task 2
DATA_FILE   = next(sorted((PROJECT_ROOT / 'data' / 'raw').glob('flexible_kmax_*.h5'),
                           key=lambda p: p.stat().st_mtime, reverse=True), None)
OUTPUT_BASE = PROJECT_ROOT / 'outputs' / 'comparison' / 'loss_benchmark'
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

assert DATA_FILE is not None and DATA_FILE.exists(), \
    f"Dataset not found. Run Task 2 first.\nLooked in: {PROJECT_ROOT / 'data' / 'raw'}"
print(f"Dataset:  {DATA_FILE.name}")
print(f"Outputs:  {OUTPUT_BASE}")

# ── Hyperparams ────────────────────────────────────────────────────────────
SEED       = 42
EPOCHS     = 30
BATCH_SIZE = 16
LR         = 1e-4

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device:   {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")

# ── Targets for Phase 2 (k_min and k_max) ─────────────────────────────────
TARGETS_P2 = {
    'k_min': {'param_key': 'k_min', 'param_min': 1.0,  'param_max': 32.0},
    'k_max': {'param_key': 'k_max', 'param_min': 1.0,  'param_max': 64.0},
}
```

- [ ] **Step 2: Add Cell 2 — Shared utility code**

```python
# Cell 2 — Shared code: NumpyEncoder, HDF5Dataset, model, loss, train/eval

# ── JSON encoder that handles numpy scalars/arrays ─────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── Lazy-loading HDF5 dataset (per-image min-max normalization) ───────────
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, indices, labels, param_min, param_max):
        self.h5_path   = str(h5_path)
        self.indices   = indices
        self.labels    = labels[indices].astype(np.float32)
        self.param_min = float(param_min)
        self.param_max = float(param_max)
        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        img = self._get_h5()['images'][self.indices[idx]].astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img[np.newaxis]                              # (1, H, W)
        y   = (self.labels[idx] - self.param_min) / (self.param_max - self.param_min)
        return torch.from_numpy(img), torch.tensor(float(y), dtype=torch.float32)

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


# ── ResNet50 regression head (1-channel input) ────────────────────────────
class ResNet50Regressor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


# ── Composite regression loss ─────────────────────────────────────────────
class RegressionLoss(nn.Module):
    """loss_type: 'mae' | 'rmse' | 'rmse_mae' (alpha controls RMSE weight)"""
    def __init__(self, loss_type='rmse', alpha=0.5, eps=1e-8):
        super().__init__()
        self.loss_type = loss_type
        self.alpha     = alpha
        self.eps       = eps

    def forward(self, pred, target):
        if self.loss_type == 'mae':
            return torch.mean(torch.abs(pred - target))
        rmse = torch.sqrt(torch.mean((pred - target) ** 2) + self.eps)
        if self.loss_type == 'rmse':
            return rmse
        # rmse_mae
        mae = torch.mean(torch.abs(pred - target))
        return self.alpha * rmse + (1.0 - self.alpha) * mae


# ── Training helpers ──────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda'):
                loss = criterion(model(X), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, targets = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred  = model(X)
        total += criterion(pred, y).item() * len(X)
        preds.append(pred.cpu())
        targets.append(y.cpu())
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mae = float(np.mean(np.abs(preds - targets)))
    return total / len(loader.dataset), mae, preds, targets


# ── Loss registry ─────────────────────────────────────────────────────────
LOSSES = {
    'mae':      RegressionLoss('mae'),
    'rmse':     RegressionLoss('rmse'),
    'rmse_mae': RegressionLoss('rmse_mae', alpha=0.5),
}
LOSS_LABELS = {
    'mae':      'MAE',
    'rmse':     'RMSE',
    'rmse_mae': '0.5·RMSE + 0.5·MAE',
}
LOSS_COLORS = {'mae': '#2196F3', 'rmse': '#F44336', 'rmse_mae': '#4CAF50'}

print("Shared code loaded.")
print(f"ResNet50 params: {sum(p.numel() for p in ResNet50Regressor().parameters()):,}")
```

Expected output:
```
Shared code loaded.
ResNet50 params: 24,042,817
```

- [ ] **Step 3: Run cells 1 and 2, verify they execute without errors**

---

## Task 4: Notebook — Benchmark loop (k_min and k_max)

**Files:**
- Modify: `notebooks/comparison/loss_benchmark.ipynb` (add Cell 3)

- [ ] **Step 1: Add Cell 3 — Benchmark loop**

```python
# Cell 3 — Benchmark loop: k_min and k_max × {mae, rmse, rmse_mae}
# Resume-safe: skips any run whose results.json already exists.

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

for target_name, tcfg in TARGETS_P2.items():

    with h5py.File(DATA_FILE, 'r') as f:
        n_total      = f['images'].shape[0]
        param_values = f['parameters'][tcfg['param_key']][:]

    indices  = np.arange(n_total)
    train_idx, tmp  = train_test_split(indices, test_size=0.2,  random_state=SEED)
    val_idx,  test_idx = train_test_split(tmp,  test_size=0.5,  random_state=SEED)

    for loss_name, criterion in LOSSES.items():

        out_dir      = OUTPUT_BASE / target_name / loss_name
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / 'results.json'

        if results_path.exists():
            print(f'[SKIP] {target_name}/{loss_name}  —  results.json already exists')
            continue

        print(f'\n{"="*65}')
        print(f'  target={target_name}  loss={loss_name}')
        print(f'{"="*65}')

        pmin, pmax = tcfg['param_min'], tcfg['param_max']
        train_ds = HDF5Dataset(DATA_FILE, train_idx, param_values, pmin, pmax)
        val_ds   = HDF5Dataset(DATA_FILE, val_idx,   param_values, pmin, pmax)
        test_ds  = HDF5Dataset(DATA_FILE, test_idx,  param_values, pmin, pmax)

        kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True,  **kw)
        val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
        test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

        model     = ResNet50Regressor(in_channels=1).to(DEVICE)
        criterion_dev = criterion.to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        scaler    = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

        history        = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        best_val_loss  = float('inf')
        run_start      = time.time()

        for epoch in range(EPOCHS):
            tl = train_epoch(model, train_loader, criterion_dev, optimizer, scaler, DEVICE)
            vl, vm, _, _ = eval_epoch(model, val_loader, criterion_dev, DEVICE)
            scheduler.step(vl)
            history['train_loss'].append(float(tl))
            history['val_loss'].append(float(vl))
            history['val_mae'].append(float(vm))
            lr = optimizer.param_groups[0]['lr']
            print(f'  Ep {epoch+1:2d}/{EPOCHS} | train {tl:.4f} | val {vl:.4f} '
                  f'| mae {vm:.4f} | lr {lr:.1e}')
            if vl < best_val_loss:
                best_val_loss = vl
                torch.save(model.state_dict(), out_dir / 'best_model.pt')

        # Test evaluation
        model.load_state_dict(torch.load(out_dir / 'best_model.pt',
                                          map_location=DEVICE))
        _, _, y_pred_norm, y_true_norm = eval_epoch(
            model, test_loader, criterion_dev, DEVICE)

        y_pred = y_pred_norm * (pmax - pmin) + pmin
        y_true = y_true_norm * (pmax - pmin) + pmin
        mae_   = float(mean_absolute_error(y_true, y_pred))
        rmse_  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2_    = float(r2_score(y_true, y_pred))

        results = {
            'target': target_name, 'loss_fn': loss_name,
            'test_mae': mae_, 'test_rmse': rmse_, 'test_r2': r2_,
            'training_time_min': (time.time() - run_start) / 60.0,
            'history': history,
            'y_pred': y_pred.tolist(),
            'y_true': y_true.tolist(),
        }
        with open(results_path, 'w') as fp:
            json.dump(results, fp, cls=NumpyEncoder, indent=2)

        print(f'\n  Test — MAE:{mae_:.3f}  RMSE:{rmse_:.3f}  R²:{r2_:.4f}')
        print(f'  Saved → {results_path}')

        del model, train_ds, val_ds, test_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print('\nPhase 2 benchmark complete.')
```

> **Runtime note:** This cell runs 6 training jobs (2 targets × 3 losses) × ~4.5 min/epoch × 30 epochs ≈ **~9 hours**. Run overnight. Restart the notebook and re-run this cell to resume — completed runs are skipped automatically.

---

## Task 5: Notebook — Phase 2 comparison plots

**Files:**
- Modify: `notebooks/comparison/loss_benchmark.ipynb` (add Cells 4–8)

- [ ] **Step 1: Add Cell 4 — Load all Phase 2 results**

```python
# Cell 4 — Load Phase 2 results
def load_results(targets):
    """Load results.json for each (target, loss) combination that exists."""
    out = {}
    for t in targets:
        out[t] = {}
        for ln in LOSSES:
            p = OUTPUT_BASE / t / ln / 'results.json'
            if p.exists():
                with open(p) as f:
                    out[t][ln] = json.load(f)
            else:
                print(f'[MISSING] {t}/{ln}')
    return out

results_p2 = load_results(list(TARGETS_P2.keys()))
TARGETS_P2_DONE = [t for t in TARGETS_P2 if results_p2.get(t)]
print(f"Loaded results for: {TARGETS_P2_DONE}")
```

- [ ] **Step 2: Add Cell 5 — Plot 1: Training curves grid**

```python
# Cell 5 — Plot 1: Training curves (val loss per epoch)
loss_names = list(LOSSES.keys())
n_rows = len(TARGETS_P2_DONE)

fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
axes = np.array(axes).reshape(n_rows, 3)

for i, target in enumerate(TARGETS_P2_DONE):
    for j, ln in enumerate(loss_names):
        ax = axes[i, j]
        if ln in results_p2[target]:
            r = results_p2[target][ln]
            ax.plot(r['history']['train_loss'], label='Train', alpha=0.85)
            ax.plot(r['history']['val_loss'],   label='Val',   alpha=0.85)
        ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Training Curves — val loss per epoch', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Add Cell 6 — Plot 2 & 5: Metrics bar chart + summary table**

```python
# Cell 6 — Plot 2: Final test metrics bar chart + summary table
metrics     = ['test_mae', 'test_rmse', 'test_r2']
metric_lbls = ['MAE', 'RMSE', 'R²']
x           = np.arange(len(loss_names))

fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
axes = np.array(axes).reshape(n_rows, 3)

for i, target in enumerate(TARGETS_P2_DONE):
    for k, (metric, mlbl) in enumerate(zip(metrics, metric_lbls)):
        ax   = axes[i, k]
        vals = [results_p2[target].get(ln, {}).get(metric, 0) for ln in loss_names]
        bars = ax.bar(x, vals, color=[LOSS_COLORS[ln] for ln in loss_names], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([LOSS_LABELS[ln] for ln in loss_names], rotation=12, fontsize=8)
        ax.set_title(f'{target}  |  {mlbl}'); ax.set_ylabel(mlbl)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Test Metrics by Loss Function', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Summary table
print(f"\n{'='*72}")
print(f"{'TARGET':<10} {'LOSS':<25} {'MAE':>7} {'RMSE':>7} {'R²':>8} {'min':>7}")
print(f"{'='*72}")
for target in TARGETS_P2_DONE:
    for ln in loss_names:
        r = results_p2[target].get(ln)
        if r:
            print(f"{target:<10} {LOSS_LABELS[ln]:<25} "
                  f"{r['test_mae']:>7.3f} {r['test_rmse']:>7.3f} "
                  f"{r['test_r2']:>8.4f} {r['training_time_min']:>7.1f}")
print(f"{'='*72}")
```

- [ ] **Step 4: Add Cell 7 — Plot 3: Predicted vs true scatter**

```python
# Cell 7 — Plot 3: Predicted vs true scatter
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
axes = np.array(axes).reshape(n_rows, 3)

for i, target in enumerate(TARGETS_P2_DONE):
    for j, ln in enumerate(loss_names):
        ax = axes[i, j]
        if ln in results_p2[target]:
            r      = results_p2[target][ln]
            y_true = np.array(r['y_true'])
            y_pred = np.array(r['y_pred'])
            ax.scatter(y_true, y_pred, alpha=0.15, s=4, color=LOSS_COLORS[ln])
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(lims, lims, 'k--', linewidth=1)
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}\nR²={r["test_r2"]:.4f}')
        else:
            ax.text(0.5, 0.5, 'not run', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}')
        ax.set_xlabel(f'True {target}'); ax.set_ylabel(f'Pred {target}')
        ax.grid(True, alpha=0.3)

plt.suptitle('Predicted vs True', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Add Cell 8 — Plot 4: Residual distributions**

```python
# Cell 8 — Plot 4: Residual distributions
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
axes = np.array(axes).reshape(n_rows, 3)

for i, target in enumerate(TARGETS_P2_DONE):
    for j, ln in enumerate(loss_names):
        ax = axes[i, j]
        if ln in results_p2[target]:
            r   = results_p2[target][ln]
            res = np.array(r['y_pred']) - np.array(r['y_true'])
            ax.hist(res, bins=60, color=LOSS_COLORS[ln], alpha=0.75, edgecolor='white')
            ax.axvline(0,          color='black', linestyle='--', linewidth=1)
            ax.axvline(res.mean(), color='red',   linestyle='--', linewidth=1,
                       label=f'mean={res.mean():.3f}')
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}')
        else:
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}')
        ax.set_xlabel('Residual (pred − true)'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Residual Distributions', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## Task 6: Notebook — Sigma section (Phase 3)

**Files:**
- Modify: `notebooks/comparison/loss_benchmark.ipynb` (add Cells 9–14)

- [ ] **Step 1: Add Cell 9 — Sigma section header and global percentile computation**

```python
# Cell 9 — Sigma section: compute global image percentiles (run once)
# ─────────────────────────────────────────────────────────────────────────
# Sigma encodes the log-normal spread → visible as global image contrast.
# Per-image normalization in HDF5Dataset destroys this signal.
# Solution: normalize all images with fixed p1/p99 percentiles from the dataset.

np.random.seed(SEED)
with h5py.File(DATA_FILE, 'r') as f:
    n_total  = f['images'].shape[0]
    samp_idx = np.sort(np.random.choice(n_total, size=1000, replace=False))
    samp_imgs = f['images'][samp_idx]          # (1000, 128, 128)

IMG_P1  = float(np.percentile(samp_imgs, 1))
IMG_P99 = float(np.percentile(samp_imgs, 99))
del samp_imgs

# Log-scale normalization constants
LOG_SIGMA_MIN = float(np.log(0.01))
LOG_SIGMA_MAX = float(np.log(5.0))

print(f'Global image percentiles:  p1={IMG_P1:.4f}  p99={IMG_P99:.4f}')
print(f'Log-sigma range:           [{LOG_SIGMA_MIN:.4f}, {LOG_SIGMA_MAX:.4f}]')
```

- [ ] **Step 2: Add Cell 10 — SigmaHDF5Dataset**

```python
# Cell 10 — SigmaHDF5Dataset (fixed normalization + log-scale target)

class SigmaHDF5Dataset(Dataset):
    """
    Differences from HDF5Dataset:
      - Global percentile image normalization (preserves contrast = sigma signal)
      - log-scale target: y = (ln(sigma) - ln(0.01)) / (ln(5.0) - ln(0.01))
    """
    def __init__(self, h5_path, indices, sigma_values, img_p1, img_p99):
        self.h5_path  = str(h5_path)
        self.indices  = indices
        self.img_p1   = float(img_p1)
        self.img_p99  = float(img_p99)
        self.img_range = self.img_p99 - self.img_p1 + 1e-8

        log_sigma   = np.log(sigma_values[indices].astype(np.float64))
        self.labels = ((log_sigma - LOG_SIGMA_MIN) /
                       (LOG_SIGMA_MAX - LOG_SIGMA_MIN)).astype(np.float32)
        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        img = self._get_h5()['images'][self.indices[idx]].astype(np.float32)
        img = (img - self.img_p1) / self.img_range
        img = img[np.newaxis]
        return torch.from_numpy(img), torch.tensor(float(self.labels[idx]),
                                                    dtype=torch.float32)

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()

print("SigmaHDF5Dataset defined.")
```

- [ ] **Step 3: Add Cell 11 — Sigma benchmark loop**

```python
# Cell 11 — Sigma benchmark loop (same structure as Phase 2)
# Resume-safe: skips runs whose results.json already exists.

with h5py.File(DATA_FILE, 'r') as f:
    n_total      = f['images'].shape[0]
    sigma_values = f['parameters']['sigma'][:]

indices  = np.arange(n_total)
train_idx_s, tmp_s  = train_test_split(indices, test_size=0.2,  random_state=SEED)
val_idx_s,  test_idx_s = train_test_split(tmp_s, test_size=0.5, random_state=SEED)

for loss_name, criterion in LOSSES.items():
    out_dir      = OUTPUT_BASE / 'sigma' / loss_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / 'results.json'

    if results_path.exists():
        print(f'[SKIP] sigma/{loss_name}  —  results.json already exists')
        continue

    print(f'\n{"="*65}')
    print(f'  target=sigma  loss={loss_name}')
    print(f'{"="*65}')

    train_ds = SigmaHDF5Dataset(DATA_FILE, train_idx_s, sigma_values, IMG_P1, IMG_P99)
    val_ds   = SigmaHDF5Dataset(DATA_FILE, val_idx_s,   sigma_values, IMG_P1, IMG_P99)
    test_ds  = SigmaHDF5Dataset(DATA_FILE, test_idx_s,  sigma_values, IMG_P1, IMG_P99)

    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    model         = ResNet50Regressor(in_channels=1).to(DEVICE)
    criterion_dev = criterion.to(DEVICE)
    optimizer     = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    scaler        = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    history       = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    run_start     = time.time()

    for epoch in range(EPOCHS):
        tl = train_epoch(model, train_loader, criterion_dev, optimizer, scaler, DEVICE)
        vl, vm, _, _ = eval_epoch(model, val_loader, criterion_dev, DEVICE)
        scheduler.step(vl)
        history['train_loss'].append(float(tl))
        history['val_loss'].append(float(vl))
        history['val_mae'].append(float(vm))
        lr = optimizer.param_groups[0]['lr']
        print(f'  Ep {epoch+1:2d}/{EPOCHS} | train {tl:.4f} | val {vl:.4f} '
              f'| mae {vm:.4f} | lr {lr:.1e}')
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), out_dir / 'best_model.pt')

    # Test evaluation — denormalize from log-scale
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=DEVICE))
    _, _, y_pred_norm, y_true_norm = eval_epoch(model, test_loader, criterion_dev, DEVICE)

    # Denormalize: [0,1] → log(sigma) → sigma
    y_pred_log   = y_pred_norm * (LOG_SIGMA_MAX - LOG_SIGMA_MIN) + LOG_SIGMA_MIN
    y_true_log   = y_true_norm * (LOG_SIGMA_MAX - LOG_SIGMA_MIN) + LOG_SIGMA_MIN
    y_pred_sigma = np.exp(y_pred_log)
    y_true_sigma = np.exp(y_true_log)

    mae_  = float(mean_absolute_error(y_true_sigma, y_pred_sigma))
    rmse_ = float(np.sqrt(mean_squared_error(y_true_sigma, y_pred_sigma)))
    r2_   = float(r2_score(y_true_sigma, y_pred_sigma))

    results = {
        'target': 'sigma', 'loss_fn': loss_name,
        'test_mae': mae_, 'test_rmse': rmse_, 'test_r2': r2_,
        'training_time_min': (time.time() - run_start) / 60.0,
        'history': history,
        'y_pred': y_pred_sigma.tolist(),
        'y_true': y_true_sigma.tolist(),
    }
    with open(results_path, 'w') as fp:
        json.dump(results, fp, cls=NumpyEncoder, indent=2)

    print(f'\n  Test — MAE:{mae_:.3f}  RMSE:{rmse_:.3f}  R²:{r2_:.4f}')
    print(f'  Saved → {results_path}')

    del model, train_ds, val_ds, test_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('\nPhase 3 (sigma) benchmark complete.')
```

> **Runtime note:** 3 more training runs × ~4.5 min/epoch × 30 epochs ≈ **~7 hours** additional.

- [ ] **Step 4: Add Cell 12 — Load all results (Phases 2 + 3)**

```python
# Cell 12 — Load all results: k_min, k_max, sigma
ALL_TARGETS = ['k_min', 'k_max', 'sigma']
results_all = load_results(ALL_TARGETS)   # reuses load_results() from Cell 4
ALL_DONE    = [t for t in ALL_TARGETS if results_all.get(t)]
print(f"Results available for: {ALL_DONE}")
```

- [ ] **Step 5: Add Cell 13 — Full comparison plots (all 3 targets)**

```python
# Cell 13 — Full comparison plots (k_min + k_max + sigma)
n_rows_all = len(ALL_DONE)

# Plot 1: Training curves
fig, axes = plt.subplots(n_rows_all, 3, figsize=(15, 4 * n_rows_all))
axes = np.array(axes).reshape(n_rows_all, 3)
for i, target in enumerate(ALL_DONE):
    for j, ln in enumerate(loss_names):
        ax = axes[i, j]
        if ln in results_all.get(target, {}):
            r = results_all[target][ln]
            ax.plot(r['history']['train_loss'], label='Train', alpha=0.85)
            ax.plot(r['history']['val_loss'],   label='Val',   alpha=0.85)
        ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Training Curves — all targets', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Plot 2: Metrics bar chart
fig, axes = plt.subplots(n_rows_all, 3, figsize=(15, 4 * n_rows_all))
axes = np.array(axes).reshape(n_rows_all, 3)
for i, target in enumerate(ALL_DONE):
    for k, (metric, mlbl) in enumerate(zip(metrics, metric_lbls)):
        ax   = axes[i, k]
        vals = [results_all[target].get(ln, {}).get(metric, 0) for ln in loss_names]
        bars = ax.bar(x, vals, color=[LOSS_COLORS[ln] for ln in loss_names], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([LOSS_LABELS[ln] for ln in loss_names], rotation=12, fontsize=8)
        ax.set_title(f'{target}  |  {mlbl}'); ax.set_ylabel(mlbl)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
plt.suptitle('Test Metrics — all targets', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Plot 3: Scatter
fig, axes = plt.subplots(n_rows_all, 3, figsize=(15, 5 * n_rows_all))
axes = np.array(axes).reshape(n_rows_all, 3)
for i, target in enumerate(ALL_DONE):
    for j, ln in enumerate(loss_names):
        ax = axes[i, j]
        if ln in results_all.get(target, {}):
            r = results_all[target][ln]
            yt, yp = np.array(r['y_true']), np.array(r['y_pred'])
            ax.scatter(yt, yp, alpha=0.15, s=4, color=LOSS_COLORS[ln])
            lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            ax.plot(lims, lims, 'k--', linewidth=1)
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}\nR²={r["test_r2"]:.4f}')
        ax.set_xlabel(f'True {target}'); ax.set_ylabel(f'Pred {target}')
        ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs True — all targets', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Plot 4: Residuals
fig, axes = plt.subplots(n_rows_all, 3, figsize=(15, 4 * n_rows_all))
axes = np.array(axes).reshape(n_rows_all, 3)
for i, target in enumerate(ALL_DONE):
    for j, ln in enumerate(loss_names):
        ax = axes[i, j]
        if ln in results_all.get(target, {}):
            r   = results_all[target][ln]
            res = np.array(r['y_pred']) - np.array(r['y_true'])
            ax.hist(res, bins=60, color=LOSS_COLORS[ln], alpha=0.75, edgecolor='white')
            ax.axvline(0,          color='black', linestyle='--', linewidth=1)
            ax.axvline(res.mean(), color='red',   linestyle='--', linewidth=1,
                       label=f'mean={res.mean():.3f}')
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}')
            ax.legend(fontsize=8)
        ax.set_xlabel('Residual (pred − true)'); ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
plt.suptitle('Residual Distributions — all targets', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

- [ ] **Step 6: Add Cell 14 — Full summary table**

```python
# Cell 14 — Full summary table (all targets × all losses)
print(f"\n{'='*75}")
print(f"{'TARGET':<10} {'LOSS':<25} {'MAE':>8} {'RMSE':>8} {'R²':>9} {'min':>7}")
print(f"{'='*75}")
for target in ALL_DONE:
    for ln in loss_names:
        r = results_all[target].get(ln)
        if r:
            print(f"{target:<10} {LOSS_LABELS[ln]:<25} "
                  f"{r['test_mae']:>8.3f} {r['test_rmse']:>8.3f} "
                  f"{r['test_r2']:>9.4f} {r['training_time_min']:>7.1f}")
    print()
print(f"{'='*75}")
```

---

## Self-Review

**Spec coverage:**
- [x] New generation script with k_max constraint enforcement — Task 1
- [x] 100k dataset with k_min/k_max/sigma all varying — Task 2
- [x] NumpyEncoder fixing float32 JSON bug — Task 3, Cell 2
- [x] HDF5Dataset with per-image normalization for k_min/k_max — Task 3, Cell 2
- [x] RegressionLoss: MAE, RMSE, 0.5·RMSE+0.5·MAE — Task 3, Cell 2
- [x] Benchmark loop with skip-if-exists resume — Task 4, Cell 3
- [x] Comparison plots 1–4 inline (not saved) — Task 5
- [x] Summary table — Task 5, Cell 6
- [x] Global percentile computation for sigma — Task 6, Cell 9
- [x] SigmaHDF5Dataset with global normalization + log-scale target — Task 6, Cell 10
- [x] Sigma benchmark loop with log denormalization at eval — Task 6, Cell 11
- [x] Extended comparison plots including sigma — Task 6, Cell 13

**Type consistency:** `NumpyEncoder`, `HDF5Dataset`, `SigmaHDF5Dataset`, `ResNet50Regressor`, `RegressionLoss`, `LOSSES`, `LOSS_LABELS`, `LOSS_COLORS`, `load_results` — all defined in Task 3 and referenced consistently throughout Tasks 4–6. `LOG_SIGMA_MIN`/`LOG_SIGMA_MAX` defined in Cell 9 (Task 6, Step 1) before use in Cell 10 and Cell 11.
