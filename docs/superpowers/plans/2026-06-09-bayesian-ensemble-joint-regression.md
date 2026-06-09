# Bayesian Ensemble Joint Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-member deep ensemble of heteroscedastic ResNet50 regressors that predicts (μ ± σ) for k_min, k_max, sigma — trainable on HPC via a single SLURM submission, with local inference via a Jupyter notebook.

**Architecture:** ResNet50 backbone with a 6-output heteroscedastic head (μ + log_σ per parameter). Trained with weighted Gaussian NLL loss (k_max weight 2×). Five independent members run as a SLURM job array; evaluation is auto-triggered via `--dependency=afterok`. Inference notebook runs on Windows PC.

**Tech Stack:** PyTorch 2.x, torchvision, h5py, scikit-learn, matplotlib, scipy, pyyaml, astropy (inference notebook only), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/hpc_ensemble/config.yaml` | Create | All tunable params; HPC fields clearly marked |
| `scripts/hpc_ensemble/config.py` | Create | `load_config(path) → dict` |
| `scripts/hpc_ensemble/dataset.py` | Create | `JointHDF5Dataset`, `load_splits()`, `denorm()` |
| `scripts/hpc_ensemble/model.py` | Create | `ResNet50HeteroJoint` — output (B, 3, 2) |
| `scripts/hpc_ensemble/loss.py` | Create | `WeightedGaussianNLL` |
| `scripts/hpc_ensemble/train_member.py` | Create | CLI: trains one ensemble member |
| `scripts/hpc_ensemble/evaluate_ensemble.py` | Create | Combines 5 checkpoints, writes results + plots |
| `scripts/hpc_ensemble/submit.sh` | Create | Professor runs once — submits array + eval |
| `scripts/hpc_ensemble/job_ensemble.sh` | Create | SLURM array `--array=0-4` |
| `scripts/hpc_ensemble/job_evaluate.sh` | Create | SLURM single job, triggered by afterok |
| `scripts/hpc_ensemble/setup_env.sh` | Create | One-time conda/pip env setup on HPC |
| `scripts/hpc_ensemble/requirements.txt` | Create | Pinned deps |
| `scripts/hpc_ensemble/tests/conftest.py` | Create | Pytest fixtures (fake HDF5, fake config) |
| `scripts/hpc_ensemble/tests/test_dataset.py` | Create | Dataset shape, normalisation |
| `scripts/hpc_ensemble/tests/test_model.py` | Create | Output shape, μ in [0,1], σ > 0 |
| `scripts/hpc_ensemble/tests/test_loss.py` | Create | Loss finite, gradients flow, weighting works |
| `notebooks/comparison/bayesian_ensemble_inference.ipynb` | Create | Local inference: synthetic test set + GASS |

---

## Task 1: Scaffold, Config, and Loader

**Files:**
- Create: `scripts/hpc_ensemble/config.yaml`
- Create: `scripts/hpc_ensemble/config.py`
- Create: `scripts/hpc_ensemble/tests/conftest.py`

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p scripts/hpc_ensemble/tests
touch scripts/hpc_ensemble/__init__.py
touch scripts/hpc_ensemble/tests/__init__.py
```

- [ ] **Step 2: Write `config.yaml`**

```yaml
# scripts/hpc_ensemble/config.yaml

# ── HPC SETTINGS (professor configures these) ──────────────────────
hpc:
  partition: "gpu"               # CHANGE: GPU partition name
  gpus_per_task: 1               # CHANGE: GPUs per member job
  cpus_per_task: 8               # CHANGE: CPUs for data loading workers
  mem_gb: 32                     # CHANGE: RAM per job in GB
  time_limit: "06:00:00"         # CHANGE: wall-clock limit per member (HH:MM:SS)
  conda_env: "py311"             # CHANGE: conda environment name on HPC
  project_dir: "/path/to/cnn_astro"  # CHANGE: absolute path to project root on HPC

# ── TRAINING ────────────────────────────────────────────────────────
training:
  epochs: 80
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 1.0e-4
  base_seed: 42                  # member i uses seed = base_seed + i (i.e. 42, 43, ..., 46)

# ── ENSEMBLE ────────────────────────────────────────────────────────
ensemble:
  n_members: 5

# ── LOSS WEIGHTS ────────────────────────────────────────────────────
loss_weights:
  k_min: 1.0
  k_max: 2.0                     # upweighted — primary accuracy target
  sigma: 1.0

# ── DATA ────────────────────────────────────────────────────────────
data:
  file: "data/raw/uniform_kmin_128x128_100000.h5"
  val_split: 0.1
  test_split: 0.1
  img_p1: -2.0818
  img_p99: 0.9409
  kmin_lo: 1.0
  kmin_hi: 62.0
  kmax_lo: 5.0
  kmax_hi: 64.0
  log_sigma_lo: -4.6052          # log(0.01)
  log_sigma_hi: 1.6094           # log(5.0)

# ── OUTPUT ──────────────────────────────────────────────────────────
output:
  dir: "outputs/hpc_ensemble"
```

- [ ] **Step 3: Write `config.py`**

```python
# scripts/hpc_ensemble/config.py
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def output_dir(cfg: dict, member_id: int = None) -> Path:
    base = Path(cfg['hpc']['project_dir']) / cfg['output']['dir']
    if member_id is not None:
        return base / f'member_{member_id}'
    return base
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
# scripts/hpc_ensemble/tests/conftest.py
import numpy as np
import h5py
import pytest


@pytest.fixture
def fake_cfg(tmp_path):
    h5_path = tmp_path / 'fake.h5'
    n = 40
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('images', data=np.random.randn(n, 128, 128).astype(np.float32))
        grp = hf.create_group('parameters')
        grp.create_dataset('k_min',  data=np.random.uniform(1, 62, n).astype(np.float32))
        grp.create_dataset('k_max',  data=np.random.uniform(5, 64, n).astype(np.float32))
        grp.create_dataset('sigma',  data=np.random.uniform(0.01, 5.0, n).astype(np.float32))

    return {
        'hpc': {'project_dir': str(tmp_path)},
        'training': {'base_seed': 42, 'batch_size': 8, 'lr': 1e-4,
                     'weight_decay': 1e-4, 'epochs': 2},
        'ensemble': {'n_members': 5},
        'loss_weights': {'k_min': 1.0, 'k_max': 2.0, 'sigma': 1.0},
        'data': {
            'file': 'fake.h5',
            'val_split': 0.1, 'test_split': 0.1,
            'img_p1': -2.0818, 'img_p99': 0.9409,
            'kmin_lo': 1.0, 'kmin_hi': 62.0,
            'kmax_lo': 5.0, 'kmax_hi': 64.0,
            'log_sigma_lo': -4.6052, 'log_sigma_hi': 1.6094,
        },
        'output': {'dir': 'outputs/hpc_ensemble'},
    }
```

- [ ] **Step 5: Commit**

```bash
git add scripts/hpc_ensemble/ 
git commit -m "feat: scaffold hpc_ensemble package — config, fixtures"
```

---

## Task 2: Dataset

**Files:**
- Create: `scripts/hpc_ensemble/dataset.py`
- Create: `scripts/hpc_ensemble/tests/test_dataset.py`

Run all tests from `scripts/hpc_ensemble/` with `pytest tests/ -v`.

- [ ] **Step 1: Write the failing tests**

```python
# scripts/hpc_ensemble/tests/test_dataset.py
import numpy as np
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import JointHDF5Dataset, load_splits, denorm
import h5py


def test_dataset_len(fake_cfg, tmp_path):
    h5_path = tmp_path / 'fake.h5'
    with h5py.File(h5_path, 'r') as hf:
        n = hf['images'].shape[0]
        params = {
            'k_min': hf['parameters']['k_min'][:],
            'k_max': hf['parameters']['k_max'][:],
            'sigma': hf['parameters']['sigma'][:],
        }
    indices = np.arange(n)
    ds = JointHDF5Dataset(h5_path, indices, params, fake_cfg)
    assert len(ds) == n


def test_dataset_getitem_shapes(fake_cfg, tmp_path):
    h5_path = tmp_path / 'fake.h5'
    with h5py.File(h5_path, 'r') as hf:
        params = {k: hf['parameters'][k][:] for k in ('k_min', 'k_max', 'sigma')}
    indices = np.arange(10)
    ds = JointHDF5Dataset(h5_path, indices, params, fake_cfg)
    img, label = ds[0]
    assert img.shape == (1, 128, 128)
    assert label.shape == (3,)
    assert img.dtype == torch.float32
    assert label.dtype == torch.float32


def test_dataset_labels_normalised(fake_cfg, tmp_path):
    h5_path = tmp_path / 'fake.h5'
    with h5py.File(h5_path, 'r') as hf:
        params = {k: hf['parameters'][k][:] for k in ('k_min', 'k_max', 'sigma')}
    indices = np.arange(40)
    ds = JointHDF5Dataset(h5_path, indices, params, fake_cfg)
    labels = ds.labels  # (N, 3)
    assert labels.min() >= -0.1, "labels should be close to [0, 1]"
    assert labels.max() <= 1.1, "labels should be close to [0, 1]"


def test_denorm_roundtrip(fake_cfg):
    norm = np.array([[0.5, 0.5, 0.5],
                     [0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0]])
    result = denorm(norm, fake_cfg)
    data = fake_cfg['data']
    np.testing.assert_allclose(result['k_min'][0], (data['kmin_lo'] + data['kmin_hi']) / 2, rtol=1e-5)
    np.testing.assert_allclose(result['k_min'][1], data['kmin_lo'], rtol=1e-5)
    np.testing.assert_allclose(result['k_min'][2], data['kmin_hi'], rtol=1e-5)


def test_load_splits_sizes(fake_cfg, tmp_path):
    train_ds, val_ds, test_ds = load_splits(fake_cfg)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == 40
    assert len(train_ds) > len(val_ds)
    assert len(val_ds) > 0
    assert len(test_ds) > 0
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd scripts/hpc_ensemble
pytest tests/test_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'dataset'`

- [ ] **Step 3: Write `dataset.py`**

```python
# scripts/hpc_ensemble/dataset.py
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class JointHDF5Dataset(Dataset):
    """HDF5 dataset with global p1/p99 image normalisation and [0,1] target normalisation."""

    def __init__(self, h5_path, indices: np.ndarray, params: Dict[str, np.ndarray], cfg: dict):
        self.h5_path = str(h5_path)
        self.indices = indices
        data = cfg['data']
        self.img_p1 = data['img_p1']
        self.img_range = float(data['img_p99'] - data['img_p1']) + 1e-8

        kmin_n = (params['k_min'][indices] - data['kmin_lo']) / (data['kmin_hi'] - data['kmin_lo'])
        kmax_n = (params['k_max'][indices] - data['kmax_lo']) / (data['kmax_hi'] - data['kmax_lo'])
        log_sig = np.log(np.clip(params['sigma'][indices].astype(np.float64), 1e-9, None))
        sig_n = (log_sig - data['log_sigma_lo']) / (data['log_sigma_hi'] - data['log_sigma_lo'])

        self.labels = np.stack([kmin_n, kmax_n, sig_n], axis=1).astype(np.float32)  # (N, 3)
        self._h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        img = self._h5['images'][self.indices[idx]].astype(np.float32)
        img = (img - self.img_p1) / self.img_range
        return torch.from_numpy(img[np.newaxis]), torch.from_numpy(self.labels[idx])

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def denorm(pred_norm: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """Convert (N, 3) normalised predictions to original parameter units.

    Returns dict with keys 'k_min', 'k_max', 'sigma'.
    """
    d = cfg['data']
    kmin = pred_norm[:, 0] * (d['kmin_hi'] - d['kmin_lo']) + d['kmin_lo']
    kmax = pred_norm[:, 1] * (d['kmax_hi'] - d['kmax_lo']) + d['kmax_lo']
    sigma = np.exp(pred_norm[:, 2] * (d['log_sigma_hi'] - d['log_sigma_lo']) + d['log_sigma_lo'])
    return {'k_min': kmin, 'k_max': kmax, 'sigma': sigma}


def sigma_uncertainty_to_orig(sigma_norm: np.ndarray, mu_norm: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert normalised σ for the 'sigma' parameter to original sigma units.

    Uses the lognormal approximation: σ_orig ≈ μ_sigma_orig * σ_log_sigma
    where σ_log_sigma = σ_norm * (log_sigma_hi - log_sigma_lo).
    """
    d = cfg['data']
    mu_sigma_orig = np.exp(mu_norm * (d['log_sigma_hi'] - d['log_sigma_lo']) + d['log_sigma_lo'])
    sigma_log_sigma = sigma_norm * (d['log_sigma_hi'] - d['log_sigma_lo'])
    return mu_sigma_orig * sigma_log_sigma


def uncertainty_to_orig(sigma_norm: np.ndarray, mu_norm: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """Convert (N, 3) normalised uncertainty to original parameter units.

    Returns dict with keys 'k_min', 'k_max', 'sigma'.
    """
    d = cfg['data']
    return {
        'k_min':  sigma_norm[:, 0] * (d['kmin_hi'] - d['kmin_lo']),
        'k_max':  sigma_norm[:, 1] * (d['kmax_hi'] - d['kmax_lo']),
        'sigma':  sigma_uncertainty_to_orig(sigma_norm[:, 2], mu_norm[:, 2], cfg),
    }


def load_splits(cfg: dict):
    """Load HDF5 and return (train_ds, val_ds, test_ds)."""
    data_cfg = cfg['data']
    h5_path = Path(cfg['hpc']['project_dir']) / data_cfg['file']
    seed = cfg['training']['base_seed']

    with h5py.File(h5_path, 'r') as hf:
        n = hf['images'].shape[0]
        params = {k: hf['parameters'][k][:] for k in ('k_min', 'k_max', 'sigma')}

    indices = np.arange(n)
    test_ratio = data_cfg['val_split'] + data_cfg['test_split']
    train_idx, tmp = train_test_split(indices, test_size=test_ratio, random_state=seed)
    val_ratio = data_cfg['val_split'] / test_ratio
    val_idx, test_idx = train_test_split(tmp, test_size=1.0 - val_ratio, random_state=seed)

    kwargs = dict(params=params, cfg=cfg)
    return (
        JointHDF5Dataset(h5_path, train_idx, **kwargs),
        JointHDF5Dataset(h5_path, val_idx,   **kwargs),
        JointHDF5Dataset(h5_path, test_idx,  **kwargs),
    )
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
cd scripts/hpc_ensemble
pytest tests/test_dataset.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/hpc_ensemble/dataset.py scripts/hpc_ensemble/tests/test_dataset.py
git commit -m "feat: add JointHDF5Dataset, denorm, load_splits"
```

---

## Task 3: Model

**Files:**
- Create: `scripts/hpc_ensemble/model.py`
- Create: `scripts/hpc_ensemble/tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# scripts/hpc_ensemble/tests/test_model.py
import torch
import torch.nn.functional as F
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ResNet50HeteroJoint


@pytest.fixture
def model():
    return ResNet50HeteroJoint(in_channels=1, n_params=3)


def test_output_shape(model):
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    assert out.shape == (4, 3, 2), f"Expected (4, 3, 2), got {out.shape}"


def test_mu_bounded(model):
    x = torch.randn(8, 1, 128, 128)
    out = model(x)
    mu = out[..., 0]  # (B, 3)
    assert mu.min().item() >= 0.0, "μ must be >= 0 (Sigmoid applied)"
    assert mu.max().item() <= 1.0, "μ must be <= 1 (Sigmoid applied)"


def test_sigma_positive_after_softplus(model):
    x = torch.randn(8, 1, 128, 128)
    out = model(x)
    log_sigma = out[..., 1]  # (B, 3)
    sigma = F.softplus(log_sigma)
    assert (sigma > 0).all(), "σ must be > 0 after softplus"


def test_gradients_flow(model):
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found"


def test_param_count(model):
    n = sum(p.numel() for p in model.parameters())
    assert n > 20_000_000, "ResNet50 should have >20M params"
    assert n < 30_000_000, "ResNet50 should have <30M params"
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd scripts/hpc_ensemble
pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Write `model.py`**

```python
# scripts/hpc_ensemble/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50HeteroJoint(nn.Module):
    """ResNet50 with a heteroscedastic regression head.

    Forward output shape: (B, n_params, 2)
        [..., 0] = μ  — passed through Sigmoid, bounded [0, 1] in normalised space
        [..., 1] = log_σ — unconstrained; apply F.softplus() at inference to get σ > 0

    Parameter order: [k_min, k_max, sigma]
    """

    def __init__(self, in_channels: int = 1, n_params: int = 3):
        super().__init__()
        self.n_params = n_params
        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_params * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                       # (B, 2048)
        raw = self.head(features)                          # (B, n_params * 2)
        raw = raw.view(-1, self.n_params, 2)              # (B, n_params, 2)
        mu = torch.sigmoid(raw[..., 0])                   # (B, n_params) — [0, 1]
        log_sigma = raw[..., 1]                           # (B, n_params) — unconstrained
        return torch.stack([mu, log_sigma], dim=-1)       # (B, n_params, 2)
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
cd scripts/hpc_ensemble
pytest tests/test_model.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/hpc_ensemble/model.py scripts/hpc_ensemble/tests/test_model.py
git commit -m "feat: add ResNet50HeteroJoint heteroscedastic model"
```

---

## Task 4: Loss Function

**Files:**
- Create: `scripts/hpc_ensemble/loss.py`
- Create: `scripts/hpc_ensemble/tests/test_loss.py`

- [ ] **Step 1: Write the failing tests**

```python
# scripts/hpc_ensemble/tests/test_loss.py
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loss import WeightedGaussianNLL


@pytest.fixture
def criterion():
    return WeightedGaussianNLL(weights=[1.0, 2.0, 1.0])


def _make_pred(B=8, n=3, mu_val=0.5, log_sigma_val=0.0):
    mu = torch.full((B, n), mu_val)
    log_sigma = torch.full((B, n), log_sigma_val)
    return torch.stack([mu, log_sigma], dim=-1)


def test_loss_is_finite(criterion):
    pred = _make_pred()
    target = torch.rand(8, 3)
    loss = criterion(pred, target)
    assert torch.isfinite(loss), "Loss must be finite"


def test_loss_is_scalar(criterion):
    pred = _make_pred()
    target = torch.rand(8, 3)
    loss = criterion(pred, target)
    assert loss.shape == (), f"Loss must be scalar, got shape {loss.shape}"


def test_gradients_flow(criterion):
    pred = _make_pred()
    pred.requires_grad_(True)
    target = torch.rand(8, 3)
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None


def test_higher_sigma_reduces_penalty_for_wrong_pred(criterion):
    target = torch.zeros(8, 3)
    pred_tight = _make_pred(mu_val=0.9, log_sigma_val=-3.0)  # very confident, wrong
    pred_wide  = _make_pred(mu_val=0.9, log_sigma_val=+3.0)  # very uncertain, same μ
    loss_tight = criterion(pred_tight, target).item()
    loss_wide  = criterion(pred_wide,  target).item()
    assert loss_tight > loss_wide, "Confident wrong prediction should have higher loss"


def test_kmax_weight_applied():
    """Loss with k_max weight=2 should be higher than weight=1 when k_max error is large."""
    criterion_equal  = WeightedGaussianNLL(weights=[1.0, 1.0, 1.0])
    criterion_kmax2  = WeightedGaussianNLL(weights=[1.0, 2.0, 1.0])

    pred = _make_pred(mu_val=0.0, log_sigma_val=0.0)
    target = torch.ones(8, 3)                     # all params wrong
    loss_equal = criterion_equal(pred, target).item()
    loss_kmax2 = criterion_kmax2(pred, target).item()
    assert loss_kmax2 > loss_equal, "Upweighting k_max should increase total loss"
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd scripts/hpc_ensemble
pytest tests/test_loss.py -v
```

Expected: `ModuleNotFoundError: No module named 'loss'`

- [ ] **Step 3: Write `loss.py`**

```python
# scripts/hpc_ensemble/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedGaussianNLL(nn.Module):
    """Weighted Gaussian negative log-likelihood loss for heteroscedastic regression.

    Args:
        weights: per-parameter loss weights, length == n_params.
                 Use [1.0, 2.0, 1.0] to upweight k_max.
    """

    def __init__(self, weights: list):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, n_params, 2) — [..., 0]=μ (Sigmoid-bounded), [..., 1]=log_σ
            target: (B, n_params)    — normalised targets in [0, 1]
        Returns:
            Scalar weighted NLL loss
        """
        mu = pred[..., 0]                               # (B, n_params)
        log_sigma = pred[..., 1]                        # (B, n_params)
        sigma = F.softplus(log_sigma) + 1e-6            # (B, n_params), σ > 0

        # Gaussian NLL: 0.5 * [(y - μ)² / σ² + 2·log(σ)]
        nll = 0.5 * ((target - mu).pow(2) / sigma.pow(2) + 2.0 * torch.log(sigma))  # (B, n_params)
        per_param = nll.mean(dim=0)                     # (n_params,)
        return (self.weights * per_param).sum()         # scalar
```

- [ ] **Step 4: Run all tests — confirm they all pass**

```bash
cd scripts/hpc_ensemble
pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/hpc_ensemble/loss.py scripts/hpc_ensemble/tests/test_loss.py
git commit -m "feat: add WeightedGaussianNLL heteroscedastic loss"
```

---

## Task 5: Training Script

**Files:**
- Create: `scripts/hpc_ensemble/train_member.py`

No unit tests for the training loop itself — it is validated by a smoke run (step 4).

- [ ] **Step 1: Write `train_member.py`**

```python
# scripts/hpc_ensemble/train_member.py
"""Train one ensemble member. Usage:
    python train_member.py --config config.yaml --member-id 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_config, output_dir
from dataset import load_splits
from loss import WeightedGaussianNLL
from model import ResNet50HeteroJoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True, help='Path to config.yaml')
    p.add_argument('--member-id', required=True, type=int, help='Ensemble member index (0-4)')
    return p.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
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
def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total += criterion(model(X), y).item() * len(X)
    return total / len(loader.dataset)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    member_id = args.member_id
    seed = cfg['training']['base_seed'] + member_id
    set_seeds(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[member {member_id}] seed={seed} | device={device}')

    out_dir = output_dir(cfg, member_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_splits(cfg)
    loader_kw = dict(batch_size=cfg['training']['batch_size'], num_workers=0, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    model     = ResNet50HeteroJoint().to(device)
    lw        = cfg['loss_weights']
    criterion = WeightedGaussianNLL([lw['k_min'], lw['k_max'], lw['sigma']]).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['training']['lr'],
                            weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5, min_lr=1e-7)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history        = {'train_loss': [], 'val_loss': []}
    best_val_loss  = float('inf')
    best_ckpt      = out_dir / 'best_model.pt'
    epochs         = cfg['training']['epochs']
    t0             = time.time()

    for epoch in range(epochs):
        tl = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step(vl)

        history['train_loss'].append(float(tl))
        history['val_loss'].append(float(vl))

        lr = optimizer.param_groups[0]['lr']
        print(f'  Ep {epoch+1:3d}/{epochs} | train {tl:.4f} | val {vl:.4f} | lr {lr:.1e}')

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), best_ckpt)

    elapsed = (time.time() - t0) / 60.0
    print(f'[member {member_id}] done in {elapsed:.1f} min — best val {best_val_loss:.4f}')

    with open(out_dir / 'history.json', 'w') as f:
        json.dump({
            'member_id': member_id,
            'seed': seed,
            'best_val_loss': best_val_loss,
            'training_time_min': elapsed,
            'history': history,
        }, f, indent=2)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test the training script (2 epochs, CPU)**

Run from `scripts/hpc_ensemble/`:
```bash
python train_member.py --config config.yaml --member-id 0
```

Before running: temporarily set `training.epochs: 2` and `training.batch_size: 4` in config.yaml, and point `hpc.project_dir` to the project root and `data.file` to the real HDF5 file.

Expected output ends with:
```
[member 0] done in X.X min — best val X.XXXX
```
And `outputs/hpc_ensemble/member_0/best_model.pt` is created.

Restore `epochs: 80` and `batch_size: 64` in config.yaml after smoke test passes.

- [ ] **Step 3: Commit**

```bash
git add scripts/hpc_ensemble/train_member.py
git commit -m "feat: add train_member.py — single ensemble member training"
```

---

## Task 6: Ensemble Evaluation Script

**Files:**
- Create: `scripts/hpc_ensemble/evaluate_ensemble.py`

- [ ] **Step 1: Write `evaluate_ensemble.py`**

```python
# scripts/hpc_ensemble/evaluate_ensemble.py
"""Combine 5 trained ensemble members and evaluate on the test set. Usage:
    python evaluate_ensemble.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from config import load_config, output_dir
from dataset import load_splits, denorm, uncertainty_to_orig
from model import ResNet50HeteroJoint

PARAMS = ['k_min', 'k_max', 'sigma']
COLORS = {'k_min': '#2196F3', 'k_max': '#F44336', 'sigma': '#4CAF50'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    return p.parse_args()


@torch.no_grad()
def predict_member(model: ResNet50HeteroJoint, loader: DataLoader, device) -> np.ndarray:
    """Returns (N, 3, 2): [..., 0]=μ, [..., 1]=log_σ in normalised space."""
    model.eval()
    preds = []
    for X, _ in loader:
        preds.append(model(X.to(device)).cpu())
    return torch.cat(preds).numpy()


def combine_ensemble(member_preds: list) -> dict:
    """Combine list of (N, 3, 2) arrays into ensemble statistics.

    Returns dict with keys: mu, sigma_aleatoric, sigma_epistemic, sigma_total — each (N, 3).
    All values are in normalised [0, 1] space.
    """
    mus        = np.stack([p[..., 0] for p in member_preds])   # (M, N, 3)
    log_sigmas = np.stack([p[..., 1] for p in member_preds])   # (M, N, 3)
    sigmas = np.logaddexp(0, log_sigmas)                        # softplus: log(1+exp(x)), numerically stable (M, N, 3)

    mu_ens          = mus.mean(axis=0)                         # (N, 3)
    sigma_aleatoric = np.sqrt((sigmas ** 2).mean(axis=0))      # (N, 3)
    sigma_epistemic = mus.std(axis=0)                          # (N, 3)
    sigma_total     = np.sqrt(sigma_aleatoric**2 + sigma_epistemic**2)  # (N, 3)

    return {
        'mu':              mu_ens,
        'sigma_aleatoric': sigma_aleatoric,
        'sigma_epistemic': sigma_epistemic,
        'sigma_total':     sigma_total,
    }


def plot_scatter(y_true: dict, y_pred: dict, y_sigma: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, param in zip(axes, PARAMS):
        yt, yp, ys = y_true[param], y_pred[param], y_sigma[param]
        r2  = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        ax.errorbar(yt, yp, yerr=ys, fmt='none', alpha=0.08, color=COLORS[param], elinewidth=0.5)
        ax.scatter(yt, yp, alpha=0.15, s=4, color=COLORS[param], rasterized=True)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Pred {param}')
        ax.set_title(f'{param}\nR²={r2:.4f}  MAE={mae:.3f}')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Predicted vs True — Ensemble (μ ± σ_total)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'scatter.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true: dict, y_pred: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, param in zip(axes, PARAMS):
        res = y_pred[param] - y_true[param]
        ax.hist(res, bins=80, color=COLORS[param], alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(res.mean(), color='red', linestyle='--', linewidth=1,
                   label=f'mean={res.mean():.3f}  std={res.std():.3f}')
        ax.set_xlabel(f'Residual (pred − true {param})')
        ax.set_ylabel('Count')
        ax.set_title(param)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Residual Distributions — Ensemble', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'residuals.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration(y_true: dict, y_pred: dict, y_sigma: dict, out_dir: Path):
    """Reliability diagram: observed coverage vs expected confidence level."""
    confidence_levels = np.linspace(0.05, 0.95, 19)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, param in zip(axes, PARAMS):
        yt, yp, ys = y_true[param], y_pred[param], y_sigma[param]
        observed = []
        for conf in confidence_levels:
            z = scipy.stats.norm.ppf((1.0 + conf) / 2.0)
            observed.append((np.abs(yt - yp) <= z * ys).mean())
        ax.plot(confidence_levels, observed, 'o-', color=COLORS[param], markersize=4, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
        ax.set_xlabel('Expected confidence')
        ax.set_ylabel('Observed coverage')
        ax.set_title(param)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Uncertainty Calibration — Reliability Diagram', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'uncertainty_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_members = cfg['ensemble']['n_members']
    base_dir  = output_dir(cfg)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load test set
    _, _, test_ds = load_splits(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'],
                             shuffle=False, num_workers=0)
    y_true_norm = test_ds.labels  # (N, 3)

    # Collect predictions from all members
    member_preds = []
    for mid in range(n_members):
        ckpt = output_dir(cfg, mid) / 'best_model.pt'
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
        model = ResNet50HeteroJoint().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        member_preds.append(predict_member(model, test_loader, device))
        print(f'  Loaded member {mid}')

    # Combine
    ens = combine_ensemble(member_preds)  # all (N, 3), normalised space

    # Denormalise to original units
    y_true = denorm(y_true_norm, cfg)
    y_pred = denorm(ens['mu'], cfg)
    y_sigma_total     = uncertainty_to_orig(ens['sigma_total'],     ens['mu'], cfg)
    y_sigma_aleatoric = uncertainty_to_orig(ens['sigma_aleatoric'], ens['mu'], cfg)
    y_sigma_epistemic = uncertainty_to_orig(ens['sigma_epistemic'], ens['mu'], cfg)

    # Metrics
    metrics = {}
    for i, param in enumerate(PARAMS):
        metrics[param] = {
            'mae':              float(mean_absolute_error(y_true[param], y_pred[param])),
            'rmse':             float(np.sqrt(mean_squared_error(y_true[param], y_pred[param]))),
            'r2':               float(r2_score(y_true[param], y_pred[param])),
            'mean_sigma_total': float(y_sigma_total[param].mean()),
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Ensemble ({n_members} members) — test set performance")
    print(f"{'='*60}")
    print(f"{'PARAM':<8} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'σ_total':>10}")
    print(f"{'-'*60}")
    for param, m in metrics.items():
        print(f"{param:<8} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.4f} {m['mean_sigma_total']:>10.4f}")
    print(f"{'='*60}")

    # Save JSON results
    results = {
        'metrics': metrics,
        'n_members': n_members,
        'ensemble': {
            'mu':              ens['mu'].tolist(),
            'sigma_total':     ens['sigma_total'].tolist(),
            'sigma_aleatoric': ens['sigma_aleatoric'].tolist(),
            'sigma_epistemic': ens['sigma_epistemic'].tolist(),
        },
        'y_true_norm': y_true_norm.tolist(),
    }
    with open(base_dir / 'results.json', 'w') as f:
        json.dump(results, f)

    # Write human-readable summary
    with open(base_dir / 'results_summary.txt', 'w') as f:
        f.write(f"Ensemble ({n_members} members) — test set\n")
        f.write(f"{'='*50}\n")
        f.write(f"{'PARAM':<8} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'σ_total':>10}\n")
        for param, m in metrics.items():
            f.write(f"{param:<8} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
                    f"{m['r2']:>8.4f} {m['mean_sigma_total']:>10.4f}\n")

    # Plots
    plot_scatter(y_true, y_pred, y_sigma_total, base_dir)
    plot_residuals(y_true, y_pred, base_dir)
    plot_calibration(y_true, y_pred, y_sigma_total, base_dir)
    print(f'Results saved to {base_dir}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/hpc_ensemble/evaluate_ensemble.py
git commit -m "feat: add evaluate_ensemble.py — combine 5 members, uncertainty plots"
```

---

## Task 7: SLURM Scripts and Requirements

**Files:**
- Create: `scripts/hpc_ensemble/requirements.txt`
- Create: `scripts/hpc_ensemble/setup_env.sh`
- Create: `scripts/hpc_ensemble/job_ensemble.sh`
- Create: `scripts/hpc_ensemble/job_evaluate.sh`
- Create: `scripts/hpc_ensemble/submit.sh`

- [ ] **Step 1: Write `requirements.txt`**

```
torch>=2.0.0
torchvision>=0.15.0
h5py>=3.8.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
pyyaml>=6.0
astropy>=5.3.0
pytest>=7.0.0
```

- [ ] **Step 2: Write `setup_env.sh`**

```bash
#!/bin/bash
# setup_env.sh — run once on HPC before submitting jobs.
# Creates a conda env named as specified in config.yaml.
# Usage: bash setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="py311"        # CHANGE: match hpc.conda_env in config.yaml

echo "=== Setting up environment: $ENV_NAME ==="

# Load conda (adapt module name to your HPC)
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists — updating packages"
    conda activate "$ENV_NAME"
else
    echo "Creating environment $ENV_NAME with Python 3.11"
    conda create -y -n "$ENV_NAME" python=3.11
    conda activate "$ENV_NAME"
fi

pip install -r "$SCRIPT_DIR/requirements.txt"

echo "=== Setup complete ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

- [ ] **Step 3: Write `job_ensemble.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=fractal_ensemble
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu                  # CHANGE: set in config.yaml hpc.partition
#SBATCH --gres=gpu:1                     # CHANGE: match hpc.gpus_per_task
#SBATCH --cpus-per-task=8               # CHANGE: match hpc.cpus_per_task
#SBATCH --mem=32G                        # CHANGE: match hpc.mem_gb
#SBATCH --time=06:00:00                  # CHANGE: match hpc.time_limit
#SBATCH --output=logs/ensemble_%A_%a.out
#SBATCH --error=logs/ensemble_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=juan.vasconez@yachaytech.edu.ec  # CHANGE: your email

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=================================================="
echo "Ensemble member: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=================================================="

cd "$PROJECT_DIR"
mkdir -p logs

# Load conda — adapt module name to your HPC
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311   # CHANGE: match hpc.conda_env in config.yaml

echo "Python: $(which python)"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

python scripts/hpc_ensemble/train_member.py \
    --config scripts/hpc_ensemble/config.yaml \
    --member-id "${SLURM_ARRAY_TASK_ID}"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
```

- [ ] **Step 4: Write `job_evaluate.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=fractal_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu                  # CHANGE: match hpc.partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juan.vasconez@yachaytech.edu.ec  # CHANGE: your email

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=================================================="
echo "Ensemble evaluation — Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=================================================="

cd "$PROJECT_DIR"

module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311   # CHANGE: match hpc.conda_env in config.yaml

python scripts/hpc_ensemble/evaluate_ensemble.py \
    --config scripts/hpc_ensemble/config.yaml

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
```

- [ ] **Step 5: Write `submit.sh`**

```bash
#!/bin/bash
# submit.sh — professor runs once to launch full ensemble pipeline.
# Usage: bash scripts/hpc_ensemble/submit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"
mkdir -p logs

echo "Submitting ensemble training (5 members as SLURM array)..."
ARRAY_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/job_ensemble.sh")
echo "  Ensemble job ID: $ARRAY_JOB_ID  (members 0–4)"

echo "Submitting evaluation (will run after all 5 members succeed)..."
EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB_ID "$SCRIPT_DIR/job_evaluate.sh")
echo "  Evaluation job ID: $EVAL_JOB_ID"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel $ARRAY_JOB_ID $EVAL_JOB_ID"
echo ""
echo "Results will be written to: outputs/hpc_ensemble/"
```

- [ ] **Step 6: Make scripts executable**

```bash
chmod +x scripts/hpc_ensemble/submit.sh
chmod +x scripts/hpc_ensemble/job_ensemble.sh
chmod +x scripts/hpc_ensemble/job_evaluate.sh
chmod +x scripts/hpc_ensemble/setup_env.sh
```

- [ ] **Step 7: Commit**

```bash
git add scripts/hpc_ensemble/requirements.txt \
        scripts/hpc_ensemble/setup_env.sh \
        scripts/hpc_ensemble/job_ensemble.sh \
        scripts/hpc_ensemble/job_evaluate.sh \
        scripts/hpc_ensemble/submit.sh
git commit -m "feat: add SLURM scripts — job array, auto-eval, submit entrypoint"
```

---

## Task 8: Local Inference Notebook

**Files:**
- Create: `notebooks/comparison/bayesian_ensemble_inference.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/comparison/bayesian_ensemble_inference.ipynb` with the following cells in order.

**Cell 1 — Imports and path setup:**
```python
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Add hpc_ensemble to path so we can reuse model/dataset modules
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'hpc_ensemble'))

from model import ResNet50HeteroJoint
from dataset import denorm, uncertainty_to_orig, load_splits
from config import load_config
from evaluate_ensemble import combine_ensemble, predict_member

CONFIG_PATH   = PROJECT_ROOT / 'scripts' / 'hpc_ensemble' / 'config.yaml'
ENSEMBLE_DIR  = PROJECT_ROOT / 'outputs' / 'hpc_ensemble'
N_MEMBERS     = 5
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = load_config(CONFIG_PATH)
# Override project_dir to local path
cfg['hpc']['project_dir'] = str(PROJECT_ROOT)

print(f'Device: {DEVICE}')
print(f'Ensemble dir: {ENSEMBLE_DIR}')
```

**Cell 2 — Load all 5 checkpoints:**
```python
models = []
for mid in range(N_MEMBERS):
    ckpt = ENSEMBLE_DIR / f'member_{mid}' / 'best_model.pt'
    m = ResNet50HeteroJoint().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    models.append(m)
    print(f'  Loaded member {mid} from {ckpt}')

print(f'\nAll {N_MEMBERS} members loaded.')
```

**Cell 3 — Synthetic test set inference:**
```python
from torch.utils.data import DataLoader

_, _, test_ds = load_splits(cfg)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
y_true_norm = test_ds.labels  # (N, 3)

member_preds = [predict_member(m, test_loader, DEVICE) for m in models]
ens = combine_ensemble(member_preds)

y_true  = denorm(y_true_norm, cfg)
y_pred  = denorm(ens['mu'], cfg)
y_sigma = uncertainty_to_orig(ens['sigma_total'], ens['mu'], cfg)

PARAMS = ['k_min', 'k_max', 'sigma']
COLORS = {'k_min': '#2196F3', 'k_max': '#F44336', 'sigma': '#4CAF50'}

print(f"\n{'='*60}")
print(f"{'PARAM':<8} {'MAE':>8} {'R²':>8} {'σ_total (mean)':>16}")
print(f"{'-'*60}")
for param in PARAMS:
    mae = mean_absolute_error(y_true[param], y_pred[param])
    r2  = r2_score(y_true[param], y_pred[param])
    print(f"{param:<8} {mae:>8.3f} {r2:>8.4f} {y_sigma[param].mean():>16.4f}")
print(f"{'='*60}")

# Scatter with error bars
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, param in zip(axes, PARAMS):
    yt, yp, ys = y_true[param], y_pred[param], y_sigma[param]
    ax.errorbar(yt, yp, yerr=ys, fmt='none', alpha=0.06, color=COLORS[param], elinewidth=0.4)
    ax.scatter(yt, yp, alpha=0.12, s=4, color=COLORS[param], rasterized=True)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlabel(f'True {param}'); ax.set_ylabel(f'Pred {param}')
    ax.set_title(f'{param}  R²={r2_score(yt, yp):.4f}  MAE={mean_absolute_error(yt, yp):.3f}')
    ax.grid(True, alpha=0.3)
plt.suptitle('Ensemble — Synthetic Test Set (μ ± σ_total)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Cell 4 — GASS inference:**
```python
from astropy.io import fits

FITS_PATH = PROJECT_ROOT / 'observational_images' / 'gass_314_-28_1774393621.fits.gz'
IMG_P1  = cfg['data']['img_p1']
IMG_P99 = cfg['data']['img_p99']

def preprocess_image(img2d: np.ndarray) -> torch.Tensor:
    """Normalise a single 2D image and return a (1, 1, H, W) tensor."""
    img = img2d.astype(np.float32)
    img = (img - IMG_P1) / (IMG_P99 - IMG_P1 + 1e-8)
    return torch.from_numpy(img[np.newaxis, np.newaxis])  # (1, 1, H, W)

@torch.no_grad()
def infer_image(img2d: np.ndarray) -> dict:
    """Run ensemble inference on a single 2D image. Returns μ ± σ dicts."""
    x = preprocess_image(img2d).to(DEVICE)
    preds = [m(x).cpu().numpy() for m in models]  # list of (1, 3, 2)
    ens_single = combine_ensemble(preds)           # all (1, 3)
    mu    = denorm(ens_single['mu'],          cfg)
    sigma = uncertainty_to_orig(ens_single['sigma_total'], ens_single['mu'], cfg)
    return {'mu': mu, 'sigma': sigma}

# Load FITS cube
with fits.open(FITS_PATH) as hdul:
    cube = hdul[0].data.astype(np.float32)  # (n_channels, H, W)

print(f'Cube shape: {cube.shape}')
channel_labels = ['Full integration'] + [f'Channel {i+1}' for i in range(cube.shape[0] - 1)]
results_gass = [infer_image(cube[i]) for i in range(cube.shape[0])]

# Bar chart: μ ± σ_total per channel
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(results_gass))
for ax, param in zip(axes, PARAMS):
    mu_vals    = np.array([r['mu'][param][0]    for r in results_gass])
    sigma_vals = np.array([r['sigma'][param][0] for r in results_gass])
    ax.bar(x, mu_vals, yerr=sigma_vals, capsize=4, color=COLORS[param], alpha=0.8, error_kw={'ecolor': 'black'})
    ax.set_xticks(x); ax.set_xticklabels(channel_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(param); ax.set_title(f'{param} — CNN ensemble (μ ± σ_total)')
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('GASS HI Inference — Bayesian Ensemble', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Run the notebook end-to-end locally**

After receiving the trained checkpoints from HPC:
- Place them at `outputs/hpc_ensemble/member_{0..4}/best_model.pt`
- Run all cells top-to-bottom
- Confirm: scatter plots render with error bars, GASS bar chart shows μ ± σ per channel

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/bayesian_ensemble_inference.ipynb
git commit -m "feat: add bayesian_ensemble_inference notebook — local inference with uncertainty"
```

---

## Task 9: Final Integration Check

- [ ] **Step 1: Run full test suite**

```bash
cd scripts/hpc_ensemble
pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 2: Verify config.yaml HPC fields are clearly marked**

Open `scripts/hpc_ensemble/config.yaml` and confirm every field under `hpc:` has a `# CHANGE:` comment. No other fields should have `# CHANGE:`.

- [ ] **Step 3: Dry-run SLURM submission (parse-only)**

On an HPC login node:
```bash
sbatch --test-only scripts/hpc_ensemble/job_ensemble.sh
```

Expected: `sbatch: Job 0 to start at ... using 1 processors on nodes ... in partition gpu`

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "feat: complete bayesian ensemble package — ready for HPC submission"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| ResNet50 + heteroscedastic head (μ + log_σ per param) | Task 3 |
| Gaussian NLL loss, k_max weighted 2× | Task 4 |
| 5 ensemble members, seed = base_seed + member_id | Task 5 |
| σ_aleatoric + σ_epistemic + σ_total combination | Task 6 |
| config.yaml — HPC fields marked, single source of truth | Task 1 |
| SLURM array --array=0-4, one member per job | Task 7 |
| Auto-triggered evaluation via afterok dependency | Task 7 |
| submit.sh — professor runs once | Task 7 |
| setup_env.sh — one-time env setup | Task 7 |
| scatter.png, residuals.png, calibration.png | Task 6 |
| results.json + results_summary.txt | Task 6 |
| Local inference notebook (synthetic + GASS) | Task 8 |
| μ ± σ_total GASS bar chart | Task 8 |
