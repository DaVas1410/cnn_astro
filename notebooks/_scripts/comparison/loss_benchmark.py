# AUTO-GENERATED from notebooks/comparison/loss_benchmark.ipynb — do not edit by hand
#!/usr/bin/env python
# coding: utf-8

# # Loss Function Benchmark — k_min, k_max, sigma
# Compare MAE vs RMSE vs 0.5·RMSE+0.5·MAE for predicting fractal parameters.

# In[1]:


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
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

raw_dir = PROJECT_ROOT / 'data' / 'raw'

if 'H5_PATH' in globals() and H5_PATH is not None and Path(H5_PATH).exists():
    DATA_FILE = Path(H5_PATH)
else:
    candidates = sorted(
        raw_dir.glob('flexible_kmax_*.h5'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    DATA_FILE = candidates[0] if candidates else None
OUTPUT_BASE = PROJECT_ROOT / 'outputs' / 'comparison' / 'loss_benchmark'
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

assert DATA_FILE is not None and DATA_FILE.exists(), \
    f"Dataset not found.\nLooked in: {PROJECT_ROOT / 'data' / 'raw'}"
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

# ── Targets (k_min and k_max) ──────────────────────────────────────────────
# Dataset: k_max ~ U[5, 64] independent; k_min ~ U[1, k_max-2] conditional
# So k_min range is [1, 62] and k_max range is [5, 64].
TARGETS_P2 = {
    'k_min': {'param_key': 'k_min', 'param_min': 1.0,  'param_max': 62.0},
    'k_max': {'param_key': 'k_max', 'param_min': 5.0,  'param_max': 64.0},
}


# In[2]:


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
    'rmse_mae': '0.5\u00b7RMSE + 0.5\u00b7MAE',
}
LOSS_COLORS = {'mae': '#2196F3', 'rmse': '#F44336', 'rmse_mae': '#4CAF50'}

print("Shared code loaded.")
print(f"ResNet50 params: {sum(p.numel() for p in ResNet50Regressor().parameters()):,}")


# In[3]:


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
            print(f'[SKIP] {target_name}/{loss_name}  \u2014  results.json already exists')
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

        print(f'\n  Test \u2014 MAE:{mae_:.3f}  RMSE:{rmse_:.3f}  R\u00b2:{r2_:.4f}')
        print(f'  Saved \u2192 {results_path}')

        del model, train_ds, val_ds, test_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print('\nPhase 2 benchmark complete.')


# In[4]:


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


# In[5]:


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

plt.suptitle('Training Curves \u2014 val loss per epoch', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[6]:


# Cell 6 — Plot 2: Final test metrics bar chart + summary table
metrics     = ['test_mae', 'test_rmse', 'test_r2']
metric_lbls = ['MAE', 'RMSE', 'R\u00b2']
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
_r2_hdr = 'R²'
print(f"{'TARGET':<10} {'LOSS':<25} {'MAE':>7} {'RMSE':>7} {_r2_hdr:>8} {'min':>7}")
print(f"{'='*72}")
for target in TARGETS_P2_DONE:
    for ln in loss_names:
        r = results_p2[target].get(ln)
        if r:
            print(f"{target:<10} {LOSS_LABELS[ln]:<25} "
                  f"{r['test_mae']:>7.3f} {r['test_rmse']:>7.3f} "
                  f"{r['test_r2']:>8.4f} {r['training_time_min']:>7.1f}")
print(f"{'='*72}")


# In[7]:


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
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}\nR\u00b2={r["test_r2"]:.4f}')
        else:
            ax.text(0.5, 0.5, 'not run', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}')
        ax.set_xlabel(f'True {target}'); ax.set_ylabel(f'Pred {target}')
        ax.grid(True, alpha=0.3)

plt.suptitle('Predicted vs True', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()


# In[8]:


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
        ax.set_xlabel('Residual (pred \u2212 true)'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Residual Distributions', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()


# In[9]:


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


# In[10]:


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


# In[11]:


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
        print(f'[SKIP] sigma/{loss_name}  \u2014  results.json already exists')
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

    print(f'\n  Test \u2014 MAE:{mae_:.3f}  RMSE:{rmse_:.3f}  R\u00b2:{r2_:.4f}')
    print(f'  Saved \u2192 {results_path}')

    del model, train_ds, val_ds, test_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('\nPhase 3 (sigma) benchmark complete.')


# In[ ]:


# Cell 12 — Load all results: k_min, k_max, sigma
ALL_TARGETS = ['k_min', 'k_max', 'sigma']
results_all = load_results(ALL_TARGETS)   # reuses load_results() from Cell 4
ALL_DONE    = [t for t in ALL_TARGETS if results_all.get(t)]
print(f"Results available for: {ALL_DONE}")


# In[ ]:


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
plt.suptitle('Training Curves \u2014 all targets', fontsize=14, fontweight='bold')
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
plt.suptitle('Test Metrics \u2014 all targets', fontsize=14, fontweight='bold')
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
            ax.set_title(f'{target}  |  {LOSS_LABELS[ln]}\nR\u00b2={r["test_r2"]:.4f}')
        ax.set_xlabel(f'True {target}'); ax.set_ylabel(f'Pred {target}')
        ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs True \u2014 all targets', fontsize=14, fontweight='bold')
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
        ax.set_xlabel('Residual (pred \u2212 true)'); ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
plt.suptitle('Residual Distributions \u2014 all targets', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()


# In[ ]:


# Cell 14 — Full summary table (all targets × all losses)
print(f"\n{'='*75}")
_r2_hdr = 'R²'
print(f"{'TARGET':<10} {'LOSS':<25} {'MAE':>8} {'RMSE':>8} {_r2_hdr:>9} {'min':>7}")
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

