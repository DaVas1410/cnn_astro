# MC Dropout Uncertainty Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single self-contained notebook that produces per-prediction Bayesian confidence (error bars) for the v3 joint regressor via MC Dropout, with no retraining, and compares its calibration against the deep ensemble.

**Architecture:** Load the existing `joint_regression_v3/best_model.pt`, reproduce the v3 preprocessing exactly, re-enable only the Dropout layers at inference, run T stochastic forward passes over the held-out test set, and aggregate (denorm-then-mean/std) into μ ± σ. Emit metrics, scatter/residual/calibration plots, and an ensemble-vs-MC-Dropout comparison.

**Tech Stack:** PyTorch, torchvision ResNet50, NumPy, h5py, scikit-learn (metrics), matplotlib, scipy.stats. All already installed in `.venv_py311` (CUDA available).

## Global Constraints

- Deliverable is **notebook-only**: `notebooks/comparison/mc_dropout_uncertainty.ipynb`. No `scripts/` module.
- **No retraining.** Reuse `outputs/comparison/joint_regression_v3/best_model.pt` verbatim.
- Preprocessing MUST match the v3 notebook exactly: `SEED=42`; `IMG_P1`/`IMG_P99` from a seed-42 2000-image subsample; `train_test_split(indices, test_size=0.2, random_state=42)` then `test_size=0.5, random_state=42`; evaluate on the **test** split only.
- Model: `ResNet50Joint4` — `resnet50(weights=None)`, `conv1` replaced for 1-channel input, head `Linear(2048→256)→ReLU→Dropout(0.3)→Linear(256→64)→ReLU→Dropout(0.2)→Linear(64→4)→Sigmoid`.
- Targets (4, in order): `['k_min','k_max','sigma','beta']`. Ranges: KMIN 1.0–62.0, KMAX 3.0–64.0, sigma log-scale `LOG_SIG_LO=log(0.01)`, `LOG_SIG_HI=log(5.0)`, BETA −3.0..−1.0.
- Data: `data/raw/balanced_4param_128x128_100000_kmaxfix.h5`, layout `hf['images']` + `hf['parameters']['{k_min,k_max,sigma,beta}']`.
- Windows console is cp1252: any file with σ/² glyphs must be written `encoding='utf-8'`. Avoid printing σ/² to stdout (use `sigma`/`^2` in prints) to dodge the cp1252 stdout crash.
- Ensemble baseline to compare against (test set): k_min R²=0.9922 MAE=0.936 σ=2.8243; k_max R²=0.8116 MAE=4.189 σ=6.5759; sigma R²=0.9800 MAE=0.178 σ=0.4204.
- Output artifacts → `outputs/comparison/mc_dropout/`.
- `PASSES=50`. Config exposes `TEST_SUBSET` (int or None); keep it small (e.g. 500) while building, set to `None` for the final full run.

---

### Task 1: Notebook scaffold + config cell

**Files:**
- Create: `notebooks/comparison/mc_dropout_uncertainty.ipynb`

**Interfaces:**
- Produces (module-level names used by later cells): `PROJECT_ROOT`, `DATA_FILE`, `V3_DIR` (Path to v3 outputs), `OUTPUT_DIR`, `SEED`, `PASSES`, `TEST_SUBSET`, `BATCH_SIZE`, all target range constants (`KMIN_LO`…`BETA_HI`, `LOG_SIG_LO`, `LOG_SIG_HI`), `TARGET_NAMES`, `N_TARGETS`, `DEVICE`.

- [ ] **Step 1: Create the notebook with a markdown title cell and the config code cell**

Markdown cell:
```markdown
# MC Dropout Uncertainty — v3 Joint Regressor
Post-hoc Bayesian confidence (μ ± σ) on the existing `joint_regression_v3` model, no retraining. Second uncertainty method to cross-check the deep ensemble.
```

Code cell:
```python
# Cell 1 — Imports & config
from pathlib import Path
import json, sys
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_FILE  = PROJECT_ROOT / 'data' / 'raw' / 'balanced_4param_128x128_100000_kmaxfix.h5'
V3_DIR     = PROJECT_ROOT / 'outputs' / 'comparison' / 'joint_regression_v3'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'comparison' / 'mc_dropout'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED       = 42
BATCH_SIZE = 64
PASSES     = 50        # MC Dropout stochastic forward passes
TEST_SUBSET = 500      # set to None for the full test set (final run)

KMIN_LO, KMIN_HI = 1.0,  62.0
KMAX_LO, KMAX_HI = 3.0,  64.0
LOG_SIG_LO = float(np.log(0.01))
LOG_SIG_HI = float(np.log(5.0))
BETA_LO, BETA_HI = -3.0, -1.0

TARGET_NAMES = ['k_min', 'k_max', 'sigma', 'beta']
N_TARGETS    = len(TARGET_NAMES)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert DATA_FILE.exists(), f'Missing dataset: {DATA_FILE}'
assert (V3_DIR / 'best_model.pt').exists(), f'Missing checkpoint: {V3_DIR / "best_model.pt"}'
print(f'Project: {PROJECT_ROOT}')
print(f'Device:  {DEVICE}')
print(f'Passes:  {PASSES}   Test subset: {TEST_SUBSET}')
print(f'Targets: {TARGET_NAMES}')
```

- [ ] **Step 2: Execute the notebook to verify the config cell runs**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: completes with no error; output shows `Device: cuda` (or cpu) and both asserts pass (no `AssertionError`).

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb
git commit -m "feat(mc-dropout): notebook scaffold + config cell"
```

---

### Task 2: Data load, image percentiles, and test split

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (append cells)

**Interfaces:**
- Consumes: config names from Task 1.
- Produces: `n_total`, arrays `kmin,kmax,sigma,beta` (physical), `IMG_P1`, `IMG_P99`, `test_idx` (np.ndarray), class `JointHDF5Dataset4`, `test_ds`, `test_loader`, `y_true_n` (N,4 normalized labels for the test set).

- [ ] **Step 1: Append the data-load + percentiles cell**

```python
# Cell 2 — Load parameters, image percentiles (seed-42, matches v3)
with h5py.File(DATA_FILE, 'r') as hf:
    n_total = hf['images'].shape[0]
    kmin  = hf['parameters']['k_min'][:]
    kmax  = hf['parameters']['k_max'][:]
    sigma = hf['parameters']['sigma'][:]
    beta  = hf['parameters']['beta'][:]

np.random.seed(SEED)
with h5py.File(DATA_FILE, 'r') as hf:
    samp_idx = np.sort(np.random.choice(n_total, size=2000, replace=False))
    samp = hf['images'][samp_idx]
IMG_P1  = float(np.percentile(samp, 1))
IMG_P99 = float(np.percentile(samp, 99))
del samp
print(f'N={n_total:,}  p1={IMG_P1:.4f}  p99={IMG_P99:.4f}  (v3 ref p1=-2.0818 p99=0.9409)')
```

- [ ] **Step 2: Append the dataset + split cell (copied verbatim from v3 for identical preprocessing)**

```python
# Cell 3 — Dataset + test split (identical to v3)
class JointHDF5Dataset4(Dataset):
    def __init__(self, h5_path, indices, kmin_v, kmax_v, sigma_v, beta_v, img_p1, img_p99):
        self.h5_path   = str(h5_path)
        self.indices   = indices
        self.img_p1    = float(img_p1)
        self.img_range = float(img_p99 - img_p1) + 1e-8
        kmin_n = (kmin_v[indices] - KMIN_LO) / (KMIN_HI - KMIN_LO)
        kmax_n = (kmax_v[indices] - KMAX_LO) / (KMAX_HI - KMAX_LO)
        log_sig = np.log(np.clip(sigma_v[indices].astype(np.float64), 1e-9, None))
        sig_n  = (log_sig - LOG_SIG_LO) / (LOG_SIG_HI - LOG_SIG_LO)
        beta_n = (beta_v[indices] - BETA_LO) / (BETA_HI - BETA_LO)
        self.labels = np.stack([kmin_n, kmax_n, sig_n, beta_n], axis=1).astype(np.float32)
        self._h5 = None
    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        img = self._get_h5()['images'][self.indices[idx]].astype(np.float32)
        img = (img - self.img_p1) / self.img_range
        return torch.from_numpy(img[np.newaxis]), torch.from_numpy(self.labels[idx])
    def __del__(self):
        if self._h5 is not None:
            self._h5.close()

indices = np.arange(n_total)
train_idx, tmp = train_test_split(indices, test_size=0.2, random_state=SEED)
val_idx, test_idx = train_test_split(tmp, test_size=0.5, random_state=SEED)
if TEST_SUBSET is not None:
    test_idx = test_idx[:TEST_SUBSET]

test_ds = JointHDF5Dataset4(DATA_FILE, test_idx, kmin_v=kmin, kmax_v=kmax,
                            sigma_v=sigma, beta_v=beta, img_p1=IMG_P1, img_p99=IMG_P99)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)
y_true_n = test_ds.labels  # (N, 4) normalized
print(f'Test images: {len(test_ds):,}  labels shape: {y_true_n.shape}')
```

- [ ] **Step 3: Execute and verify split sizes**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: `p1`/`p99` print close to the v3 reference (-2.08 / 0.94); with `TEST_SUBSET=500`, `Test images: 500  labels shape: (500, 4)`. No error.

- [ ] **Step 4: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb
git commit -m "feat(mc-dropout): data load, percentiles, test split"
```

---

### Task 3: Model definition + checkpoint load

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (append cell)

**Interfaces:**
- Consumes: `DEVICE`, `V3_DIR`, `N_TARGETS`.
- Produces: class `ResNet50Joint4`, loaded `model` (on `DEVICE`, weights from checkpoint).

- [ ] **Step 1: Append the model cell**

```python
# Cell 4 — Model + load v3 checkpoint
class ResNet50Joint4(nn.Module):
    def __init__(self, in_channels=1, n_outputs=N_TARGETS):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_outputs),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.backbone(x)

model = ResNet50Joint4(in_channels=1).to(DEVICE)
state = torch.load(V3_DIR / 'best_model.pt', map_location=DEVICE)
model.load_state_dict(state)
model.eval()
n_drop = sum(isinstance(m, nn.Dropout) for m in model.modules())
print(f'Loaded checkpoint. Params: {sum(p.numel() for p in model.parameters()):,}  Dropout layers: {n_drop}')
```

- [ ] **Step 2: Execute and verify the checkpoint loads with matching keys**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: prints `Dropout layers: 2` and a param count (~24–25M); no `RuntimeError` about missing/unexpected keys (state_dict matches architecture).

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb
git commit -m "feat(mc-dropout): ResNet50Joint4 + checkpoint load"
```

---

### Task 4: Denorm helper + deterministic baseline sanity check

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (append cell)

**Interfaces:**
- Consumes: `model`, `test_loader`, `y_true_n`, range constants, `DEVICE`, `V3_DIR`, `TARGET_NAMES`.
- Produces: `denorm_cols(arr_n)` → dict of physical arrays keyed by target; `y_true` (dict, physical); `y_det` (dict, physical, deterministic point estimate with dropout OFF).

- [ ] **Step 1: Append the denorm + deterministic-pass cell**

```python
# Cell 5 — Denorm helper + deterministic baseline (dropout OFF)
def denorm_cols(arr_n):
    """(N,4) normalized -> dict of physical-unit arrays. Sigma via exp (log-scale)."""
    def _back(col, lo, hi):
        return col * (hi - lo) + lo
    return {
        'k_min': _back(arr_n[:, 0], KMIN_LO, KMIN_HI),
        'k_max': _back(arr_n[:, 1], KMAX_LO, KMAX_HI),
        'sigma': np.exp(_back(arr_n[:, 2], LOG_SIG_LO, LOG_SIG_HI)),
        'beta':  _back(arr_n[:, 3], BETA_LO, BETA_HI),
    }

@torch.no_grad()
def forward_all(model, loader):
    out = []
    for X, _ in loader:
        out.append(model(X.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

model.eval()  # dropout OFF
det_n = forward_all(model, test_loader)
y_true = denorm_cols(y_true_n)
y_det  = denorm_cols(det_n)
print('Deterministic baseline (dropout OFF) R^2:')
for p in TARGET_NAMES:
    print(f'  {p:<6} R2={r2_score(y_true[p], y_det[p]):.4f}  MAE={mean_absolute_error(y_true[p], y_det[p]):.3f}')
```

- [ ] **Step 2: Execute and verify baseline R² is sane**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: on the subset, `k_min` and `sigma` R² are high (≳0.95), `k_max` R² noticeably lower (~0.7–0.85). This confirms the checkpoint + preprocessing reproduce v3 behavior. (Values on a 500-subset will differ slightly from the full-set v3 numbers — that's fine; the check is the *pattern*: k_max clearly worst.)

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb
git commit -m "feat(mc-dropout): denorm helper + deterministic baseline"
```

---

### Task 5: MC Dropout inference (T stochastic passes)

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (append cell)

**Interfaces:**
- Consumes: `model`, `test_loader`, `PASSES`, `DEVICE`, `N_TARGETS`, `len(test_ds)`.
- Produces: `mc_raw` — np.ndarray `(PASSES, N_test, 4)` of normalized predictions.

- [ ] **Step 1: Append the MC Dropout cell**

```python
# Cell 6 — MC Dropout: enable dropout only, run T passes
def enable_dropout(model):
    """Put ONLY nn.Dropout modules in train mode; BatchNorm stays in eval."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

model.eval()
enable_dropout(model)

N_test = len(test_loader.dataset)
mc_raw = np.zeros((PASSES, N_test, N_TARGETS), dtype=np.float32)
with torch.no_grad():
    for t in range(PASSES):
        mc_raw[t] = forward_all(model, test_loader)
        if (t + 1) % 10 == 0:
            print(f'  pass {t+1}/{PASSES}')

# Sanity: dropout should induce nonzero spread across passes
spread = mc_raw.std(axis=0).mean(axis=0)  # (4,)
print('Mean per-target std across passes (normalized):',
      {p: float(f'{s:.4f}') for p, s in zip(TARGET_NAMES, spread)})
assert (spread > 1e-5).all(), 'Dropout not active — zero variance across passes'
```

- [ ] **Step 2: Execute and verify nonzero spread**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: prints progress every 10 passes and per-target std all > 0; the `assert` passes (confirms dropout is actually stochastic at inference).

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb
git commit -m "feat(mc-dropout): T-pass stochastic inference"
```

---

### Task 6: Aggregate (denorm-then-mean/std), metrics, save results

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (append cell)

**Interfaces:**
- Consumes: `mc_raw`, `denorm_cols`, `y_true`, `y_det`, `TARGET_NAMES`, `OUTPUT_DIR`, `PASSES`.
- Produces: `y_pred` (dict μ, physical), `y_sigma` (dict σ, physical), `metrics` (dict), files `results.json`, `results_summary.txt`.

- [ ] **Step 1: Append the aggregate + save cell**

```python
# Cell 7 — Denorm every pass, THEN aggregate; metrics + save
# Physical-space stacks per target: (PASSES, N)
phys_stacks = {p: np.empty((PASSES, N_test), dtype=np.float64) for p in TARGET_NAMES}
for t in range(PASSES):
    d = denorm_cols(mc_raw[t])
    for p in TARGET_NAMES:
        phys_stacks[p][t] = d[p]

y_pred  = {p: phys_stacks[p].mean(axis=0) for p in TARGET_NAMES}   # mu
y_sigma = {p: phys_stacks[p].std(axis=0)  for p in TARGET_NAMES}   # confidence

metrics = {}
for p in TARGET_NAMES:
    metrics[p] = {
        'mae':        float(mean_absolute_error(y_true[p], y_pred[p])),
        'rmse':       float(np.sqrt(mean_squared_error(y_true[p], y_pred[p]))),
        'r2':         float(r2_score(y_true[p], y_pred[p])),
        'mean_sigma': float(y_sigma[p].mean()),
        'r2_deterministic': float(r2_score(y_true[p], y_det[p])),
    }

print(f"{'PARAM':<8}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'meanSig':>10}{'R2_det':>9}")
for p in TARGET_NAMES:
    m = metrics[p]
    print(f"{p:<8}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.4f}{m['mean_sigma']:>10.4f}{m['r2_deterministic']:>9.4f}")

results = {
    'method': 'mc_dropout', 'passes': PASSES, 'n_test': int(N_test),
    'params': TARGET_NAMES, 'metrics': metrics,
    'y_true':  {p: y_true[p].tolist()  for p in TARGET_NAMES},
    'y_pred':  {p: y_pred[p].tolist()  for p in TARGET_NAMES},
    'y_sigma': {p: y_sigma[p].tolist() for p in TARGET_NAMES},
    'y_deterministic': {p: y_det[p].tolist() for p in TARGET_NAMES},
}
with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f)

with open(OUTPUT_DIR / 'results_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f"MC Dropout ({PASSES} passes) — test set (N={N_test})\n")
    f.write(f"{'PARAM':<8}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'mean_sigma':>12}\n")
    for p in TARGET_NAMES:
        m = metrics[p]
        f.write(f"{p:<8}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.4f}{m['mean_sigma']:>12.4f}\n")
print(f'Saved results.json + results_summary.txt to {OUTPUT_DIR}')
```

- [ ] **Step 2: Execute and verify outputs written**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: metrics table prints; `results.json` and `results_summary.txt` exist under `outputs/comparison/mc_dropout/`. `mean_sigma` per target is > 0. `r2` ≈ `r2_deterministic` (small drift from active dropout).

Verify files:
`ls outputs/comparison/mc_dropout/` → shows `results.json`, `results_summary.txt`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb outputs/comparison/mc_dropout/results_summary.txt
git commit -m "feat(mc-dropout): aggregate, metrics, save results"
```

---

### Task 7: Plots — scatter±σ, residuals, calibration

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (append cell)

**Interfaces:**
- Consumes: `y_true`, `y_pred`, `y_sigma`, `TARGET_NAMES`, `OUTPUT_DIR`.
- Produces: files `scatter.png`, `residuals.png`, `calibration.png`.

- [ ] **Step 1: Append the plotting cell**

```python
# Cell 8 — Scatter (mu±sigma), residuals, calibration
COLORS = {'k_min': '#2196F3', 'k_max': '#F44336', 'sigma': '#4CAF50', 'beta': '#9C27B0'}

# Scatter with error bars
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for ax, p in zip(axes, TARGET_NAMES):
    yt, yp, ys = y_true[p], y_pred[p], y_sigma[p]
    ax.errorbar(yt, yp, yerr=ys, fmt='none', alpha=0.10, color=COLORS[p], elinewidth=0.5)
    ax.scatter(yt, yp, s=5, alpha=0.20, color=COLORS[p], rasterized=True)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlabel(f'True {p}'); ax.set_ylabel(f'Pred {p}')
    ax.set_title(f'{p}  R2={r2_score(yt, yp):.4f}  MAE={mean_absolute_error(yt, yp):.3f}')
    ax.grid(True, alpha=0.3)
plt.suptitle('MC Dropout — Predicted vs True (mu +/- sigma)', fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'scatter.png', dpi=150, bbox_inches='tight'); plt.show()

# Residuals
fig, axes = plt.subplots(1, 4, figsize=(19, 4))
for ax, p in zip(axes, TARGET_NAMES):
    res = y_pred[p] - y_true[p]
    ax.hist(res, bins=60, color=COLORS[p], alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvline(res.mean(), color='red', ls='--', lw=1, label=f'mean={res.mean():.3f} std={res.std():.3f}')
    ax.set_xlabel(f'Residual {p}'); ax.set_ylabel('count'); ax.set_title(p); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('MC Dropout — Residuals', fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'residuals.png', dpi=150, bbox_inches='tight'); plt.show()

# Calibration / reliability diagram
conf_levels = np.linspace(0.05, 0.95, 19)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, p in zip(axes, TARGET_NAMES):
    yt, yp, ys = y_true[p], y_pred[p], y_sigma[p]
    observed = [ (np.abs(yt - yp) <= scipy.stats.norm.ppf((1+c)/2) * ys).mean() for c in conf_levels ]
    ax.plot(conf_levels, observed, 'o-', color=COLORS[p], ms=4, label='Model')
    ax.plot([0,1],[0,1],'k--', lw=1, label='Perfect')
    ax.set_xlabel('Expected confidence'); ax.set_ylabel('Observed coverage'); ax.set_title(p); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('MC Dropout — Calibration (reliability diagram)', fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'calibration.png', dpi=150, bbox_inches='tight'); plt.show()
print('Saved scatter.png, residuals.png, calibration.png')
```

- [ ] **Step 2: Execute and verify the three PNGs are written**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: no error; `ls outputs/comparison/mc_dropout/` shows `scatter.png`, `residuals.png`, `calibration.png`. Calibration curves may sit below the diagonal (under-confidence) — that is an expected finding, not an error.

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb
git commit -m "feat(mc-dropout): scatter, residual, calibration plots"
```

---

### Task 8: Comparison vs the deep ensemble

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (append cell)

**Interfaces:**
- Consumes: `metrics`, `y_true`, `y_pred`, `y_sigma`, `OUTPUT_DIR`.
- Produces: file `comparison.png`; printed side-by-side table.

- [ ] **Step 1: Append the comparison cell**

```python
# Cell 9 — MC Dropout vs Deep Ensemble (shared targets: k_min, k_max, sigma)
ENSEMBLE = {  # from the Colab ensemble run (test set)
    'k_min': {'mae': 0.936, 'rmse': 1.270, 'r2': 0.9922, 'sigma': 2.8243},
    'k_max': {'mae': 4.189, 'rmse': 6.266, 'r2': 0.8116, 'sigma': 6.5759},
    'sigma': {'mae': 0.178, 'rmse': 0.203, 'r2': 0.9800, 'sigma': 0.4204},
}
shared = ['k_min', 'k_max', 'sigma']

print(f"{'param':<7}{'method':<10}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'meanSig':>10}")
for p in shared:
    e, m = ENSEMBLE[p], metrics[p]
    print(f"{p:<7}{'ensemble':<10}{e['mae']:>9.3f}{e['rmse']:>9.3f}{e['r2']:>9.4f}{e['sigma']:>10.4f}")
    print(f"{'':<7}{'mc_dropout':<10}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.4f}{m['mean_sigma']:>10.4f}")

# Overlaid calibration for the shared targets
conf_levels = np.linspace(0.05, 0.95, 19)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, p in zip(axes, shared):
    yt, yp, ys = y_true[p], y_pred[p], y_sigma[p]
    observed = [ (np.abs(yt - yp) <= scipy.stats.norm.ppf((1+c)/2) * ys).mean() for c in conf_levels ]
    ax.plot(conf_levels, observed, 'o-', color='#F44336', ms=4, label='MC Dropout')
    ax.plot([0,1],[0,1],'k--', lw=1, label='Perfect')
    ax.set_xlabel('Expected confidence'); ax.set_ylabel('Observed coverage')
    ax.set_title(f'{p}  (MC R2={metrics[p]["r2"]:.3f} vs Ens {ENSEMBLE[p]["r2"]:.3f})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('MC Dropout calibration vs Deep Ensemble R2', fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'comparison.png', dpi=150, bbox_inches='tight'); plt.show()
print('Saved comparison.png')
```

- [ ] **Step 2: Execute and verify the comparison output**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: side-by-side table prints both methods for k_min/k_max/sigma; `comparison.png` written. No error.

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb outputs/comparison/mc_dropout/
git commit -m "feat(mc-dropout): ensemble-vs-MC-dropout comparison"
```

---

### Task 9: Final full-test-set run

**Files:**
- Modify: `notebooks/comparison/mc_dropout_uncertainty.ipynb` (config edit + full execute)

- [ ] **Step 1: Set `TEST_SUBSET = None` in Cell 1**

Change `TEST_SUBSET = 500` to `TEST_SUBSET = None` so the final artifacts use the full ~10k test set.

- [ ] **Step 2: Execute the full notebook end-to-end**

Run: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/comparison/mc_dropout_uncertainty.ipynb`
Expected: completes (a few minutes on GPU; longer on CPU). `results_summary.txt` shows full-set metrics; k_min/sigma R² high, k_max ~0.8. All five PNGs regenerated at full resolution.

- [ ] **Step 3: Verify final artifacts**

Run: `ls outputs/comparison/mc_dropout/`
Expected: `results.json`, `results_summary.txt`, `scatter.png`, `residuals.png`, `calibration.png`, `comparison.png`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/comparison/mc_dropout_uncertainty.ipynb outputs/comparison/mc_dropout/
git commit -m "feat(mc-dropout): final full-test-set uncertainty run"
```

---

## Notes for the implementer

- Execute the notebook with `jupyter nbconvert --to notebook --execute --inplace <path>` after each task; the notebook is cumulative, so a clean full run is the test. Keep `TEST_SUBSET=500` until Task 9 to keep iterations fast.
- If `jupyter` is not on PATH, use `python -m jupyter nbconvert ...` inside `.venv_py311`.
- Do NOT print σ or ² to stdout on Windows (cp1252 crash) — the cells use `sigma`/`R2` in prints and `encoding='utf-8'` for the summary file. Keep it that way.
- The deterministic baseline (Task 4) is the key correctness anchor: if k_min/sigma R² are not high and k_max clearly lowest, the checkpoint or preprocessing is wrong — stop and fix before proceeding.
