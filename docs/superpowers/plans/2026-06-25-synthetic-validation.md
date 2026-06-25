# Synthetic Validation (Classical vs CNN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/comparison/synthetic_validation.ipynb` that runs classical (FFT) and CNN inference on held-out synthetic images and compares both against ground truth, proving the CNN's near-exact in-distribution accuracy and diagnosing k_max/sigma error.

**Architecture:** A single Jupyter notebook assembled cell-by-cell with `nbformat`. Each task appends one or more cells, then executes the whole notebook headlessly via `jupyter nbconvert --execute` with a small `SYNTH_VAL_N_EVAL` to keep the test cycle fast. The notebook reuses the trained `ResNet50Joint4` checkpoint, the exact normalization from `joint_regression_v2.ipynb`, and the `fourier_params()` classical estimator from `io-fits.ipynb`.

**Tech Stack:** Python, PyTorch, torchvision (ResNet50), h5py, numpy, scipy.stats, scikit-learn, matplotlib, nbformat, jupyter (nbconvert).

## Global Constraints

- Notebook path: `notebooks/comparison/synthetic_validation.ipynb` (do not create a separate `.py` module — notebook is the deliverable).
- Output dir: `outputs/comparison/synthetic_validation/`.
- Dataset: `data/raw/balanced_4param_128x128_100000.h5` (layout: `/images` (N,128,128), `/parameters/{k_min,k_max,sigma,beta}`).
- CNN checkpoint: `outputs/comparison/joint_regression_v2/best_model.pt`.
- Normalization constants (verbatim): `KMIN_LO,KMIN_HI = 1.0, 62.0` · `KMAX_LO,KMAX_HI = 3.0, 64.0` · `LOG_SIG_LO,LOG_SIG_HI = log(0.01), log(5.0)` · `BETA_LO,BETA_HI = -3.0, -1.0`.
- `IMG_P1`/`IMG_P99` are READ from `outputs/comparison/joint_regression_v2/results.json` keys `img_p1`/`img_p99` (do NOT recompute — avoids seed-sample mismatch).
- Split (verbatim, matches the trained model): `SEED = 42`; `train_idx, tmp = train_test_split(np.arange(n_total), test_size=0.2, random_state=42)`; `val_idx, test_idx = train_test_split(tmp, test_size=0.5, random_state=42)`.
- Eval subset size from env: `N_EVAL = int(os.environ.get('SYNTH_VAL_N_EVAL', 2000))`, capped at `len(test_idx)`, first N of `test_idx`.
- Classical method targets k_min/k_max only; sigma is CNN-only.
- **Interpreter:** use the py311 ML venv `.venv_py311/Scripts/python.exe` (torch 2.6.0+cu124, CUDA available, h5py, scipy, sklearn, torchvision, nbformat, nbconvert, ipykernel all present). Do NOT use `uv run` — the project's default uv env lacks torch.
  - Python: `.venv_py311/Scripts/python.exe -c "..."`
  - Notebook execution: `.venv_py311/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace <nb>` (the notebook's `python3` kernelspec resolves to this venv).
  - Set `SYNTH_VAL_N_EVAL` in PowerShell as `$env:SYNTH_VAL_N_EVAL='64'; <cmd>` (Bash-style `VAR=x cmd` does not work in PowerShell). The Bash tool is also available if preferred.

---

### Task 1: Scaffold notebook — title + setup cell

**Files:**
- Create: `notebooks/comparison/synthetic_validation.ipynb`
- Create (helper, throwaway): `scripts/_build_synth_val.py` (cell-builder; committed with the notebook so the build is reproducible)

**Interfaces:**
- Produces: a notebook whose first code cell defines `PROJECT_ROOT, DEVICE, N_EVAL`, all normalization constants, `IMG_P1, IMG_P99`, `DATA_FILE`, `CKPT`, `OUT_DIR`, and asserts the dataset + checkpoint exist.

- [ ] **Step 1: Create the notebook with a title cell and the setup cell using nbformat**

Create `scripts/_build_synth_val.py`:

```python
"""Builds notebooks/comparison/synthetic_validation.ipynb cell-by-cell.
Re-runnable: rewrites the notebook from the CELLS list below.
Each plan task appends its cell sources here, then runs this script."""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / 'notebooks' / 'comparison' / 'synthetic_validation.ipynb'

CELLS = []

def md(src):  CELLS.append(('md', src))
def code(src): CELLS.append(('code', src))

# ── Title ────────────────────────────────────────────────────────────────
md("""# Synthetic Validation — Classical (FFT) vs CNN

Runs the trained non-Bayesian joint CNN (`ResNet50Joint4`) and the classical
azimuthal power-spectrum estimator on held-out **synthetic** test images, and
compares both to ground truth. In-distribution synthetic images should yield
near-exact CNN predictions. Classical covers k_min/k_max only; sigma is CNN-only.
""")

# ── Cell 1: imports, paths, constants ────────────────────────────────────
code('''# Setup — imports, paths, constants
import os, json, sys
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'balanced_4param_128x128_100000.h5'
CKPT      = PROJECT_ROOT / 'outputs' / 'comparison' / 'joint_regression_v2' / 'best_model.pt'
RESULTS   = PROJECT_ROOT / 'outputs' / 'comparison' / 'joint_regression_v2' / 'results.json'
OUT_DIR   = PROJECT_ROOT / 'outputs' / 'comparison' / 'synthetic_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_FILE.exists(), f'Dataset not found: {DATA_FILE}'
assert CKPT.exists(),      f'Checkpoint not found: {CKPT}'
assert RESULTS.exists(),   f'results.json not found: {RESULTS}'

# Normalization constants — verbatim from joint_regression_v2.ipynb
KMIN_LO, KMIN_HI = 1.0,  62.0
KMAX_LO, KMAX_HI = 3.0,  64.0
LOG_SIG_LO, LOG_SIG_HI = float(np.log(0.01)), float(np.log(5.0))
BETA_LO, BETA_HI = -3.0, -1.0
SEED = 42

# img_p1/p99 read from the trained run (NOT recomputed)
_res    = json.load(open(RESULTS))
IMG_P1  = float(_res['img_p1'])
IMG_P99 = float(_res['img_p99'])
IMG_RANGE = (IMG_P99 - IMG_P1) + 1e-8

N_EVAL = int(os.environ.get('SYNTH_VAL_N_EVAL', 2000))
print(f'Device={DEVICE}  IMG_P1={IMG_P1:.4f}  IMG_P99={IMG_P99:.4f}  N_EVAL={N_EVAL}')
''')

def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s)
                for t, s in CELLS]
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    NB_PATH.write_text(nbf.writes(nb), encoding='utf-8')
    print(f'Wrote {len(nb.cells)} cells -> {NB_PATH}')

if __name__ == '__main__':
    build()
```

- [ ] **Step 2: Build the notebook**

Run: `uv run python scripts/_build_synth_val.py`
Expected: `Wrote 2 cells -> .../synthetic_validation.ipynb`

- [ ] **Step 3: Execute headlessly to verify the setup cell runs**

Run: `SYNTH_VAL_N_EVAL=64 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/synthetic_validation.ipynb`
Expected: completes with no exception; the executed cell prints `Device=...  IMG_P1=-2.0854  IMG_P99=0.9466  N_EVAL=64`.

- [ ] **Step 4: Commit**

```bash
git add scripts/_build_synth_val.py notebooks/comparison/synthetic_validation.ipynb
git commit -m "feat: scaffold synthetic_validation notebook — setup cell"
```

---

### Task 2: Load held-out test images + ground truth

**Files:**
- Modify: `scripts/_build_synth_val.py` (append cell), regenerate notebook.

**Interfaces:**
- Consumes: `DATA_FILE, SEED, N_EVAL, IMG_P1, IMG_RANGE` from Task 1.
- Produces: `eval_idx` (np.ndarray, length ≤ N_EVAL), `imgs_raw` (N_EVAL,128,128 float32 — stored log10 image, for classical FFT), `imgs_norm` (N_EVAL,1,128,128 float32 — p1/p99 normalized, for CNN), and `gt` dict with keys `k_min,k_max,sigma,beta` (each float64 array length N_EVAL).

- [ ] **Step 1: Append the data-loading cell to the builder**

Append to `CELLS` in `scripts/_build_synth_val.py` (before `def build()`):

```python
code('''# Load held-out synthetic test images + ground truth (same split as training)
with h5py.File(DATA_FILE, 'r') as hf:
    n_total = hf['images'].shape[0]
    kmin_all  = hf['parameters/k_min'][:]
    kmax_all  = hf['parameters/k_max'][:]
    sigma_all = hf['parameters/sigma'][:]
    beta_all  = hf['parameters/beta'][:]

indices = np.arange(n_total)
train_idx, tmp = train_test_split(indices, test_size=0.2, random_state=SEED)
val_idx, test_idx = train_test_split(tmp,   test_size=0.5, random_state=SEED)

# Honesty check: eval indices must not appear in the training split
assert len(np.intersect1d(test_idx, train_idx)) == 0

eval_idx = np.sort(test_idx[:min(N_EVAL, len(test_idx))])

with h5py.File(DATA_FILE, 'r') as hf:
    imgs_raw = hf['images'][eval_idx].astype(np.float32)        # (N,128,128) stored log10

imgs_norm = ((imgs_raw - IMG_P1) / IMG_RANGE)[:, None, :, :]    # (N,1,128,128)

gt = {
    'k_min': kmin_all[eval_idx].astype(np.float64),
    'k_max': kmax_all[eval_idx].astype(np.float64),
    'sigma': sigma_all[eval_idx].astype(np.float64),
    'beta':  beta_all[eval_idx].astype(np.float64),
}
print(f'eval images: {imgs_raw.shape}   test split size: {len(test_idx):,}')
print(f'gt k_max range: [{gt["k_max"].min():.2f}, {gt["k_max"].max():.2f}]  '
      f'sigma range: [{gt["sigma"].min():.3f}, {gt["sigma"].max():.3f}]')
''')
```

- [ ] **Step 2: Rebuild + execute**

Run: `uv run python scripts/_build_synth_val.py && SYNTH_VAL_N_EVAL=64 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/synthetic_validation.ipynb`
Expected: no exception; prints `eval images: (64, 128, 128)   test split size: 10,000` and sane gt ranges (k_max within [3,64], sigma within [0.01,5]).

- [ ] **Step 3: Commit**

```bash
git add scripts/_build_synth_val.py notebooks/comparison/synthetic_validation.ipynb
git commit -m "feat: load held-out synthetic test images + ground truth"
```

---

### Task 3: CNN inference

**Files:**
- Modify: `scripts/_build_synth_val.py` (append cells), regenerate notebook.

**Interfaces:**
- Consumes: `imgs_norm, DEVICE, CKPT` and normalization constants from earlier tasks.
- Produces: `ResNet50Joint4` class; `cnn_pred` dict with keys `k_min,k_max,sigma,beta` (each float64 array length N_EVAL) in physical units.

- [ ] **Step 1: Append the model definition + inference cells**

Append to `CELLS`:

```python
code('''# CNN model — identical architecture to joint_regression_v2.ipynb
class ResNet50Joint4(nn.Module):
    def __init__(self, in_channels=1, n_outputs=4):
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

def _back(col, lo, hi):
    return col * (hi - lo) + lo

def denorm_preds(preds_n):
    """Map (N,4) Sigmoid outputs in [0,1] back to physical units."""
    return {
        'k_min': _back(preds_n[:, 0], KMIN_LO, KMIN_HI),
        'k_max': _back(preds_n[:, 1], KMAX_LO, KMAX_HI),
        'sigma': np.exp(_back(preds_n[:, 2], LOG_SIG_LO, LOG_SIG_HI)),
        'beta':  _back(preds_n[:, 3], BETA_LO, BETA_HI),
    }
''')

code('''# Run CNN inference on the eval images
model = ResNet50Joint4(in_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

preds_n = []
with torch.no_grad():
    for i in range(0, len(imgs_norm), 256):
        batch = torch.from_numpy(imgs_norm[i:i+256]).to(DEVICE)
        preds_n.append(model(batch).cpu().numpy())
preds_n = np.concatenate(preds_n, axis=0)            # (N,4) in [0,1]
cnn_pred = {k: v.astype(np.float64) for k, v in denorm_preds(preds_n).items()}
print('CNN pred k_max[:5]:', np.round(cnn_pred['k_max'][:5], 2))
print('CNN pred sigma[:5]:', np.round(cnn_pred['sigma'][:5], 3))
''')
```

- [ ] **Step 2: Rebuild + execute**

Run: `uv run python scripts/_build_synth_val.py && SYNTH_VAL_N_EVAL=64 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/synthetic_validation.ipynb`
Expected: no exception; `cnn_pred` k_max values fall within ~[3,64] and sigma within ~[0.01,5].

- [ ] **Step 3: Commit**

```bash
git add scripts/_build_synth_val.py notebooks/comparison/synthetic_validation.ipynb
git commit -m "feat: add CNN inference cells to synthetic validation"
```

---

### Task 4: Classical FFT inference + NaN handling

**Files:**
- Modify: `scripts/_build_synth_val.py` (append cells), regenerate notebook.

**Interfaces:**
- Consumes: `imgs_raw` from Task 2.
- Produces: `fourier_params(img2d, k_lo_frac, k_hi_frac, thr)` (returns `{'alpha','k_min','k_max'}`); `classical_pred` dict with keys `k_min,k_max` (float64 arrays length N_EVAL, may contain NaN); `classical_valid` boolean mask (True where both k_min and k_max are finite); `n_dropped` int.

- [ ] **Step 1: Append the classical estimator + run cells**

Append to `CELLS`:

```python
code('''# Classical estimator — azimuthal power-spectrum (reused from io-fits.ipynb)
from numpy.fft import fft2, fftshift
from scipy.stats import linregress as _lr

def fourier_params(img2d, k_lo_frac=0.05, k_hi_frac=0.70, thr=0.5):
    """Spectral index alpha + k_min/k_max from the azimuthal power spectrum.
    Returns NaNs when the inertial-range fit has < 3 usable points."""
    F = fftshift(fft2(img2d.astype(np.float64)))
    P = np.abs(F) ** 2
    ny, nx = P.shape
    cy, cx = ny // 2, nx // 2
    y_i, x_i = np.mgrid[:ny, :nx]
    r_i = np.sqrt((x_i - cx)**2 + (y_i - cy)**2).astype(int)
    km  = min(cx, cy)
    kr  = np.arange(1, km)
    Pr  = np.array([P[r_i == k].mean() if (r_i == k).any() else 0.0 for k in kr])
    klo = max(2, int(k_lo_frac * km)); khi = int(k_hi_frac * km)
    mf  = (kr >= klo) & (kr <= khi) & (Pr > 0)
    if mf.sum() < 3:
        return {'alpha': np.nan, 'k_min': np.nan, 'k_max': np.nan}
    sl, ic, _, _, _ = _lr(np.log10(kr[mf].astype(float)), np.log10(Pr[mf]))
    res = np.log10(Pr + 1e-30) - (ic + sl * np.log10(kr.astype(float)))
    inr = np.abs(res) < thr
    return {'alpha': sl,
            'k_min': float(kr[inr][0])  if inr.any() else float(klo),
            'k_max': float(kr[inr][-1]) if inr.any() else float(khi)}
''')

code('''# Run classical inference on every eval image
_c_kmin, _c_kmax = [], []
for img in imgs_raw:
    fp = fourier_params(img)
    _c_kmin.append(fp['k_min'])
    _c_kmax.append(fp['k_max'])
classical_pred = {'k_min': np.array(_c_kmin, dtype=np.float64),
                  'k_max': np.array(_c_kmax, dtype=np.float64)}
classical_valid = np.isfinite(classical_pred['k_min']) & np.isfinite(classical_pred['k_max'])
n_dropped = int((~classical_valid).sum())
print(f'classical valid: {classical_valid.sum()}/{len(classical_valid)}  dropped(NaN): {n_dropped}')
''')
```

- [ ] **Step 2: Rebuild + execute**

Run: `uv run python scripts/_build_synth_val.py && SYNTH_VAL_N_EVAL=64 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/synthetic_validation.ipynb`
Expected: no exception; prints `classical valid: N/64  dropped(NaN): K` with N+K = 64.

- [ ] **Step 3: Commit**

```bash
git add scripts/_build_synth_val.py notebooks/comparison/synthetic_validation.ipynb
git commit -m "feat: add classical FFT inference + NaN handling"
```

---

### Task 5: Metrics computation + metrics.json

**Files:**
- Modify: `scripts/_build_synth_val.py` (append cells), regenerate notebook.

**Interfaces:**
- Consumes: `gt, cnn_pred, classical_pred, classical_valid, n_dropped, OUT_DIR`.
- Produces: `compute_metrics(y_true, y_pred)` (returns `{'mae','rmse','r2','bias','n'}`); `metrics` dict `{'cnn': {param: {...}}, 'classical': {param: {...}}, 'n_eval': int, 'n_dropped_classical': int}`; writes `OUT_DIR/metrics.json`.

- [ ] **Step 1: Append the metrics cells**

Append to `CELLS`:

```python
code('''# Metrics — MAE / RMSE / R2 / bias
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    return {
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2':   float(r2_score(y_true, y_pred)),
        'bias': float(np.mean(y_pred - y_true)),
        'n':    int(len(y_true)),
    }

metrics = {'cnn': {}, 'classical': {}, 'n_eval': int(len(gt['k_min'])),
           'n_dropped_classical': n_dropped}

for p in ['k_min', 'k_max', 'sigma']:                 # CNN: all three
    metrics['cnn'][p] = compute_metrics(gt[p], cnn_pred[p])

for p in ['k_min', 'k_max']:                          # Classical: k_min/k_max only, valid rows
    m = classical_valid
    metrics['classical'][p] = compute_metrics(gt[p][m], classical_pred[p][m])

with open(OUT_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"{'PARAM':<7} {'CNN_R2':>8} {'CNN_MAE':>9} {'CLS_R2':>8} {'CLS_MAE':>9}")
for p in ['k_min', 'k_max', 'sigma']:
    c = metrics['cnn'][p]
    cl = metrics['classical'].get(p)
    cls_r2  = f"{cl['r2']:.4f}"  if cl else '   --   '
    cls_mae = f"{cl['mae']:.3f}" if cl else '   --   '
    print(f"{p:<7} {c['r2']:>8.4f} {c['mae']:>9.3f} {cls_r2:>8} {cls_mae:>9}")
''')
```

- [ ] **Step 2: Rebuild + execute, then assert metrics.json is sane**

Run: `uv run python scripts/_build_synth_val.py && SYNTH_VAL_N_EVAL=512 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/synthetic_validation.ipynb`

Then verify the CNN is accurate in-distribution:

Run:
```bash
uv run python -c "import json; m=json.load(open('outputs/comparison/synthetic_validation/metrics.json')); \
print('cnn k_min R2', m['cnn']['k_min']['r2']); \
print('cnn k_max R2', m['cnn']['k_max']['r2']); \
print('cnn sigma R2', m['cnn']['sigma']['r2']); \
assert m['cnn']['k_min']['r2'] > 0.9, 'CNN k_min R2 too low for in-distribution'; \
print('OK: metrics.json written and CNN accurate')"
```
Expected: prints the three R² values (all should be high, > 0.9 for at least k_min which is the model's strongest param) and `OK: ...`. If k_max/sigma R² are notably lower, that is the expected diagnostic signal feeding the second-training spec — not a failure.

- [ ] **Step 3: Commit**

```bash
git add scripts/_build_synth_val.py notebooks/comparison/synthetic_validation.ipynb
git commit -m "feat: compute CNN/classical metrics + write metrics.json"
```

---

### Task 6: Figures — scatter, residuals, diagnostic binned MAE

**Files:**
- Modify: `scripts/_build_synth_val.py` (append cells), regenerate notebook.

**Interfaces:**
- Consumes: `gt, cnn_pred, classical_pred, classical_valid, metrics, OUT_DIR`.
- Produces: PNG files `scatter.png`, `residuals.png`, `cnn_error_by_value.png` in `OUT_DIR`.

- [ ] **Step 1: Append the plotting cells**

Append to `CELLS`:

```python
code('''# Figure 1 — pred-vs-true scatter (CNN + classical overlaid)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAMS = ['k_min', 'k_max', 'sigma']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, p in zip(axes, PARAMS):
    ax.scatter(gt[p], cnn_pred[p], s=6, alpha=0.3, color='#1565C0', label='CNN', rasterized=True)
    if p in classical_pred:
        m = classical_valid
        ax.scatter(gt[p][m], classical_pred[p][m], s=6, alpha=0.3, color='#EF6C00',
                   label='Classical', rasterized=True)
    lims = [min(gt[p].min(), cnn_pred[p].min()), max(gt[p].max(), cnn_pred[p].max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    c = metrics['cnn'][p]
    ax.set_xlabel(f'True {p}'); ax.set_ylabel(f'Pred {p}')
    ax.set_title(f"{p}\\nCNN R2={c['r2']:.4f} MAE={c['mae']:.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs True — Synthetic (in-distribution)', fontweight='bold')
plt.tight_layout(); plt.savefig(OUT_DIR / 'scatter.png', dpi=150, bbox_inches='tight'); plt.close()
print('wrote scatter.png')
''')

code('''# Figure 2 — residual histograms (CNN)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, p in zip(axes, PARAMS):
    res = cnn_pred[p] - gt[p]
    ax.hist(res, bins=60, color='#1565C0', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(res.mean(), color='red', linestyle='--', linewidth=1,
               label=f'mean={res.mean():.3f} std={res.std():.3f}')
    ax.set_xlabel(f'Residual (pred - true {p})'); ax.set_ylabel('Count')
    ax.set_title(p); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('CNN Residuals — Synthetic', fontweight='bold')
plt.tight_layout(); plt.savefig(OUT_DIR / 'residuals.png', dpi=150, bbox_inches='tight'); plt.close()
print('wrote residuals.png')
''')

code('''# Figure 3 — DIAGNOSTIC: CNN MAE binned by true value of k_max and sigma
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, p in zip(axes, ['k_max', 'sigma']):
    yt = gt[p]
    abs_err = np.abs(cnn_pred[p] - yt)
    bins = np.linspace(yt.min(), yt.max(), 11)
    idx  = np.digitize(yt, bins) - 1
    idx  = np.clip(idx, 0, len(bins) - 2)
    centers, mae_bin = [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.any():
            centers.append(0.5 * (bins[b] + bins[b+1]))
            mae_bin.append(abs_err[sel].mean())
    ax.plot(centers, mae_bin, 'o-', color='#C62828')
    ax.set_xlabel(f'True {p}'); ax.set_ylabel('CNN MAE')
    ax.set_title(f'CNN error vs true {p}'); ax.grid(True, alpha=0.3)
plt.suptitle('Diagnostic — where the CNN fails (input for 2nd training)', fontweight='bold')
plt.tight_layout(); plt.savefig(OUT_DIR / 'cnn_error_by_value.png', dpi=150, bbox_inches='tight'); plt.close()
print('wrote cnn_error_by_value.png')
''')
```

- [ ] **Step 2: Rebuild + execute, then assert figures exist**

Run: `uv run python scripts/_build_synth_val.py && SYNTH_VAL_N_EVAL=512 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/synthetic_validation.ipynb`

Run:
```bash
uv run python -c "from pathlib import Path; d=Path('outputs/comparison/synthetic_validation'); \
[print('OK', f) or 0 for f in ['scatter.png','residuals.png','cnn_error_by_value.png'] if (d/f).exists()]; \
assert all((d/f).exists() for f in ['scatter.png','residuals.png','cnn_error_by_value.png']), 'missing figure'"
```
Expected: prints `OK scatter.png`, `OK residuals.png`, `OK cnn_error_by_value.png`.

- [ ] **Step 3: Commit**

```bash
git add scripts/_build_synth_val.py notebooks/comparison/synthetic_validation.ipynb
git commit -m "feat: add scatter, residual, and diagnostic figures"
```

---

### Task 7: Full-size run + final verification

**Files:**
- Modify: `notebooks/comparison/synthetic_validation.ipynb` (executed at full N_EVAL, outputs committed).

**Interfaces:**
- Consumes: the complete notebook from Tasks 1–6.
- Produces: executed notebook + `outputs/comparison/synthetic_validation/{metrics.json,scatter.png,residuals.png,cnn_error_by_value.png}` at the default N_EVAL=2000.

- [ ] **Step 1: Execute the full notebook at default N_EVAL (2000)**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/comparison/synthetic_validation.ipynb`
Expected: completes with no exception (may take a few minutes on GPU; longer on CPU).

- [ ] **Step 2: Print the final metrics summary**

Run:
```bash
uv run python -c "import json; m=json.load(open('outputs/comparison/synthetic_validation/metrics.json')); \
print('n_eval', m['n_eval'], 'dropped_classical', m['n_dropped_classical']); \
[print(p, 'CNN R2=%.4f MAE=%.3f' % (m['cnn'][p]['r2'], m['cnn'][p]['mae'])) for p in ['k_min','k_max','sigma']]; \
[print(p, 'CLS R2=%.4f MAE=%.3f' % (m['classical'][p]['r2'], m['classical'][p]['mae'])) for p in ['k_min','k_max']]"
```
Expected: full-size metrics. CNN R² high across params (in-distribution); classical k_min/k_max lower than CNN.

- [ ] **Step 3: Commit notebook outputs + results**

```bash
git add notebooks/comparison/synthetic_validation.ipynb outputs/comparison/synthetic_validation/
git commit -m "feat: full-size synthetic validation run — metrics + figures"
```

---

## Self-Review notes

- **Spec coverage:** Load test split (Task 2) · CNN inference (Task 3) · classical FFT (Task 4) · metrics MAE/RMSE/R²/bias + metrics.json (Task 5) · scatter/residual/diagnostic figures (Task 6) · NaN handling (Task 4) · file-existence asserts (Task 1) · full run (Task 7). All spec sections mapped.
- **Sigma is CNN-only:** enforced in Task 5 (`classical` loop iterates only `k_min,k_max`) and Task 6 (classical scatter guarded by `if p in classical_pred`). Consistent with spec.
- **Normalization match:** IMG_P1/P99 read from results.json; KMIN/KMAX/SIG/BETA constants verbatim; denorm sigma uses `np.exp` — matches `joint_regression_v2.ipynb`.
- **Type consistency:** `fourier_params` returns float k_min/k_max (Task 4) → `classical_pred` float64 arrays → `compute_metrics` consumes arrays (Task 5) → figures consume same dicts (Task 6). `cnn_pred`/`gt`/`classical_pred` share keys throughout.
