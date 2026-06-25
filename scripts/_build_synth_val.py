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

def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s)
                for t, s in CELLS]
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    NB_PATH.write_text(nbf.writes(nb), encoding='utf-8')
    print(f'Wrote {len(nb.cells)} cells -> {NB_PATH}')

if __name__ == '__main__':
    build()
