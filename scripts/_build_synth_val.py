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

def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s)
                for t, s in CELLS]
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    NB_PATH.write_text(nbf.writes(nb), encoding='utf-8')
    print(f'Wrote {len(nb.cells)} cells -> {NB_PATH}')

if __name__ == '__main__':
    build()
