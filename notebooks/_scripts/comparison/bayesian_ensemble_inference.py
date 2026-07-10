# AUTO-GENERATED from notebooks/comparison/bayesian_ensemble_inference.ipynb — do not edit by hand
#!/usr/bin/env python
# coding: utf-8

# # Bayesian Ensemble Joint Regression — Local Inference
# 
# Loads a 5-member deep ensemble of heteroscedastic ResNet50 models (trained on HPC) and runs
# inference with calibrated uncertainty (μ ± σ_total = σ_aleatoric + σ_epistemic) on:
# 1. The synthetic test set
# 2. Real GASS HI survey FITS data
# 
# **Prerequisites:** trained checkpoints at `outputs/hpc_ensemble/member_{0..4}/best_model.pt`
# (produced by the HPC SLURM pipeline in `scripts/hpc_ensemble/`).

# In[ ]:


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


# In[ ]:


models = []
for mid in range(N_MEMBERS):
    ckpt = ENSEMBLE_DIR / f'member_{mid}' / 'best_model.pt'
    m = ResNet50HeteroJoint().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    models.append(m)
    print(f'  Loaded member {mid} from {ckpt}')

print(f'\nAll {N_MEMBERS} members loaded.')


# In[ ]:


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


# In[ ]:


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

# NOTE: real GASS data is in raw brightness-temperature units, not the
# log10-fractal-cube units the ensemble was trained on (IMG_P1/IMG_P99).
# Predictions on real data are illustrative only -- a large domain shift
# is expected. Following the observational_images/io-fits.ipynb workflow,
# integrate the frequency axis into one full map plus 5 velocity-channel
# bins (6 maps total) rather than running inference on all raw channels.
n = cube.shape[0]
maps = [np.sum(cube, axis=0)]  # full integration
edges = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n]
for i in range(5):
    maps.append(np.sum(cube[edges[i]:edges[i + 1]], axis=0))

channel_labels = ['Full integration'] + [f'Channel {i+1}' for i in range(5)]
results_gass = [infer_image(m) for m in maps]

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

