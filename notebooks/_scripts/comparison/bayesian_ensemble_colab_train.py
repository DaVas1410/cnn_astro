# AUTO-GENERATED from notebooks/comparison/bayesian_ensemble_colab_train.ipynb — do not edit by hand
#!/usr/bin/env python
# coding: utf-8

# # Bayesian Ensemble — Colab Training (3 members)
# 
# Trains the ResNet50 heteroscedastic joint-regression ensemble (`k_min` / `k_max` / `sigma`
# with uncertainty) — the **same model and config** as `scripts/hpc_ensemble/` — on a Colab GPU.
# Code and dataset both live in **Google Drive**.
# 
# **How to run:** `Runtime → Change runtime type → GPU`, then edit the one **EDIT-ME** cell
# below and `Runtime → Run all`.
# 
# - 3 ensemble members → aleatoric **and** epistemic uncertainty.
# - Single run, no resume. Results are saved to Drive so they survive a disconnect.
# - Your tracked `config.yaml` is never modified (overrides go to a throwaway `config_colab.yaml`).

# ## 1. GPU check

# In[ ]:


get_ipython().system('nvidia-smi')
import torch
print('\ntorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('\n!!!  WARNING: no GPU detected. Set Runtime -> Change runtime type -> GPU  !!!')


# ## 2. Mount Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ## 3. ✏️ EDIT-ME — paths and run settings
# 
# This is the **only cell you need to edit**. Point `REPO_DIR` and `DATA_FILE` at where
# your `cnn_astro` folder and `.h5` dataset live inside Drive.
# 
# In Colab, Drive is mounted at `/content/drive/MyDrive/...` (your "My Drive" root).

# In[ ]:


# --- paths in Drive (EDIT THESE) --------------------------------------------
REPO_DIR  = '/content/drive/MyDrive/cnn_astro'                         # folder containing scripts/, notebooks/, ...
DATA_FILE = '/content/drive/MyDrive/cnn_astro/data/raw/uniform_kmin_128x128_100000.h5'  # the .h5 dataset

# where results (checkpoints, metrics, plots) are written — kept on Drive so they survive a disconnect
OUTPUT_DIR = '/content/drive/MyDrive/cnn_astro/outputs/colab_ensemble'

# --- run settings -----------------------------------------------------------
N_MEMBERS   = 3      # 1 = aleatoric only; >=2 adds epistemic uncertainty
EPOCHS      = 80     # per member; lower (e.g. 20) for a quick pipeline test
BATCH_SIZE  = 256    # bigger = fills more GPU + faster; lower to 64/32 if you hit out-of-memory
NUM_WORKERS = 8      # dataloader processes; more = keeps the GPU fed
COPY_TO_LOCAL = True # copy the .h5 to the local Colab SSD first — the main speedup vs reading off Drive

# --- sanity: verify the paths exist before doing anything expensive ---------
import os
assert os.path.isdir(REPO_DIR),  f'REPO_DIR not found: {REPO_DIR}'
assert os.path.isfile(DATA_FILE), f'DATA_FILE not found: {DATA_FILE}'
assert os.path.isdir(os.path.join(REPO_DIR, 'scripts', 'hpc_ensemble')), \
    f'scripts/hpc_ensemble not found under REPO_DIR: {REPO_DIR}'
print('REPO_DIR   :', REPO_DIR)
print('DATA_FILE  :', DATA_FILE, f'({os.path.getsize(DATA_FILE)/1e9:.2f} GB)')
print('OUTPUT_DIR :', OUTPUT_DIR)
print(f'\n{N_MEMBERS} member(s), {EPOCHS} epochs, batch {BATCH_SIZE}, {NUM_WORKERS} workers')


# ## 4. Install deps & put the code on the path
# 
# Colab already ships torch, torchvision, scikit-learn, scipy, matplotlib, numpy.
# We only add what's missing (`h5py`, `pyyaml`).

# In[ ]:


get_ipython().system('pip install -q h5py pyyaml')

import sys
HPC_DIR = os.path.join(REPO_DIR, 'scripts', 'hpc_ensemble')
if HPC_DIR not in sys.path:
    sys.path.insert(0, HPC_DIR)

# import the existing, unmodified modules
from config import load_config, output_dir
from dataset import load_splits, denorm, uncertainty_to_orig
from loss import WeightedGaussianNLL
from model import ResNet50HeteroJoint
import train_member as tm
import evaluate_ensemble as ev
print('Imported hpc_ensemble modules from:', HPC_DIR)


# ## 5. Build the runtime config
# 
# Load the tracked `config.yaml`, override only the Colab-specific fields, and write a
# throwaway `config_colab.yaml`. The tracked config is never touched.

# In[ ]:


import yaml

TRACKED_CONFIG = os.path.join(HPC_DIR, 'config.yaml')
cfg = load_config(TRACKED_CONFIG)

# data.file is resolved relative to hpc.project_dir by dataset.load_splits, so
# point project_dir at the Drive repo and make data.file the path relative to it.
cfg['hpc']['project_dir'] = REPO_DIR
cfg['data']['file'] = os.path.relpath(DATA_FILE, REPO_DIR)
cfg['output']['dir'] = OUTPUT_DIR                 # absolute -> used as-is
cfg['ensemble']['n_members'] = N_MEMBERS
cfg['training']['epochs'] = EPOCHS
cfg['training']['batch_size'] = BATCH_SIZE

os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG_COLAB = os.path.join(OUTPUT_DIR, 'config_colab.yaml')
with open(CONFIG_COLAB, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print('Runtime config written to:', CONFIG_COLAB)
print('  project_dir :', cfg['hpc']['project_dir'])
print('  data.file   :', cfg['data']['file'])
print('  output.dir  :', cfg['output']['dir'])
print('  n_members   :', cfg['ensemble']['n_members'])
print('  epochs      :', cfg['training']['epochs'])
print('  loss_weights:', cfg['loss_weights'])


# ## 6. Dataset sanity check
# 
# Confirm the `.h5` has the expected shape and that the parameter ranges fall inside the
# config's normalization bounds — catches a wrong/mismatched dataset **before** a multi-hour run.

# In[ ]:


import h5py, numpy as np

with h5py.File(DATA_FILE, 'r') as hf:
    imgs = hf['images']
    print('images shape :', imgs.shape, imgs.dtype)
    for k in ('k_min', 'k_max', 'sigma'):
        v = hf['parameters'][k][:]
        print(f'  {k:6s}: min={v.min():.4g}  max={v.max():.4g}  mean={v.mean():.4g}')

d = cfg['data']
print('\nconfig normalization bounds:')
print(f"  k_min in [{d['kmin_lo']}, {d['kmin_hi']}]")
print(f"  k_max in [{d['kmax_lo']}, {d['kmax_hi']}]")
print(f"  sigma in [{np.exp(d['log_sigma_lo']):.3g}, {np.exp(d['log_sigma_hi']):.3g}]")
print('\nIf the data ranges sit far outside these bounds, you may be using a different\n'
      'dataset than the config was tuned for — check before training.')


# ## 6b. Copy dataset to local disk (speed)
# 
# Reading the `.h5` off the Google Drive FUSE mount is slow (random reads over the network) and
# is the main reason the GPU sits idle. Copying it once to the local Colab SSD (`/content/`) makes
# every epoch read much faster. Runs only if `COPY_TO_LOCAL = True`.

# In[ ]:


import shutil, time

if COPY_TO_LOCAL:
    local_path = os.path.join('/content', os.path.basename(DATA_FILE))
    if not os.path.exists(local_path) or os.path.getsize(local_path) != os.path.getsize(DATA_FILE):
        print(f'Copying {os.path.getsize(DATA_FILE)/1e9:.2f} GB to local SSD (one-time)...')
        t0 = time.time()
        shutil.copy(DATA_FILE, local_path)
        print(f'  done in {time.time()-t0:.0f}s -> {local_path}')
    else:
        print('Local copy already present:', local_path)
    # repoint the config at the local file so load_splits reads from SSD, not Drive
    cfg['hpc']['project_dir'] = '/content'
    cfg['data']['file'] = os.path.basename(DATA_FILE)
    with open(CONFIG_COLAB, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print('Config now reads data from:', os.path.join(cfg['hpc']['project_dir'], cfg['data']['file']))
else:
    print('COPY_TO_LOCAL is False — reading directly from Drive (slower).')

# cuDNN autotuner: pick fastest conv algorithms for the fixed 128x128 input size
import torch
torch.backends.cudnn.benchmark = True


# ## 7. Train the ensemble members
# 
# Sequential, one member at a time. Each member uses seed `base_seed + i` and saves its best
# checkpoint + training history to Drive. Reuses the exact training loop from `train_member.py`.

# In[ ]:


import json, time
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Load splits once and reuse across members (same split — seeded by base_seed).
train_ds, val_ds, _ = load_splits(cfg)
print(f'train={len(train_ds)}  val={len(val_ds)}')
loader_kw = dict(batch_size=cfg['training']['batch_size'], num_workers=NUM_WORKERS,
                 pin_memory=(device.type == 'cuda'),
                 persistent_workers=(NUM_WORKERS > 0),
                 prefetch_factor=(4 if NUM_WORKERS > 0 else None))
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

lw = cfg['loss_weights']

for member_id in range(cfg['ensemble']['n_members']):
    seed = cfg['training']['base_seed'] + member_id
    tm.set_seeds(seed)
    out_dir = output_dir(cfg, member_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n{"="*60}\n[member {member_id}] seed={seed} -> {out_dir}\n{"="*60}')

    model = ResNet50HeteroJoint().to(device)
    criterion = WeightedGaussianNLL([lw['k_min'], lw['k_max'], lw['sigma']]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                            weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    best_ckpt = out_dir / 'best_model.pt'
    epochs = cfg['training']['epochs']
    t0 = time.time()

    for epoch in range(epochs):
        tl = tm.train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl = tm.eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl)
        history['train_loss'].append(float(tl))
        history['val_loss'].append(float(vl))
        lr = optimizer.param_groups[0]['lr']
        print(f'  Ep {epoch+1:3d}/{epochs} | train {tl:.4f} | val {vl:.4f} | lr {lr:.1e}')
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_ckpt)

    elapsed = (time.time() - t0) / 60.0
    print(f'[member {member_id}] done in {elapsed:.1f} min — best val {best_val:.4f}')
    with open(out_dir / 'history.json', 'w') as f:
        json.dump({'member_id': member_id, 'seed': seed, 'best_val_loss': best_val,
                   'training_time_min': elapsed, 'history': history}, f, indent=2)

print('\nAll members trained.')


# ## 8. Evaluate the ensemble
# 
# Combines all members, computes metrics on the test split, saves `results.json` /
# `results_summary.txt` and the three diagnostic plots to Drive, and shows them inline.
# Reuses `evaluate_ensemble.py` unchanged.

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_, _, test_ds = load_splits(cfg)
test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'],
                         shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=(device.type == 'cuda'))
y_true_norm = test_ds.labels

member_preds = []
for mid in range(cfg['ensemble']['n_members']):
    ckpt = output_dir(cfg, mid) / 'best_model.pt'
    m = ResNet50HeteroJoint().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    member_preds.append(ev.predict_member(m, test_loader, device))
    print('Loaded member', mid)

ens = ev.combine_ensemble(member_preds)
y_true = denorm(y_true_norm, cfg)
y_pred = denorm(ens['mu'], cfg)
y_sig_total = uncertainty_to_orig(ens['sigma_total'],     ens['mu'], cfg)
y_sig_alea  = uncertainty_to_orig(ens['sigma_aleatoric'], ens['mu'], cfg)
y_sig_epis  = uncertainty_to_orig(ens['sigma_epistemic'], ens['mu'], cfg)

base_dir = output_dir(cfg)
base_dir.mkdir(parents=True, exist_ok=True)

metrics = {}
print(f"\n{'PARAM':<8}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'sig_tot':>11}")
print('-' * 46)
for p in ev.PARAMS:
    metrics[p] = {
        'mae':  float(mean_absolute_error(y_true[p], y_pred[p])),
        'rmse': float(np.sqrt(mean_squared_error(y_true[p], y_pred[p]))),
        'r2':   float(r2_score(y_true[p], y_pred[p])),
        'mean_sigma_total': float(y_sig_total[p].mean()),
    }
    m = metrics[p]
    print(f"{p:<8}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.4f}{m['mean_sigma_total']:>11.4f}")

results = {
    'metrics': metrics, 'n_members': cfg['ensemble']['n_members'], 'params': ev.PARAMS,
    'y_true':          {p: y_true[p].tolist()      for p in ev.PARAMS},
    'y_pred':          {p: y_pred[p].tolist()      for p in ev.PARAMS},
    'sigma_total':     {p: y_sig_total[p].tolist() for p in ev.PARAMS},
    'sigma_aleatoric': {p: y_sig_alea[p].tolist()  for p in ev.PARAMS},
    'sigma_epistemic': {p: y_sig_epis[p].tolist()  for p in ev.PARAMS},
}
with open(base_dir / 'results.json', 'w') as f:
    json.dump(results, f)
with open(base_dir / 'results_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f"Ensemble ({cfg['ensemble']['n_members']} members) — test set\n")
    f.write('=' * 50 + '\n')
    f.write(f"{'PARAM':<8}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'sig_tot':>11}\n")
    for p, m in metrics.items():
        f.write(f"{p:<8}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.4f}{m['mean_sigma_total']:>11.4f}\n")

ev.plot_scatter(y_true, y_pred, y_sig_total, base_dir)
ev.plot_residuals(y_true, y_pred, base_dir)
ev.plot_calibration(y_true, y_pred, y_sig_total, base_dir)
print('\nSaved results + plots to:', base_dir)


# In[ ]:


# Show the saved diagnostic plots inline
from IPython.display import Image, display
for name in ('scatter.png', 'residuals.png', 'uncertainty_calibration.png'):
    print(name)
    display(Image(filename=str(base_dir / name)))


# ## 9. Training curves per member

# In[ ]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
for mid in range(cfg['ensemble']['n_members']):
    with open(output_dir(cfg, mid) / 'history.json') as f:
        h = json.load(f)['history']
    ax.plot(h['train_loss'], '--', alpha=0.6, label=f'm{mid} train')
    ax.plot(h['val_loss'],   '-',  label=f'm{mid} val')
ax.set_xlabel('epoch'); ax.set_ylabel('weighted Gaussian NLL')
ax.set_title('Training curves'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# ---
# ### Outputs (in your Drive `OUTPUT_DIR`)
# 
# - `member_{0..N}/best_model.pt`, `history.json` — per-member weights + curves
# - `results.json`, `results_summary.txt` — ensemble metrics (MAE / RMSE / R² / σ_total)
# - `scatter.png`, `residuals.png`, `uncertainty_calibration.png` — diagnostics
# - `config_colab.yaml` — the exact config used
# 
# Copy `member_*/best_model.pt` back to your PC for
# `notebooks/comparison/bayesian_ensemble_inference.ipynb` to run inference on real GASS data.
