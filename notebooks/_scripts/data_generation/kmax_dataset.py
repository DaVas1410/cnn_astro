# AUTO-GENERATED from notebooks/data_generation/kmax_dataset.ipynb — do not edit by hand
#!/usr/bin/env python
# coding: utf-8

# # Balanced kmax Dataset Generation (Local PC)
# 
# This notebook generates a dataset with random `k_min` and **balanced discrete `k_max` labels**.
# 
# - Total images: `32000`
# - Resolution: `128x128`
# - `k_min`: random in `[1, 16]`
# - `k_max`: balanced by **discrete integer label** in `[2, 64]`
# - Constraint: `k_max > k_min + MIN_SEPARATION`

# In[1]:


from pathlib import Path
import os
import sys
import subprocess

project_root = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
script_path = project_root / 'scripts', 'run_generation_random_both.py'
output_dir = project_root / 'data', 'raw'

print('Project root:', project_root)
print('Generator script exists:', script_path.exists())
print('Python executable:', sys.executable)


# In[2]:


# If this fails, install missing packages in this notebook kernel:
# %pip install numpy h5py pyFC

import numpy as np
import h5py
print('numpy version:', np.__version__)
print('h5py version:', h5py.__version__)


# In[3]:


# Generation settings (edit if needed)
NUM_IMAGES = 32000
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
KMIN_LOW = 1
KMIN_HIGH = 16
KMAX_LOW = 2
KMAX_HIGH = 64
MIN_SEPARATION = 2.5
BALANCED_KMAX = True
BATCH_SIZE = 256
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 2)

env = os.environ.copy()
env.update({
    'NUM_IMAGES': str(NUM_IMAGES),
    'IMAGE_WIDTH': str(IMAGE_WIDTH),
    'IMAGE_HEIGHT': str(IMAGE_HEIGHT),
    'KMIN_LOW': str(KMIN_LOW),
    'KMIN_HIGH': str(KMIN_HIGH),
    'KMAX_LOW': str(KMAX_LOW),
    'KMAX_HIGH': str(KMAX_HIGH),
    'MIN_SEPARATION': str(MIN_SEPARATION),
    'BALANCED_KMAX': '1' if BALANCED_KMAX else '0',
    'BATCH_SIZE': str(BATCH_SIZE),
    'NUM_WORKERS': str(NUM_WORKERS),
})

if not script_path.exists():
    raise FileNotFoundError(f'Missing script: {script_path}')

mode_label = 'balancedkmax' if BALANCED_KMAX else 'uniformkmax'
cmd = [sys.executable, str(script_path)]
print('Running command:', ' '.join(cmd))
print('\nConfiguration:')
print(f'  - Images: {NUM_IMAGES}')
print(f'  - Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}')
print(f'  - k_min range: [{KMIN_LOW}, {KMIN_HIGH}]')
print(f'  - k_max range: [{KMAX_LOW}, {KMAX_HIGH}]')
print(f'  - Balanced discrete kmax labels: {BALANCED_KMAX}')
print(f'  - Min separation (kmax - kmin): {MIN_SEPARATION}')
print(f'  - Mode label: {mode_label}')
print(f'  - Batch size: {BATCH_SIZE}')
print(f'  - Workers: {NUM_WORKERS}')
print('\nThis may take a while for 32k images...')

result = subprocess.run(cmd, cwd=str(project_root), env=env, check=False)
print('Exit code:', result.returncode)
if result.returncode != 0:
    raise RuntimeError('Dataset generation failed. Check notebook output above for details.')


# In[5]:


# Verify output file, constraints, and class-balance quality
import numpy as np
import h5py
import matplotlib.pyplot as plt

mode_label = 'balancedkmax' if BALANCED_KMAX else 'uniformkmax'
expected_name = f'randomboth_{mode_label}_kmin{KMIN_LOW}-{KMIN_HIGH}_kmax{KMAX_LOW}-{KMAX_HIGH}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}.h5'
dataset_file = output_dir / expected_name

if not dataset_file.exists():
    raise FileNotFoundError(f'Expected dataset not found: {dataset_file}')

with h5py.File(dataset_file, 'r') as f:
    images_shape = f['images'].shape
    kmins = f['parameters']['k_min'][:]
    kmaxs = f['parameters']['k_max'][:]

kmax_labels = np.rint(kmaxs).astype(np.int32)
classes, counts = np.unique(kmax_labels, return_counts=True)
imbalance_ratio = float(counts.max() / max(counts.min(), 1))

sep_ok = np.all(kmaxs > (kmins + MIN_SEPARATION - 1e-6))
basic_ok = np.all(kmins < kmaxs)

print('Dataset file:', dataset_file)
print('Images shape:', images_shape)
print()
print('k_min statistics:')
print(f'  min/max: {float(kmins.min()):.3f} / {float(kmins.max()):.3f}')
print(f'  mean/std: {float(kmins.mean()):.3f} / {float(kmins.std()):.3f}')
print()
print('k_max statistics:')
print(f'  min/max: {float(kmaxs.min()):.3f} / {float(kmaxs.max()):.3f}')
print(f'  mean/std: {float(kmaxs.mean()):.3f} / {float(kmaxs.std()):.3f}')
print()
print(f'Constraint k_min < k_max satisfied: {basic_ok} ({np.sum(kmins < kmaxs)}/{len(kmins)})')
print(f'Constraint k_max > k_min + {MIN_SEPARATION} satisfied: {sep_ok} ({np.sum(kmaxs > (kmins + MIN_SEPARATION - 1e-6))}/{len(kmins)})')
print()
print('Discrete k_max class-balance check:')
print(f'  Number of classes: {len(classes)}')
print(f'  Class range: {int(classes.min())} -> {int(classes.max())}')
print(f'  Min/Max class count: {int(counts.min())} / {int(counts.max())}')
print(f'  Imbalance ratio (max/min): {imbalance_ratio:.4f}')

preview_n = min(12, len(classes))
print('\nFirst class counts preview:')
for c, n in zip(classes[:preview_n], counts[:preview_n]):
    print(f'  kmax={int(c):2d} -> {int(n):5d}')

# Histograms + discrete class counts
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].hist(kmins, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('k_min')
axes[0].set_ylabel('Count')
axes[0].set_title(f'k_min Distribution (range: [{KMIN_LOW}, {KMIN_HIGH}])')
axes[0].grid(True, alpha=0.3)

axes[1].hist(kmaxs, bins=20, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_xlabel('k_max (continuous)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'k_max Distribution (range: [{KMAX_LOW}, {KMAX_HIGH}])')
axes[1].grid(True, alpha=0.3)

axes[2].bar(classes, counts, color='seagreen', alpha=0.85)
axes[2].set_xlabel('k_max discrete label (rounded)')
axes[2].set_ylabel('Count')
axes[2].set_title('Discrete k_max Class Counts')
axes[2].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:




