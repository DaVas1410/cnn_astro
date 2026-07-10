# AUTO-GENERATED from notebooks/data_generation/k_mas_dataset.ipynb — do not edit by hand
#!/usr/bin/env python
# coding: utf-8

# # Random kmin Dataset Generation (Local PC)
# 
# This notebook generates a dataset with **random `k_min` per image** using your local Python environment.
# 
# - Total images: `32000`
# - Resolution: `128x128`
# - `k_min`: uniform random integer in `[1, 32]`
# - `k_max`: auto (Nyquist)

# In[1]:


from pathlib import Path
import os
import sys
import subprocess

project_root = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
script_path = project_root / 'scripts', 'run_generation_random_kmin.py'
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
KMIN_HIGH = 32
BATCH_SIZE = 256
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 2)

env = os.environ.copy()
env.update({
    'NUM_IMAGES': str(NUM_IMAGES),
    'IMAGE_WIDTH': str(IMAGE_WIDTH),
    'IMAGE_HEIGHT': str(IMAGE_HEIGHT),
    'KMIN_LOW': str(KMIN_LOW),
    'KMIN_HIGH': str(KMIN_HIGH),
    'BATCH_SIZE': str(BATCH_SIZE),
    'NUM_WORKERS': str(NUM_WORKERS),
})

if not script_path.exists():
    raise FileNotFoundError(f'Missing script: {script_path}')

cmd = [sys.executable, str(script_path)]
print('Running command:', ' '.join(cmd))
print('Workers:', NUM_WORKERS)
print('This may take a while for 32k images...')

result = subprocess.run(cmd, cwd=str(project_root), env=env, check=False)
print('Exit code:', result.returncode)
if result.returncode != 0:
    raise RuntimeError('Dataset generation failed. Check notebook output above for details.')


# In[4]:


# Verify output file and k_min distribution
import numpy as np
import h5py

expected_name = f'randomkmin_{KMIN_LOW}-{KMIN_HIGH}_{IMAGE_HEIGHT}x{IMAGE_WIDTH}_{NUM_IMAGES}.h5'
dataset_file = output_dir / expected_name

if not dataset_file.exists():
    raise FileNotFoundError(f'Expected dataset not found: {dataset_file}')

with h5py.File(dataset_file, 'r') as f:
    images_shape = f['images'].shape
    kmins = f['parameters']['k_min'][:]

print('Dataset file:', dataset_file)
print('Images shape:', images_shape)
print('k_min min/max:', float(kmins.min()), float(kmins.max()))
print('k_min mean/std:', float(kmins.mean()), float(kmins.std()))

hist_counts, hist_edges = np.histogram(kmins, bins=10, range=(KMIN_LOW, KMIN_HIGH))
print('k_min histogram (10 bins):')
for i, c in enumerate(hist_counts):
    left = hist_edges[i]
    right = hist_edges[i + 1]
    print(f'  [{left:5.2f}, {right:5.2f}): {int(c)}')

