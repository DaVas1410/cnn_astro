# Fractal Classifier v0.2

A lightweight CNN regression framework for predicting the minimum turbulence scale (`k_min`)
from synthetic fractal cloud images.

## What's new in v0.2 vs v0.1

| Feature | v0.1 | v0.2 |
|---|---|---|
| Task | 6-class classification | k_min regression (1–32) |
| Model | Standard Conv2D blocks (~7M params) | Depthwise separable CNN (~180K params) |
| Input | 512×512 only | 128×128 native; 256/512 via patches |
| Dataset | 6-file discrete k HDF5 | 32-file multi-resolution HDF5 archive |
| Output | Softmax + centroid trick | Linear regression head |
| Loss | Sparse categorical crossentropy | Huber (δ=1.0) |

## Dataset

Extract the dataset archive before running:
```bash
mkdir -p /path/to/data/kmin_dataset
tar -xf kmin_auto_batch.tar -C /path/to/data/kmin_dataset
```

Each extracted HDF5 file (`task{N}_kmin{N+1}_kmax-auto_128-256-512_merged.h5`) contains:
- Groups: `128x128`, `256x256`, `512x512`
- Dataset: `{res}/images` — 1000 log10-transformed float32 images
- Dataset: `{res}/parameters/k_min` — regression target (constant per file)

## Quick Start

```python
from fractal_classifier_02.config import load_config
from fractal_classifier_02.training import FractalRegressorTrainer

config = load_config('fractal_classifier_02/config/default_config.yaml')
trainer = FractalRegressorTrainer(config)
trainer.train()
```

## Directory Structure

```
fractal_classifier_02/
├── config/          # YAML config system
├── data/            # HDF5 data loader + tf.data pipeline
├── models/          # Lightweight depthwise-separable CNN
├── training/        # Trainer, callbacks
├── evaluation/      # Regression metrics (MAE, RMSE, R²)
├── utils/           # GPU setup, patch extraction
└── visualization/   # Training curves, scatter/residual plots
```

## CLI

```bash
cd fractal_classifier_02
python scripts/train.py --config config/default_config.yaml
python scripts/evaluate.py --config config/default_config.yaml --checkpoint outputs/checkpoints/best.keras
```
