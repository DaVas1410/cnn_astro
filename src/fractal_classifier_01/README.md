# Fractal Classifier v0.1

A modular deep learning framework for classifying synthetic fractal cloud patterns based on turbulence parameters (k-values). This package provides a complete pipeline from data loading to model training, evaluation, and analysis.

## Features

- **Modular Architecture**: Clean separation of concerns with distinct modules for data, models, training, evaluation, and visualization
- **Configuration-Driven**: YAML-based configuration for all hyperparameters and settings
- **GPU Optimization**: Built-in support for mixed precision training and memory management
- **Lazy Data Loading**: Efficient HDF5 data loading without loading entire datasets into memory
- **Centroid Predictions**: Novel interpolation method for predicting intermediate k-values
- **Rich Visualization**: Training curves, confusion matrices, activation maps, and kernel weights
- **CLI Tools**: Command-line scripts for training, evaluation, and analysis
- **Python API**: Programmatic access to all functionality
- **🆕 Kernel Convergence Analysis**: Track and analyze CNN kernel evolution during training
- **🆕 Advanced Training Options**: Enhanced data augmentation, regularization, and optimizer configurations
- **🆕 TensorBoard Integration**: Real-time training monitoring and visualization
- **🆕 Reference Kernel Comparison**: Compare learned kernels with theoretical or known filters

## Installation

### Requirements

- Python 3.8 or higher
- TensorFlow 2.10 or higher
- CUDA-compatible GPU (recommended)

### Install from source

```bash
cd fractal_classifier
pip install -e .
```

This will install the package in editable mode along with all dependencies.

### Verify installation

```python
import fractal_classifier
print(fractal_classifier.__version__)  # Should print: 0.1.0
```

## Quick Start

### 1. Prepare Your Data

Organize your fractal cloud images in an HDF5 file with this structure:

```
dataset.h5
├── k_2/
│   └── images [N, 512, 512]  # float32 arrays
├── k_4/
│   └── images [M, 512, 512]
├── k_8/
│   └── images [P, 512, 512]
...
```

### 2. Training (CLI)

Train a model using the default configuration:

```bash
python -m fractal_classifier.scripts.train \
    --data-path /path/to/dataset.h5 \
    --output-dir outputs \
    --epochs 50
```

Or with a custom configuration file:

```bash
python -m fractal_classifier.scripts.train \
    --config my_config.yaml \
    --data-path /path/to/dataset.h5
```

### 3. Evaluation (CLI)

Evaluate a trained model:

```bash
python -m fractal_classifier.scripts.evaluate \
    --model checkpoints/best_model.h5 \
    --data /path/to/test_dataset.h5 \
    --output-dir evaluation_results \
    --plots \
    --confusion-matrix
```

### 4. Analysis (CLI)

Analyze training history and visualize model:

```bash
python -m fractal_classifier.scripts.analyze \
    --history outputs/training_history.csv \
    --model checkpoints/best_model.h5 \
    --data /path/to/dataset.h5 \
    --output-dir analysis_results \
    --visualize-kernels \
    --visualize-activations
```

## Python API Usage

### Training Example

```python
from fractal_classifier import load_config, FractalClassifierTrainer

# Load configuration
config = load_config('config.yaml')

# Or create config programmatically
config = load_config()  # Uses defaults
config.data.h5_path = '/path/to/dataset.h5'
config.training.epochs = 50
config.training.batch_size = 4

# Initialize trainer
trainer = FractalClassifierTrainer(config, verbose=True)

# Setup (GPU, data, model)
trainer.setup()

# Display model architecture
trainer.print_summary()

# Train
history = trainer.train()

# Save results
trainer.save_history()

# Evaluate on test set
test_results = trainer.evaluate('test')
print(f"Test accuracy: {test_results['sparse_categorical_accuracy']:.4f}")
```

### Evaluation Example

```python
from fractal_classifier import load_model, FractalEvaluator
import numpy as np

# Load trained model
model = load_model('checkpoints/best_model.h5')

# Initialize evaluator
k_values = np.array([2, 4, 8, 16, 32, 64])
evaluator = FractalEvaluator(model, k_values, batch_size=8)

# Evaluate on dataset
results_df = evaluator.evaluate_on_hdf5('test_dataset.h5')

# Print summary
evaluator.print_evaluation_summary(results_df)

# Compute confusion matrix
cm, y_true, y_pred = evaluator.compute_confusion_matrix(
    'test_dataset.h5',
    samples_per_class=20
)
```

### Custom Model Building

```python
from fractal_classifier.models import build_fractal_cnn

model = build_fractal_cnn(
    input_shape=(512, 512, 1),
    num_classes=6,
    filters=[32, 64, 128, 256],
    dense_units=64,
    dropout_rate=0.5
)

model.summary()
```

## Configuration

The configuration system uses YAML files to specify all hyperparameters. Here's a minimal example:

```yaml
# config.yaml
data:
  h5_path: "fractal_dataset.h5"
  input_shape: [512, 512, 1]
  num_classes: 6

model:
  filters: [32, 64, 128, 256]
  dense_units: 64
  dropout_rate: 0.5

training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.0001

paths:
  output_dir: "outputs"
  checkpoint_dir: "checkpoints"
```

See `fractal_classifier/config/default_config.yaml` for the full configuration template with all options.

## Project Structure

```
fractal_classifier/
├── __init__.py           # Package initialization
├── config/               # Configuration management
│   ├── config.py
│   └── default_config.yaml
├── data/                 # Data loading and pipelines
│   ├── loader.py
│   └── pipeline.py
├── models/               # Model architectures
│   └── cnn.py
├── training/             # Training orchestration
│   └── trainer.py
├── evaluation/           # Evaluation and metrics
│   ├── evaluator.py
│   └── metrics.py
├── visualization/        # Plotting utilities
│   ├── plots.py
│   ├── activations.py
│   └── kernels.py
├── utils/                # Utility functions
│   └── gpu.py
└── scripts/              # CLI entry points
    ├── train.py
    ├── evaluate.py
    └── analyze.py
```

## Model Architecture

The default model is a 4-block CNN:

```
Input (512, 512, 1)
├── Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool2D
├── Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool2D
├── Block 3: Conv2D(128) -> Conv2D(128) -> MaxPool2D
├── Block 4: Conv2D(256) -> Conv2D(256) -> MaxPool2D
├── GlobalAveragePooling2D
├── Dense(64) -> Dropout(0.5)
└── Dense(num_classes, softmax)
```

All hyperparameters are configurable via the config system.

## Centroid Prediction Method

This package implements a novel centroid-based prediction method for interpolating between discrete training classes:

```
centroid_k = sum(probability_i * k_value_i) for all classes
```

This allows the model to predict continuous k-values and generalize to intermediate values not seen during training.

## Advanced Features

### Mixed Precision Training

Automatically enabled by default for faster training on modern GPUs:

```yaml
gpu:
  mixed_precision:
    enabled: true
    policy: "mixed_float16"
```

### Custom Data Augmentation

Modify augmentation in the config:

```yaml
training:
  augmentation:
    horizontal_flip: true
    vertical_flip: true
```

### Resume Training

```bash
python -m fractal_classifier.scripts.train \
    --config config.yaml \
    --resume-from checkpoints/best_model.h5
```

## Command-Line Reference

### train.py

```bash
python -m fractal_classifier.scripts.train [OPTIONS]

Options:
  --config, -c PATH         Configuration file
  --data-path, -d PATH      HDF5 dataset path
  --output-dir, -o PATH     Output directory
  --epochs, -e INT          Number of epochs
  --batch-size, -b INT      Batch size
  --learning-rate, -lr FLOAT Learning rate
  --resume-from PATH        Resume from checkpoint
  --no-gpu                  Use CPU only
```

### evaluate.py

```bash
python -m fractal_classifier.scripts.evaluate [OPTIONS]

Options:
  --model, -m PATH          Model file (.h5)
  --data, -d PATH           HDF5 dataset
  --output-dir, -o PATH     Output directory
  --k-values INT [INT ...]  K-values for classes
  --batch-size, -b INT      Batch size
  --intermediate            Intermediate k evaluation
  --confusion-matrix        Generate confusion matrix
  --plots                   Generate plots
```

### analyze.py

```bash
python -m fractal_classifier.scripts.analyze [OPTIONS]

Options:
  --history PATH            Training history CSV
  --model, -m PATH          Model file (.h5)
  --data PATH               HDF5 dataset
  --output-dir, -o PATH     Output directory
  --visualize-kernels       Visualize kernels
  --visualize-activations   Visualize activations
  --dpi INT                 Plot DPI
  --formats [png|pdf ...]   Output formats
```

## Kernel Convergence Analysis

Track and analyze how CNN kernels evolve during training. This is useful for understanding feature learning and comparing with theoretical or reference filters.

### Enable Kernel Export

Add to your configuration:

```yaml
training:
  callbacks:
    kernel_export:
      enabled: true
      frequency: "epoch"  # Save every epoch
      format: "hdf5"

paths:
  kernel_weights_dir: "kernel_weights"
```

### Analyze Convergence

```python
from fractal_classifier.analysis import analyze_kernel_convergence

# Comprehensive convergence analysis
analyze_kernel_convergence(
    kernel_weights_dir='kernel_weights',
    output_dir='kernel_analysis',
    reference_kernels_path='reference.h5',  # Optional
    metrics=['cosine', 'euclidean', 'correlation']
)
```

This generates:
- **Convergence trajectory plots**: Similarity to final state over training
- **Layer statistics**: Mean, std, L1/L2 norms for each layer
- **Comparison with reference**: If reference kernels provided

### Compare with Reference Kernels

```python
from fractal_classifier.utils import load_kernels_hdf5
from fractal_classifier.analysis import compare_kernels_with_reference

trained, _ = load_kernels_hdf5('kernel_weights/kernels_final.h5')
reference, _ = load_kernels_hdf5('reference_kernels.h5')

comparison_df = compare_kernels_with_reference(
    trained_kernels=trained,
    reference_kernels=reference,
    output_dir='kernel_analysis'
)
```

### Example Configuration

See `examples/advanced_kernel_tracking.yaml` for a complete configuration with:
- Enhanced data augmentation (rotation, zoom, brightness)
- Regularization options (L1/L2, dropout, weight decay)
- TensorBoard logging
- Kernel export and convergence tracking

**For detailed documentation, see [KERNEL_ANALYSIS_GUIDE.md](KERNEL_ANALYSIS_GUIDE.md)**

## Tips and Best Practices

1. **Memory Management**: Start with small batch sizes (4 or 8) and increase if GPU memory allows
2. **Learning Rate**: Default 1e-4 works well; reduce if training is unstable
3. **Early Stopping**: Enabled by default with patience=10 to prevent overfitting
4. **Data Splits**: Default 70/15/15 train/val/test split with stratification
5. **Mixed Precision**: Keep enabled for ~2x speedup on modern GPUs
6. **Checkpoints**: Best model is automatically saved based on validation accuracy
7. **🆕 Kernel Tracking**: Enable for research to study feature learning convergence
8. **🆕 TensorBoard**: Use for real-time training monitoring and debugging

## Troubleshooting

### Out of Memory Errors

Reduce batch size:
```yaml
training:
  batch_size: 2  # or even 1
```

Disable mixed precision:
```yaml
gpu:
  mixed_precision:
    enabled: false
```

### Slow Data Loading

Increase parallel workers (if sufficient RAM):
```yaml
training:
  pipeline:
    num_parallel_calls: 4
    prefetch: 2
```

### Model Not Converging

Try reducing learning rate:
```yaml
training:
  learning_rate: 0.00001  # 1e-5
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{fractal_classifier2026,
  title={Fractal Classifier: Deep Learning for Fractal Cloud Classification},
  author={Your Name},
  year={2026},
  version={0.1.0},
  url={https://github.com/yourusername/fractal-classifier}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the authors.

## Acknowledgments

This package implements CNN-based classification of synthetic fractal clouds generated using the pyFC library (Lewis & Austin 2002 method).
