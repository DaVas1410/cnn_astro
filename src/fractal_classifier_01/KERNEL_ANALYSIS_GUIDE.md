# Kernel Convergence Analysis Guide

## Overview

The Fractal Classifier v0.1 package now includes comprehensive tools for studying CNN kernel convergence during training. This is particularly useful for understanding how your model learns feature detectors and comparing them with known reference kernels.

## Features

### 1. Kernel Export During Training
- **Automatic Export**: Save kernel weights at every epoch or specific intervals
- **Multiple Formats**: HDF5 (compressed) or NPZ formats
- **Minimal Overhead**: Efficient saving doesn't significantly slow training
- **Final Snapshot**: Always saves final converged kernels

### 2. Kernel Statistics Tracking
- **Layer-wise Statistics**: Mean, std, L1/L2 norms per layer
- **CSV Logging**: Compact statistics tracking without full weight storage
- **Evolution Plots**: Visualize how statistics change over training

### 3. Convergence Analysis
- **Similarity Metrics**: Cosine, Euclidean distance, Pearson correlation
- **Trajectory Plots**: See how kernels converge to final state
- **Checkpoint Comparison**: Compare any two training checkpoints
- **Reference Comparison**: Compare with theoretical or pre-trained kernels

## Configuration

### Enable Kernel Export

Add to your YAML configuration:

```yaml
training:
  callbacks:
    kernel_export:
      enabled: true
      frequency: "epoch"  # Save every epoch
      # Or save at specific epochs:
      # export_epochs: [1, 5, 10, 20, 30, 40, 50]
      format: "hdf5"  # "hdf5", "npz", or "both"

paths:
  kernel_weights_dir: "kernel_weights"
  kernel_analysis_dir: "kernel_analysis"
```

### Enable Kernel Analysis Visualization

```yaml
visualization:
  kernel_analysis:
    enabled: true
    plot_evolution: true
    compute_stats: true
    similarity_metrics: ["cosine", "euclidean", "correlation"]
    export_final_weights: true
```

## Usage

### 1. Train with Kernel Export

```python
from fractal_classifier import Config, FractalClassifierTrainer

# Load configuration with kernel export enabled
config = Config.from_yaml('examples/advanced_kernel_tracking.yaml')

# Train model - kernels will be saved automatically
trainer = FractalClassifierTrainer(config)
trainer.setup()
trainer.train()
```

This will create files like:
```
kernel_weights/
├── kernels_epoch_0001.h5
├── kernels_epoch_0002.h5
├── ...
├── kernels_epoch_0050.h5
└── kernels_final.h5
```

### 2. Load and Inspect Kernels

```python
from fractal_classifier.utils import load_kernels_hdf5, compute_kernel_statistics

# Load kernels from a checkpoint
kernels, metadata = load_kernels_hdf5('kernel_weights/kernels_epoch_0010.h5')

# Inspect available layers
print("Layers:", list(kernels.keys()))
# Output: ['conv2d', 'conv2d_1', 'conv2d_2', ...]

# Get statistics for each layer
stats = compute_kernel_statistics(kernels)
for layer_name, layer_stats in stats.items():
    print(f"{layer_name}:")
    print(f"  Shape: {layer_stats['shape']}")
    print(f"  Mean: {layer_stats['mean']:.6f}")
    print(f"  Std: {layer_stats['std']:.6f}")
    print(f"  L2 norm: {layer_stats['l2_norm']:.2f}")
```

### 3. Analyze Convergence

```python
from fractal_classifier.analysis import analyze_kernel_convergence

# Comprehensive analysis of all checkpoints
analyze_kernel_convergence(
    kernel_weights_dir='kernel_weights',
    output_dir='kernel_analysis',
    reference_kernels_path=None,  # Or path to reference kernels
    metrics=['cosine', 'euclidean', 'correlation'],
    dpi=300
)
```

This generates:
- **Convergence trajectory plots**: How similar kernels are to final state over time
- **Final statistics CSV**: Complete statistics for converged kernels
- **Similarity matrices**: Pairwise comparisons between checkpoints

### 4. Compare with Reference Kernels

If you have theoretical or reference kernels:

```python
from fractal_classifier.utils import load_kernels_hdf5
from fractal_classifier.analysis import compare_kernels_with_reference

# Load trained kernels
trained_kernels, _ = load_kernels_hdf5('kernel_weights/kernels_final.h5')

# Load reference kernels (e.g., from another model, theoretical kernels, etc.)
reference_kernels, _ = load_kernels_hdf5('reference_kernels.h5')

# Compare
comparison_df = compare_kernels_with_reference(
    trained_kernels=trained_kernels,
    reference_kernels=reference_kernels,
    output_dir='kernel_analysis'
)

# Results saved to: kernel_analysis/kernel_comparison_with_reference.csv
```

### 5. Visualize Kernel Differences

```python
from fractal_classifier.analysis import visualize_kernel_difference

# Compare two specific checkpoints
checkpoint_1, _ = load_kernels_hdf5('kernel_weights/kernels_epoch_0010.h5')
checkpoint_2, _ = load_kernels_hdf5('kernel_weights/kernels_final.h5')

# Visualize difference for first conv layer
layer_name = 'conv2d'
visualize_kernel_difference(
    kernel1=checkpoint_1[layer_name],
    kernel2=checkpoint_2[layer_name],
    layer_name=layer_name,
    output_path='kernel_analysis/kernel_diff_conv2d.png',
    max_filters=16  # Show first 16 filters
)
```

### 6. Plot Convergence Trajectory

```python
from fractal_classifier.analysis import plot_kernel_convergence_trajectory
from pathlib import Path

# Find all checkpoint files
kernel_files = sorted(Path('kernel_weights').glob('kernels_epoch_*.h5'))

# Plot convergence
plot_kernel_convergence_trajectory(
    kernel_files=kernel_files,
    reference_kernels=None,  # Uses final checkpoint as reference
    output_dir='kernel_analysis',
    metric='cosine',
    dpi=300
)
```

## Analysis Outputs

### Convergence Trajectory Plots
Shows how each layer's kernels evolve toward their final state:
- X-axis: Training epoch
- Y-axis: Similarity to final/reference kernels
- One line per conv layer

Interpretation:
- **Fast convergence**: Layer learns quickly (curve plateaus early)
- **Slow convergence**: Layer continues adapting (curve rises throughout training)
- **Layer differences**: Earlier layers often converge faster than deeper layers

### Similarity Metrics

1. **Cosine Similarity** (-1 to 1, higher = more similar)
   - Measures angle between kernel weight vectors
   - 1.0 = identical direction, 0 = orthogonal, -1 = opposite

2. **Euclidean Distance** (0 to ∞, lower = more similar)
   - Absolute difference between kernel weights
   - Sensitive to weight magnitudes

3. **Pearson Correlation** (-1 to 1, higher = more similar)
   - Linear relationship between kernel values
   - 1.0 = perfect positive correlation

### Final Kernel Statistics CSV

Example output:
```csv
layer,mean,std,min,max,l1_norm,l2_norm,shape,num_params
conv2d,-0.002,0.15,-0.8,0.7,85.3,12.4,"(3, 3, 1, 32)",288
conv2d_1,0.001,0.12,-0.6,0.6,245.7,24.8,"(3, 3, 32, 32)",9216
...
```

## Advanced Use Cases

### 1. Study Layer-Specific Convergence

```python
import numpy as np
from fractal_classifier.utils import load_kernels_hdf5, compute_kernel_similarity

# Load multiple checkpoints
epochs = [1, 10, 20, 30, 40, 50]
checkpoints = [load_kernels_hdf5(f'kernel_weights/kernels_epoch_{e:04d}.h5')[0] 
               for e in epochs]

# Focus on one layer
layer_name = 'conv2d_2'  # Third conv layer

# Compute pairwise similarities
n = len(checkpoints)
sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            sim_matrix[i, j] = 1.0
        else:
            k1 = {layer_name: checkpoints[i][layer_name]}
            k2 = {layer_name: checkpoints[j][layer_name]}
            sim = compute_kernel_similarity(k1, k2, metric='cosine')
            sim_matrix[i, j] = sim[layer_name]

print(f"Convergence matrix for {layer_name}:")
print(sim_matrix)
```

### 2. Compare with Known Filter Types

If you have theoretical filters (e.g., Gabor filters, edge detectors):

```python
import h5py
import numpy as np

# Create reference kernels (example: edge detectors)
def create_edge_detector_kernel(size=3):
    """Create a simple edge detection kernel."""
    kernel = np.zeros((size, size, 1, 1))
    kernel[:, 0, 0, 0] = -1
    kernel[:, -1, 0, 0] = 1
    return kernel

# Save reference kernels
with h5py.File('reference_edge_detectors.h5', 'w') as f:
    f.create_dataset('edge_detector', data=create_edge_detector_kernel())

# Now compare with trained kernels
from fractal_classifier.utils import load_kernels_hdf5
from fractal_classifier.analysis import compare_kernels_with_reference

trained, _ = load_kernels_hdf5('kernel_weights/kernels_final.h5')
reference, _ = load_kernels_hdf5('reference_edge_detectors.h5')

compare_kernels_with_reference(trained, reference, 'kernel_analysis')
```

### 3. Export for External Analysis

```python
# Export to NumPy arrays for use in other tools
from fractal_classifier.utils import load_kernels_hdf5
import numpy as np

kernels, metadata = load_kernels_hdf5('kernel_weights/kernels_final.h5')

# Export each layer to separate .npy file
for layer_name, kernel_array in kernels.items():
    np.save(f'exported_kernels/{layer_name}.npy', kernel_array)
    print(f"Exported {layer_name}: shape {kernel_array.shape}")
```

## CLI Integration

The analyze script supports kernel analysis:

```bash
# Full model analysis including kernels
python -m fractal_classifier.scripts.analyze \
    --history outputs/training_history.csv \
    --model checkpoints/best_model.h5 \
    --data path/to/dataset.h5 \
    --visualize-kernels \
    --analyze-convergence \
    --kernel-weights-dir kernel_weights \
    --reference-kernels path/to/reference.h5
```

## Performance Considerations

### Storage Requirements

- **HDF5 format**: ~1-5 MB per checkpoint (compressed)
- **NPZ format**: ~2-8 MB per checkpoint
- **50 epochs**: ~50-250 MB total

Tips:
- Use `export_epochs` to save only specific epochs
- Use `frequency` to save every N epochs
- Use HDF5 format for better compression

### Training Overhead

- **Kernel export**: < 1 second per epoch
- **Statistics logging**: < 0.1 seconds per epoch
- **Minimal impact** on overall training time

## Troubleshooting

### Out of Memory

If kernel export causes memory issues:
1. Reduce `export_epochs` to save fewer checkpoints
2. Increase `frequency` to save less often
3. Use NPZ format (sometimes more memory-efficient)

### Missing Layers

If some layers don't appear in exports:
- Only Conv2D layers are automatically exported
- To export all layers, use `extract_all_layer_weights()` 

### File Not Found

Ensure paths in config match your directory structure:
```yaml
paths:
  kernel_weights_dir: "kernel_weights"  # Will create if doesn't exist
```

## References

For understanding kernel convergence:
- Zeiler & Fergus (2014): "Visualizing and Understanding Convolutional Networks"
- Bau et al. (2017): "Network Dissection"
- Li et al. (2018): "Measuring the Intrinsic Dimension of Objective Landscapes"

## Example Workflow

Complete workflow for kernel convergence study:

```bash
# 1. Train with kernel export enabled
python -m fractal_classifier.scripts.train \
    --config examples/advanced_kernel_tracking.yaml \
    --data-path fractal_dataset.h5

# 2. Analyze convergence
python -c "
from fractal_classifier.analysis import analyze_kernel_convergence
analyze_kernel_convergence(
    'kernel_weights',
    'kernel_analysis',
    dpi=300
)
"

# 3. View results
ls kernel_analysis/
# kernel_convergence_cosine.png
# kernel_convergence_euclidean.png
# kernel_convergence_correlation.png
# final_kernel_statistics.csv
```
