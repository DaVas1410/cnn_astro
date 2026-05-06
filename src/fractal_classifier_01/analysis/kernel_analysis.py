"""
Kernel convergence analysis and visualization utilities.

This module provides functions for analyzing how CNN kernels evolve
during training and comparing them with known/reference kernels.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from ..utils.kernels import (
    load_kernels_hdf5,
    load_kernels_npz,
    compute_kernel_statistics,
    compute_kernel_similarity
)


def plot_kernel_statistics_evolution(
    stats_csv: Union[str, Path],
    output_dir: Union[str, Path],
    dpi: int = 300
) -> None:
    """
    Plot evolution of kernel statistics over training epochs.
    
    Args:
        stats_csv: Path to CSV file with kernel statistics (from KernelStatisticsCallback)
        output_dir: Directory to save plots
        dpi: Resolution for saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load statistics
    df = pd.read_csv(stats_csv)
    epochs = df['epoch'].values
    
    # Extract layer names
    layer_names = []
    for col in df.columns:
        if col.endswith('_mean'):
            layer_name = col.replace('_mean', '')
            layer_names.append(layer_name)
    
    # Plot for each statistic type
    stat_types = ['mean', 'std', 'l2_norm']
    
    for stat_type in stat_types:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for layer_name in layer_names:
            col_name = f'{layer_name}_{stat_type}'
            if col_name in df.columns:
                values = df[col_name].values
                ax.plot(epochs, values, marker='o', label=layer_name, alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(stat_type.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Kernel {stat_type.replace("_", " ").title()} Evolution', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'kernel_{stat_type}_evolution.png', dpi=dpi, bbox_inches='tight')
        plt.close()


def plot_kernel_similarity_matrix(
    similarity_matrix: np.ndarray,
    checkpoint_labels: List[str],
    layer_name: str,
    metric: str,
    output_path: Union[str, Path],
    dpi: int = 300
) -> None:
    """
    Plot similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: NxN similarity matrix
        checkpoint_labels: Labels for each checkpoint
        layer_name: Name of the layer
        metric: Similarity metric used
        output_path: Path to save figure
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        similarity_matrix,
        xticklabels=checkpoint_labels,
        yticklabels=checkpoint_labels,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0 if metric == 'euclidean' else 0.5,
        ax=ax,
        cbar_kws={'label': f'{metric.title()} Similarity'}
    )
    
    ax.set_title(f'Kernel Similarity: {layer_name}\n({metric.title()} metric)', fontsize=14)
    ax.set_xlabel('Checkpoint', fontsize=12)
    ax.set_ylabel('Checkpoint', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_kernel_convergence_trajectory(
    kernel_files: List[Union[str, Path]],
    reference_kernels: Optional[Dict[str, np.ndarray]],
    output_dir: Union[str, Path],
    metric: str = 'cosine',
    dpi: int = 300
) -> None:
    """
    Plot convergence trajectory by computing similarity to final/reference kernels.
    
    Args:
        kernel_files: List of kernel files in chronological order
        reference_kernels: Optional reference kernels to compare against (if None, uses final)
        output_dir: Directory to save plots
        metric: Similarity metric to use
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all kernel sets
    all_kernels = []
    epochs = []
    
    for filepath in kernel_files:
        filepath = Path(filepath)
        if filepath.suffix == '.h5':
            kernels, metadata = load_kernels_hdf5(filepath)
        elif filepath.suffix == '.npz':
            kernels, metadata = load_kernels_npz(filepath)
        else:
            continue
        
        all_kernels.append(kernels)
        epochs.append(metadata.get('epoch', -1))
    
    if len(all_kernels) == 0:
        print("No kernel files found to analyze")
        return
    
    # Use final checkpoint as reference if not provided
    if reference_kernels is None:
        reference_kernels = all_kernels[-1]
    
    # Get common layers
    common_layers = set(all_kernels[0].keys())
    for kernels in all_kernels[1:]:
        common_layers &= set(kernels.keys())
    common_layers = sorted(common_layers)
    
    # Compute similarities over time for each layer
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for layer_name in common_layers:
        similarities = []
        
        for kernels in all_kernels:
            kernels_subset = {layer_name: kernels[layer_name]}
            ref_subset = {layer_name: reference_kernels[layer_name]}
            
            sim = compute_kernel_similarity(kernels_subset, ref_subset, metric)
            similarities.append(sim[layer_name])
        
        ax.plot(epochs, similarities, marker='o', label=layer_name, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'{metric.title()} Similarity to Reference', fontsize=12)
    ax.set_title('Kernel Convergence Trajectory', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'kernel_convergence_{metric}.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def compare_kernels_with_reference(
    trained_kernels: Dict[str, np.ndarray],
    reference_kernels: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    layer_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare trained kernels with reference kernels using multiple metrics.
    
    Args:
        trained_kernels: Kernels from trained model
        reference_kernels: Reference kernels to compare against
        output_dir: Directory to save results
        layer_names: Specific layers to compare (None for all common layers)
        
    Returns:
        DataFrame with comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get layers to compare
    if layer_names is None:
        layer_names = sorted(set(trained_kernels.keys()) & set(reference_kernels.keys()))
    
    # Compute multiple similarity metrics
    metrics = ['cosine', 'euclidean', 'correlation']
    results = []
    
    for layer_name in layer_names:
        if layer_name not in trained_kernels or layer_name not in reference_kernels:
            continue
        
        result_row = {'layer': layer_name}
        
        for metric in metrics:
            trained_subset = {layer_name: trained_kernels[layer_name]}
            ref_subset = {layer_name: reference_kernels[layer_name]}
            
            sim = compute_kernel_similarity(trained_subset, ref_subset, metric)
            result_row[f'{metric}_similarity'] = sim[layer_name]
        
        # Add shape information
        result_row['shape'] = str(trained_kernels[layer_name].shape)
        result_row['num_params'] = trained_kernels[layer_name].size
        
        results.append(result_row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = output_dir / 'kernel_comparison_with_reference.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Kernel comparison saved to: {csv_path}")
    print("\nSummary:")
    print(df.to_string(index=False))
    
    return df


def visualize_kernel_difference(
    kernel1: np.ndarray,
    kernel2: np.ndarray,
    layer_name: str,
    output_path: Union[str, Path],
    max_filters: int = 16,
    dpi: int = 150
) -> None:
    """
    Visualize the difference between two kernel sets for a layer.
    
    Args:
        kernel1: First kernel array (h, w, in_c, out_c)
        kernel2: Second kernel array (same shape as kernel1)
        layer_name: Name of the layer
        output_path: Path to save figure
        max_filters: Maximum number of filters to visualize
        dpi: Resolution
    """
    if kernel1.shape != kernel2.shape:
        raise ValueError(f"Kernel shapes must match: {kernel1.shape} vs {kernel2.shape}")
    
    # Compute difference
    diff = kernel1 - kernel2
    
    # Select first few filters
    num_filters = min(max_filters, kernel1.shape[-1])
    
    # Create figure with subplots for each filter
    ncols = 4
    nrows = (num_filters + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols * 3, figsize=(4 * ncols * 3, 4 * nrows))
    axes = axes.flatten()
    
    for i in range(num_filters):
        # Get first channel of each filter for visualization
        k1 = kernel1[:, :, 0, i]
        k2 = kernel2[:, :, 0, i]
        d = diff[:, :, 0, i]
        
        # Normalize for visualization
        vmin = min(k1.min(), k2.min())
        vmax = max(k1.max(), k2.max())
        
        # Plot kernel 1
        ax_idx = i * 3
        im1 = axes[ax_idx].imshow(k1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[ax_idx].set_title(f'Filter {i+1} - Kernel 1')
        axes[ax_idx].axis('off')
        plt.colorbar(im1, ax=axes[ax_idx], fraction=0.046)
        
        # Plot kernel 2
        ax_idx = i * 3 + 1
        im2 = axes[ax_idx].imshow(k2, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[ax_idx].set_title(f'Filter {i+1} - Kernel 2')
        axes[ax_idx].axis('off')
        plt.colorbar(im2, ax=axes[ax_idx], fraction=0.046)
        
        # Plot difference
        ax_idx = i * 3 + 2
        im3 = axes[ax_idx].imshow(d, cmap='seismic', vmin=-abs(d).max(), vmax=abs(d).max())
        axes[ax_idx].set_title(f'Filter {i+1} - Difference')
        axes[ax_idx].axis('off')
        plt.colorbar(im3, ax=axes[ax_idx], fraction=0.046)
    
    # Hide unused subplots
    for i in range(num_filters * 3, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'Kernel Comparison: {layer_name}', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def analyze_kernel_convergence(
    kernel_weights_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reference_kernels_path: Optional[Union[str, Path]] = None,
    metrics: List[str] = ['cosine', 'euclidean', 'correlation'],
    dpi: int = 300
) -> None:
    """
    Comprehensive kernel convergence analysis.
    
    This function:
    1. Loads all kernel checkpoints
    2. Computes statistics and similarities
    3. Generates convergence plots
    4. Compares with reference if provided
    
    Args:
        kernel_weights_dir: Directory containing kernel checkpoint files
        output_dir: Directory to save analysis results
        reference_kernels_path: Optional path to reference kernels
        metrics: List of similarity metrics to use
        dpi: Resolution for plots
    """
    kernel_weights_dir = Path(kernel_weights_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing kernel convergence from: {kernel_weights_dir}")
    
    # Find all kernel checkpoint files
    kernel_files = sorted(kernel_weights_dir.glob('kernels_epoch_*.h5'))
    kernel_files += sorted(kernel_weights_dir.glob('kernels_epoch_*.npz'))
    
    if len(kernel_files) == 0:
        print("No kernel checkpoint files found!")
        return
    
    print(f"Found {len(kernel_files)} checkpoint files")
    
    # Load reference kernels if provided
    reference_kernels = None
    if reference_kernels_path:
        reference_path = Path(reference_kernels_path)
        if reference_path.suffix == '.h5':
            reference_kernels, _ = load_kernels_hdf5(reference_path)
        elif reference_path.suffix == '.npz':
            reference_kernels, _ = load_kernels_npz(reference_path)
        print(f"Loaded reference kernels from: {reference_path}")
    
    # Generate convergence trajectory plot
    print("Plotting convergence trajectory...")
    for metric in metrics:
        plot_kernel_convergence_trajectory(
            kernel_files,
            reference_kernels,
            output_dir,
            metric=metric,
            dpi=dpi
        )
    
    # Load final kernels for comparison
    final_file = kernel_weights_dir / 'kernels_final.h5'
    if not final_file.exists():
        final_file = kernel_weights_dir / 'kernels_final.npz'
    
    if final_file.exists():
        if final_file.suffix == '.h5':
            final_kernels, _ = load_kernels_hdf5(final_file)
        else:
            final_kernels, _ = load_kernels_npz(final_file)
        
        # Compute and save statistics
        stats = compute_kernel_statistics(final_kernels)
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv(output_dir / 'final_kernel_statistics.csv')
        print(f"\nFinal kernel statistics saved to: {output_dir / 'final_kernel_statistics.csv'}")
        
        # Compare with reference if available
        if reference_kernels:
            print("\nComparing with reference kernels...")
            compare_kernels_with_reference(
                final_kernels,
                reference_kernels,
                output_dir
            )
    
    print(f"\nKernel convergence analysis complete. Results saved to: {output_dir}")
