"""
Kernel extraction, export, and analysis utilities.

This module provides functions for:
- Extracting convolutional layer weights from models
- Saving kernel weights to disk (HDF5, NPZ formats)
- Analyzing kernel convergence during training
- Computing similarity metrics between kernel sets
"""

import os
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras


def extract_conv_kernels(model: keras.Model) -> Dict[str, np.ndarray]:
    """
    Extract all convolutional layer kernels from a model.
    
    Args:
        model: Keras model to extract kernels from
        
    Returns:
        Dictionary mapping layer names to kernel weights (numpy arrays)
        Format: {layer_name: kernel_array}
        Kernel shape: (height, width, in_channels, out_channels)
    """
    kernels = {}
    
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            # Get the kernel weights (first element of layer.get_weights())
            weights = layer.get_weights()
            if len(weights) > 0:
                kernel = weights[0]  # Shape: (h, w, in_c, out_c)
                kernels[layer.name] = kernel
                
    return kernels


def extract_all_layer_weights(model: keras.Model) -> Dict[str, List[np.ndarray]]:
    """
    Extract all trainable weights from a model (not just conv layers).
    
    Args:
        model: Keras model to extract weights from
        
    Returns:
        Dictionary mapping layer names to list of weight arrays
        Format: {layer_name: [kernel, bias, ...]}
    """
    all_weights = {}
    
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            all_weights[layer.name] = weights
            
    return all_weights


def save_kernels_hdf5(
    kernels: Dict[str, np.ndarray],
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save kernel weights to HDF5 file.
    
    Args:
        kernels: Dictionary of layer_name -> kernel_array
        filepath: Path to save HDF5 file
        metadata: Optional metadata dictionary to save as attributes
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Save metadata as root attributes
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value
        
        # Save each layer's kernels as a dataset
        for layer_name, kernel in kernels.items():
            dataset = f.create_dataset(
                layer_name,
                data=kernel,
                compression='gzip',
                compression_opts=4
            )
            # Add layer-specific attributes
            dataset.attrs['shape'] = kernel.shape
            dataset.attrs['dtype'] = str(kernel.dtype)
            dataset.attrs['size'] = kernel.size


def save_kernels_npz(
    kernels: Dict[str, np.ndarray],
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save kernel weights to compressed NPZ file.
    
    Args:
        kernels: Dictionary of layer_name -> kernel_array
        filepath: Path to save NPZ file
        metadata: Optional metadata dictionary (saved as separate array)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for npz (can only save arrays, not mixed types)
    save_dict = dict(kernels)
    
    # Save metadata as a structured array if provided
    if metadata:
        # Convert metadata to strings for safe storage
        metadata_str = {k: str(v) for k, v in metadata.items()}
        save_dict['_metadata'] = np.array([metadata_str], dtype=object)
    
    np.savez_compressed(filepath, **save_dict)


def load_kernels_hdf5(filepath: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load kernel weights from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Tuple of (kernels_dict, metadata_dict)
    """
    kernels = {}
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata from root attributes
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
        
        # Load each dataset
        for layer_name in f.keys():
            kernels[layer_name] = f[layer_name][()]
    
    return kernels, metadata


def load_kernels_npz(filepath: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load kernel weights from NPZ file.
    
    Args:
        filepath: Path to NPZ file
        
    Returns:
        Tuple of (kernels_dict, metadata_dict)
    """
    data = np.load(filepath, allow_pickle=True)
    
    kernels = {}
    metadata = {}
    
    for key in data.files:
        if key == '_metadata':
            # Extract metadata
            metadata_array = data[key]
            if len(metadata_array) > 0:
                metadata = dict(metadata_array[0])
        else:
            kernels[key] = data[key]
    
    return kernels, metadata


def compute_kernel_statistics(kernels: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each layer's kernels.
    
    Args:
        kernels: Dictionary of layer_name -> kernel_array
        
    Returns:
        Dictionary of layer_name -> statistics_dict
        Statistics include: mean, std, min, max, l1_norm, l2_norm
    """
    stats = {}
    
    for layer_name, kernel in kernels.items():
        stats[layer_name] = {
            'mean': float(np.mean(kernel)),
            'std': float(np.std(kernel)),
            'min': float(np.min(kernel)),
            'max': float(np.max(kernel)),
            'l1_norm': float(np.sum(np.abs(kernel))),
            'l2_norm': float(np.sqrt(np.sum(kernel ** 2))),
            'shape': kernel.shape,
            'num_params': kernel.size
        }
    
    return stats


def compute_kernel_similarity(
    kernels1: Dict[str, np.ndarray],
    kernels2: Dict[str, np.ndarray],
    metric: str = 'cosine'
) -> Dict[str, float]:
    """
    Compute similarity between two sets of kernels (layer by layer).
    
    Args:
        kernels1: First set of kernels
        kernels2: Second set of kernels
        metric: Similarity metric ('cosine', 'euclidean', 'correlation')
        
    Returns:
        Dictionary mapping layer_name to similarity score
    """
    similarities = {}
    
    # Get common layers
    common_layers = set(kernels1.keys()) & set(kernels2.keys())
    
    for layer_name in common_layers:
        k1 = kernels1[layer_name].flatten()
        k2 = kernels2[layer_name].flatten()
        
        if metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(k1, k2)
            norm1 = np.linalg.norm(k1)
            norm2 = np.linalg.norm(k2)
            similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
            
        elif metric == 'euclidean':
            # Negative euclidean distance (higher is more similar)
            similarity = -np.linalg.norm(k1 - k2)
            
        elif metric == 'correlation':
            # Pearson correlation
            similarity = np.corrcoef(k1, k2)[0, 1]
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        similarities[layer_name] = float(similarity)
    
    return similarities


def compare_kernel_evolution(
    kernel_files: List[Union[str, Path]],
    output_dir: Union[str, Path],
    metrics: List[str] = ['cosine', 'euclidean', 'correlation']
) -> Dict[str, np.ndarray]:
    """
    Compare kernel evolution across multiple checkpoints.
    
    Args:
        kernel_files: List of paths to kernel files (in chronological order)
        output_dir: Directory to save comparison results
        metrics: List of similarity metrics to compute
        
    Returns:
        Dictionary with similarity matrices for each layer and metric
        Format: {f"{layer_name}_{metric}": similarity_matrix}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all kernel sets
    all_kernels = []
    all_metadata = []
    
    for filepath in kernel_files:
        filepath = Path(filepath)
        if filepath.suffix == '.h5':
            kernels, metadata = load_kernels_hdf5(filepath)
        elif filepath.suffix == '.npz':
            kernels, metadata = load_kernels_npz(filepath)
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")
        
        all_kernels.append(kernels)
        all_metadata.append(metadata)
    
    # Get common layers across all checkpoints
    common_layers = set(all_kernels[0].keys())
    for kernels in all_kernels[1:]:
        common_layers &= set(kernels.keys())
    common_layers = sorted(common_layers)
    
    # Compute pairwise similarities for each layer and metric
    results = {}
    n_checkpoints = len(all_kernels)
    
    for layer_name in common_layers:
        for metric in metrics:
            # Create similarity matrix
            sim_matrix = np.zeros((n_checkpoints, n_checkpoints))
            
            for i in range(n_checkpoints):
                for j in range(n_checkpoints):
                    if i == j:
                        sim_matrix[i, j] = 1.0 if metric in ['cosine', 'correlation'] else 0.0
                    else:
                        kernels_i = {layer_name: all_kernels[i][layer_name]}
                        kernels_j = {layer_name: all_kernels[j][layer_name]}
                        sim = compute_kernel_similarity(kernels_i, kernels_j, metric)
                        sim_matrix[i, j] = sim[layer_name]
            
            key = f"{layer_name}_{metric}"
            results[key] = sim_matrix
            
            # Save to file
            np.save(output_dir / f"{key}.npy", sim_matrix)
    
    return results


def export_model_kernels(
    model: keras.Model,
    output_dir: Union[str, Path],
    epoch: Optional[int] = None,
    format: str = 'hdf5',
    prefix: str = 'kernels'
) -> str:
    """
    Export all convolutional kernels from a model to disk.
    
    Args:
        model: Keras model to export kernels from
        output_dir: Directory to save kernel files
        epoch: Optional epoch number (added to filename)
        format: Export format ('hdf5', 'npz', or 'both')
        prefix: Filename prefix
        
    Returns:
        Path to saved file (or primary file if format='both')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract kernels
    kernels = extract_conv_kernels(model)
    
    # Prepare metadata
    metadata = {
        'epoch': epoch if epoch is not None else -1,
        'num_layers': len(kernels),
        'total_params': sum(k.size for k in kernels.values())
    }
    
    # Generate filename
    if epoch is not None:
        filename_base = f"{prefix}_epoch_{epoch:04d}"
    else:
        filename_base = f"{prefix}_final"
    
    saved_path = None
    
    if format in ['hdf5', 'both']:
        filepath = output_dir / f"{filename_base}.h5"
        save_kernels_hdf5(kernels, filepath, metadata)
        saved_path = str(filepath)
    
    if format in ['npz', 'both']:
        filepath = output_dir / f"{filename_base}.npz"
        save_kernels_npz(kernels, filepath, metadata)
        if saved_path is None:
            saved_path = str(filepath)
    
    return saved_path
