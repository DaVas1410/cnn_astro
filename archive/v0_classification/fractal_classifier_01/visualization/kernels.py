"""Kernel (weight) visualization for Fractal Classifier."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def visualize_kernels(model,
                     output_dir: Optional[str] = None,
                     layer_names: Optional[list] = None,
                     dpi: int = 150) -> None:
    """
    Visualize convolutional kernel weights.
    
    Args:
        model: Trained Keras model
        output_dir: Directory to save kernel visualizations
        layer_names: List of layer names to visualize (default: all conv layers)
        dpi: DPI for saved images
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if layer_names is None:
        # Get all convolutional layers
        layer_names = [layer.name for layer in model.layers 
                      if isinstance(layer, tf.keras.layers.Conv2D)]
    
    if not layer_names:
        logger.warning("No convolutional layers found in model")
        return
    
    logger.info(f"Visualizing kernels for {len(layer_names)} layers")
    
    for layer_name in layer_names:
        try:
            layer = model.get_layer(layer_name)
            weights = layer.get_weights()[0]  # Shape: (kh, kw, in_channels, out_channels)
            
            kh, kw, in_ch, out_ch = weights.shape
            
            # Visualize first 16 filters (or fewer if layer has less)
            n_filters = min(16, out_ch)
            n_cols = 4
            n_rows = (n_filters + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5*n_rows))
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
            for i in range(n_filters):
                ax = axes[i]
                
                # Get kernel for this filter
                kernel = weights[:, :, :, i]
                
                # If multiple input channels, show average or first channel
                if in_ch > 1:
                    kernel = kernel.mean(axis=2)
                else:
                    kernel = kernel[:, :, 0]
                
                # Normalize for visualization
                vmin, vmax = kernel.min(), kernel.max()
                if vmax > vmin:
                    kernel_norm = (kernel - vmin) / (vmax - vmin)
                else:
                    kernel_norm = kernel
                
                ax.imshow(kernel_norm, cmap='gray', interpolation='nearest')
                ax.set_title(f'Filter {i}', fontsize=8)
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(n_filters, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Kernels: {layer_name} ({kh}x{kw}, {in_ch}->{out_ch})', 
                        fontsize=11, fontweight='bold')
            plt.tight_layout()
            
            if output_dir:
                layer_output_dir = output_dir / layer_name
                layer_output_dir.mkdir(parents=True, exist_ok=True)
                save_path = layer_output_dir / 'kernels.png'
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                logger.debug(f"Saved kernels to {save_path}")
            
            plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error visualizing kernels for layer {layer_name}: {e}")
            continue
    
    if output_dir:
        logger.info(f"Saved kernel visualizations to {output_dir}")


def visualize_first_layer_kernels(model,
                                  output_path: Optional[str] = None,
                                  dpi: int = 150,
                                  max_filters: int = 32) -> None:
    """
    Visualize first layer kernels (often interpretable).
    
    Args:
        model: Trained Keras model
        output_path: Path to save visualization
        dpi: DPI for saved image
        max_filters: Maximum number of filters to show
    """
    # Find first convolutional layer
    first_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            first_conv = layer
            break
    
    if first_conv is None:
        logger.warning("No convolutional layer found in model")
        return
    
    weights = first_conv.get_weights()[0]
    kh, kw, in_ch, out_ch = weights.shape
    
    n_filters = min(max_filters, out_ch)
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 1.5*n_rows))
    axes = axes.flatten()
    
    for i in range(n_filters):
        ax = axes[i]
        kernel = weights[:, :, :, i]
        
        # Handle grayscale vs RGB
        if in_ch == 1:
            kernel_viz = kernel[:, :, 0]
            cmap = 'gray'
        elif in_ch == 3:
            # Normalize to [0, 1] for RGB display
            kernel_viz = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
            cmap = None
        else:
            # Multiple channels: show mean
            kernel_viz = kernel.mean(axis=2)
            cmap = 'gray'
        
        ax.imshow(kernel_viz, cmap=cmap, interpolation='nearest')
        ax.set_title(f'{i}', fontsize=7)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'First Layer Kernels: {first_conv.name} ({kh}x{kw}x{in_ch}, {out_ch} filters)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved first layer kernels to {output_path}")
    
    plt.close(fig)
