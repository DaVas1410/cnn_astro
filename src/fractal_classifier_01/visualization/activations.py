"""Layer activation visualization for Fractal Classifier."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def visualize_activations(model,
                         images: np.ndarray,
                         layer_names: Optional[List[str]] = None,
                         output_dir: Optional[str] = None,
                         num_filters: int = 8,
                         dpi: int = 150) -> None:
    """
    Visualize activations from intermediate layers.
    
    Args:
        model: Trained Keras model
        images: Input images, shape (N, H, W, C)
        layer_names: List of layer names to visualize (default: all conv layers)
        output_dir: Directory to save activation images
        num_filters: Number of filters/channels to show per layer
        dpi: DPI for saved images
    """
    if layer_names is None:
        # Get all convolutional layer names
        layer_names = [layer.name for layer in model.layers 
                      if isinstance(layer, tf.keras.layers.Conv2D)]
    
    if not layer_names:
        logger.warning("No layers specified for visualization")
        return
    
    # Create model that outputs activations for specified layers
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(images, verbose=0)
    
    # Ensure activations is a list
    if not isinstance(activations, list):
        activations = [activations]
    
    # Save activations for each image
    for img_idx in range(len(images)):
        img_output_dir = Path(output_dir) / f'img_{img_idx}' if output_dir else None
        if img_output_dir:
            img_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize each layer
        for layer_idx, (layer_name, activation) in enumerate(zip(layer_names, activations)):
            layer_activation = activation[img_idx]
            
            # Determine number of filters to show
            n_filters = min(num_filters, layer_activation.shape[-1])
            
            # Create grid
            n_cols = 4
            n_rows = (n_filters + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for i in range(n_filters):
                ax = axes[i]
                ax.imshow(layer_activation[:, :, i], cmap='viridis')
                ax.set_title(f'Filter {i}', fontsize=9)
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(n_filters, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Layer: {layer_name} - Image {img_idx}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            if img_output_dir:
                save_path = img_output_dir / f'{layer_idx}_{layer_name}.png'
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                logger.debug(f"Saved activation to {save_path}")
            
            plt.close(fig)
    
    if output_dir:
        logger.info(f"Saved activations for {len(images)} images to {output_dir}")


def save_intermediate_outputs(model,
                             images: np.ndarray,
                             output_dir: str,
                             layer_pattern: str = 'conv',
                             dpi: int = 150) -> None:
    """
    Save intermediate layer outputs for analysis.
    
    Args:
        model: Trained Keras model
        images: Input images
        output_dir: Directory to save outputs
        layer_pattern: Pattern to match layer names (e.g., 'conv', 'pool')
        dpi: DPI for saved images
    """
    output_dir = Path(output_dir)
    
    # Find layers matching pattern
    layer_names = [layer.name for layer in model.layers 
                  if layer_pattern.lower() in layer.name.lower()]
    
    if not layer_names:
        logger.warning(f"No layers found matching pattern '{layer_pattern}'")
        return
    
    logger.info(f"Visualizing {len(layer_names)} layers matching '{layer_pattern}'")
    visualize_activations(model, images, layer_names, str(output_dir), dpi=dpi)
