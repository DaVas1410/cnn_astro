"""
Custom Keras callbacks for kernel tracking and export during training.
"""

import os
from pathlib import Path
from typing import Optional, List, Union
import tensorflow as tf
from tensorflow import keras

from ..utils.kernels import export_model_kernels


class KernelExportCallback(keras.callbacks.Callback):
    """
    Callback to export convolutional kernel weights during training.
    
    This callback saves kernel weights at specified epochs or intervals,
    allowing you to study kernel convergence throughout training.
    
    Args:
        output_dir: Directory to save kernel exports
        frequency: Export frequency - 'epoch' for every epoch, or integer for every N epochs
        export_epochs: List of specific epochs to export at (overrides frequency if provided)
        format: Export format - 'hdf5', 'npz', or 'both'
        prefix: Filename prefix for exported files
        verbose: Whether to print export messages
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        frequency: Union[str, int] = 'epoch',
        export_epochs: Optional[List[int]] = None,
        format: str = 'hdf5',
        prefix: str = 'kernels',
        verbose: int = 1
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.frequency = frequency
        self.export_epochs = set(export_epochs) if export_epochs else None
        self.format = format
        self.prefix = prefix
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Called at the end of each epoch."""
        # Epoch is 0-indexed, convert to 1-indexed for user-friendliness
        epoch_num = epoch + 1
        
        should_export = False
        
        if self.export_epochs is not None:
            # Export at specific epochs
            should_export = epoch_num in self.export_epochs
        elif self.frequency == 'epoch':
            # Export every epoch
            should_export = True
        elif isinstance(self.frequency, int):
            # Export every N epochs
            should_export = (epoch_num % self.frequency == 0)
        
        if should_export:
            try:
                saved_path = export_model_kernels(
                    model=self.model,
                    output_dir=self.output_dir,
                    epoch=epoch_num,
                    format=self.format,
                    prefix=self.prefix
                )
                
                if self.verbose:
                    print(f"\nKernel weights exported to: {saved_path}")
                    
            except Exception as e:
                print(f"\nWarning: Failed to export kernels at epoch {epoch_num}: {e}")
    
    def on_train_end(self, logs: Optional[dict] = None):
        """Called at the end of training - export final kernels."""
        try:
            saved_path = export_model_kernels(
                model=self.model,
                output_dir=self.output_dir,
                epoch=None,  # Mark as final
                format=self.format,
                prefix=self.prefix
            )
            
            if self.verbose:
                print(f"\nFinal kernel weights exported to: {saved_path}")
                
        except Exception as e:
            print(f"\nWarning: Failed to export final kernels: {e}")


class KernelStatisticsCallback(keras.callbacks.Callback):
    """
    Callback to track kernel statistics during training without saving full weights.
    
    This is more memory-efficient than KernelExportCallback and only tracks
    summary statistics (mean, std, norms) for each convolutional layer.
    
    Args:
        log_file: Path to CSV file for logging statistics
        frequency: How often to log (epochs)
        verbose: Whether to print statistics
    """
    
    def __init__(
        self,
        log_file: Union[str, Path],
        frequency: int = 1,
        verbose: int = 0
    ):
        super().__init__()
        self.log_file = Path(log_file)
        self.frequency = frequency
        self.verbose = verbose
        
        # Create parent directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file with header
        self._write_header = True
        self._conv_layers = []
    
    def on_train_begin(self, logs: Optional[dict] = None):
        """Initialize by identifying all conv layers."""
        self._conv_layers = [
            layer for layer in self.model.layers
            if isinstance(layer, keras.layers.Conv2D)
        ]
        
        if self._write_header:
            with open(self.log_file, 'w') as f:
                # Write header
                headers = ['epoch']
                for layer in self._conv_layers:
                    layer_name = layer.name
                    headers.extend([
                        f'{layer_name}_mean',
                        f'{layer_name}_std',
                        f'{layer_name}_l2_norm'
                    ])
                f.write(','.join(headers) + '\n')
            
            self._write_header = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Log kernel statistics at specified frequency."""
        epoch_num = epoch + 1
        
        if epoch_num % self.frequency == 0:
            import numpy as np
            
            stats = []
            stats.append(str(epoch_num))
            
            for layer in self._conv_layers:
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]  # Get kernel weights
                    
                    mean = float(np.mean(kernel))
                    std = float(np.std(kernel))
                    l2_norm = float(np.sqrt(np.sum(kernel ** 2)))
                    
                    stats.extend([f'{mean:.6e}', f'{std:.6e}', f'{l2_norm:.6e}'])
                    
                    if self.verbose:
                        print(f"{layer.name}: mean={mean:.6e}, std={std:.6e}, l2={l2_norm:.6e}")
            
            # Write to file
            with open(self.log_file, 'a') as f:
                f.write(','.join(stats) + '\n')
