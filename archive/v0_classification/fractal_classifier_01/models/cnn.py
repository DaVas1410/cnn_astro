"""
CNN model architecture for Fractal Classifier.

This module provides the model building function that creates a
convolutional neural network for fractal cloud classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def build_fractal_cnn(input_shape: Tuple[int, int, int] = (512, 512, 1),
                     num_classes: int = 6,
                     filters: List[int] = [32, 64, 128, 256],
                     kernel_size: int = 3,
                     padding: str = 'same',
                     activation: str = 'relu',
                     dense_units: int = 64,
                     dropout_rate: float = 0.5,
                     pool_size: int = 2,
                     output_activation: str = 'softmax',
                     output_dtype: str = 'float32') -> models.Model:
    """
    Build a CNN for fractal cloud classification.
    
    Architecture consists of:
    - Multiple convolutional blocks (Conv2D -> Conv2D -> MaxPool2D)
    - Global average pooling
    - Dense head with dropout
    - Softmax output layer
    
    The number of blocks is determined by the length of the filters list.
    Each block has two conv layers with the same number of filters,
    followed by max pooling.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        filters: List of filter counts for each convolutional block
        kernel_size: Kernel size for all conv layers
        padding: Padding mode for conv layers ('same' or 'valid')
        activation: Activation function for conv and dense layers
        dense_units: Number of units in the dense head
        dropout_rate: Dropout rate after dense layer (0.0-1.0)
        pool_size: Pool size for MaxPooling layers
        output_activation: Activation for output layer ('softmax' or 'sigmoid')
        output_dtype: Data type for output layer (use 'float32' for mixed precision)
    
    Returns:
        Compiled Keras Model
    
    Example:
        >>> model = build_fractal_cnn(
        ...     input_shape=(512, 512, 1),
        ...     num_classes=6,
        ...     filters=[32, 64, 128, 256]
        ... )
        >>> model.summary()
    """
    logger.info(f"Building CNN with {len(filters)} blocks, "
               f"input_shape={input_shape}, num_classes={num_classes}")
    
    # Input layer
    inp = layers.Input(shape=input_shape, name='input')
    x = inp
    
    # Build convolutional blocks
    for i, num_filters in enumerate(filters):
        block_name = f'block_{i+1}'
        
        # First conv layer in block
        x = layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            name=f'{block_name}_conv1'
        )(x)
        
        # Second conv layer in block
        x = layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            name=f'{block_name}_conv2'
        )(x)
        
        # Max pooling to downsample
        x = layers.MaxPool2D(
            pool_size=pool_size,
            name=f'{block_name}_pool'
        )(x)
        
        logger.debug(f"Block {i+1}: {num_filters} filters, "
                    f"kernel_size={kernel_size}, pool_size={pool_size}")
    
    # Global average pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Dense head
    x = layers.Dense(
        units=dense_units,
        activation=activation,
        name='dense'
    )(x)
    
    # Dropout for regularization
    x = layers.Dropout(
        rate=dropout_rate,
        name='dropout'
    )(x)
    
    # Output layer
    # Use float32 dtype for numerical stability with mixed precision
    out = layers.Dense(
        units=num_classes,
        activation=output_activation,
        dtype=output_dtype,
        name='output'
    )(x)
    
    # Create model
    model = models.Model(inputs=inp, outputs=out, name='fractal_cnn')
    
    # Log model summary info
    total_params = model.count_params()
    logger.info(f"Model built successfully: {total_params:,} total parameters")
    
    return model


def build_model_from_config(config, num_classes: int) -> models.Model:
    """
    Build model using configuration object.
    
    Args:
        config: Configuration object with model settings
        num_classes: Number of output classes
    
    Returns:
        Keras Model
    """
    model_config = config.model
    input_shape = tuple(config.data.input_shape)
    
    return build_fractal_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=model_config.filters,
        kernel_size=model_config.kernel_size,
        padding=model_config.padding,
        activation=model_config.activation,
        dense_units=model_config.dense_units,
        dropout_rate=model_config.dropout_rate,
        pool_size=model_config.pool_size,
        output_activation=model_config.output_activation,
        output_dtype=model_config.output_dtype
    )


def load_model(model_path: str) -> models.Model:
    """
    Load a saved Keras model.
    
    Args:
        model_path: Path to saved model file (.h5)
    
    Returns:
        Loaded Keras Model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    logger.info(f"Loading model from {model_path}")
    try:
        model = models.load_model(model_path)
        logger.info(f"Model loaded successfully: {model.count_params():,} parameters")
        return model
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")


def save_model(model: models.Model, output_path: str) -> None:
    """
    Save a Keras model.
    
    Args:
        model: Keras Model to save
        output_path: Path where to save the model (.h5)
    """
    logger.info(f"Saving model to {output_path}")
    try:
        model.save(output_path)
        logger.info(f"Model saved successfully to {output_path}")
    except Exception as e:
        raise ValueError(f"Error saving model to {output_path}: {e}")


def print_model_summary(model: models.Model) -> None:
    """
    Print detailed model summary.
    
    Args:
        model: Keras Model
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    model.summary()
    print("="*80)
    
    # Additional statistics
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("="*80 + "\n")
