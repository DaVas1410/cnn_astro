"""
Lightweight CNN regression model for Fractal Classifier v0.2.

Architecture uses MobileNet-style depthwise-separable convolution blocks
(~8–9× fewer multiply-adds than equivalent standard Conv2D blocks).
The output is a single linear neuron for direct k_min regression.
"""

import logging
from typing import List, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

logger = logging.getLogger(__name__)


def _ds_conv_block(
    x: tf.Tensor,
    filters: int,
    pool: bool = True,
    l2: float = 0.0,
    batch_norm: bool = True,
    name: str = '',
) -> tf.Tensor:
    """
    Depthwise-separable convolution block:
        DepthwiseConv2D → (BN) → ReLU → PointwiseConv (1×1) → (BN) → ReLU → (MaxPool)
    """
    prefix = f"{name}_" if name else ""

    # Depthwise
    x = layers.DepthwiseConv2D(
        kernel_size=3, padding='same', use_bias=not batch_norm,
        depthwise_regularizer=regularizers.l2(l2) if l2 > 0 else None,
        name=f"{prefix}dw",
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name=f"{prefix}dw_bn")(x)
    x = layers.Activation('relu', name=f"{prefix}dw_relu")(x)

    # Pointwise (1×1)
    x = layers.Conv2D(
        filters, kernel_size=1, padding='same', use_bias=not batch_norm,
        kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None,
        name=f"{prefix}pw",
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name=f"{prefix}pw_bn")(x)
    x = layers.Activation('relu', name=f"{prefix}pw_relu")(x)

    if pool:
        x = layers.MaxPooling2D(pool_size=2, name=f"{prefix}pool")(x)

    return x


def build_regression_cnn(
    input_shape: tuple = (128, 128, 1),
    filters: List[int] = (32, 64, 128),
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    l2_weight: float = 0.0,
    batch_norm: bool = True,
) -> keras.Model:
    """
    Build the lightweight depthwise-separable CNN regression model.

    Args:
        input_shape:  (H, W, C) — must match native_resolution in config.
        filters:      Filter counts for each DS-Conv block.
        dense_units:  Units in the dense head before the output neuron.
        dropout_rate: Dropout fraction applied before the output layer.
        l2_weight:    L2 regularisation coefficient (0 = disabled).
        batch_norm:   Whether to include Batch Normalisation layers.

    Returns:
        Compiled keras.Model.
    """
    inputs = keras.Input(shape=input_shape, name='input')

    # Initial standard convolution — cheap first-layer feature extractor
    x = layers.Conv2D(
        16, kernel_size=3, padding='same', use_bias=not batch_norm,
        kernel_regularizer=regularizers.l2(l2_weight) if l2_weight > 0 else None,
        name='stem_conv',
    )(inputs)
    if batch_norm:
        x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_relu')(x)

    # Depthwise-separable blocks
    for i, n_filters in enumerate(filters):
        x = _ds_conv_block(
            x, filters=n_filters, pool=True,
            l2=l2_weight, batch_norm=batch_norm,
            name=f"block{i+1}",
        )

    # Global pooling → dense head → regression output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(
        dense_units,
        kernel_regularizer=regularizers.l2(l2_weight) if l2_weight > 0 else None,
        name='dense',
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name='dense_bn')(x)
    x = layers.Activation('relu', name='dense_relu')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)

    # Output: float32 (important for mixed-precision stability)
    outputs = layers.Dense(1, dtype='float32', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='fractal_regressor_v02')

    n_params = model.count_params()
    logger.info(f"Model built: {model.name} — {n_params:,} parameters")
    return model


def build_model_from_config(config) -> keras.Model:
    """Build the regression CNN from a Config object."""
    model_cfg = config.model
    return build_regression_cnn(
        input_shape=tuple(model_cfg.input_shape),
        filters=list(model_cfg.filters),
        dense_units=int(model_cfg.dense_units),
        dropout_rate=float(model_cfg.dropout_rate),
        l2_weight=float(getattr(model_cfg, 'l2_weight', 0.0)),
        batch_norm=bool(getattr(model_cfg, 'batch_norm', True)),
    )
