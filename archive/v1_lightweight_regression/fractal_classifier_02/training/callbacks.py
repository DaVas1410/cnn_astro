"""
Keras callbacks for Fractal Classifier v0.2 training.
"""

import logging
from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


def build_callbacks(config) -> List[keras.callbacks.Callback]:
    """Build the list of Keras callbacks from a Config object."""
    cb_cfg = config.callbacks
    paths_cfg = config.paths
    callbacks: List[keras.callbacks.Callback] = []

    # ModelCheckpoint
    ckpt_cfg = cb_cfg.checkpoint
    if getattr(ckpt_cfg, 'enabled', True):
        ckpt_dir = Path(paths_cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        filepath = str(ckpt_dir / getattr(ckpt_cfg, 'filename', 'best.keras'))
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=getattr(ckpt_cfg, 'monitor', 'val_mae'),
            mode=getattr(ckpt_cfg, 'mode', 'min'),
            save_best_only=getattr(ckpt_cfg, 'save_best_only', True),
            verbose=1,
        ))
        logger.info(f"ModelCheckpoint → {filepath}")

    # EarlyStopping
    es_cfg = cb_cfg.early_stopping
    if getattr(es_cfg, 'enabled', True):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=getattr(es_cfg, 'monitor', 'val_mae'),
            patience=int(getattr(es_cfg, 'patience', 10)),
            min_delta=float(getattr(es_cfg, 'min_delta', 0.01)),
            restore_best_weights=getattr(es_cfg, 'restore_best_weights', True),
            verbose=1,
        ))
        logger.info("EarlyStopping enabled")

    # ReduceLROnPlateau
    lr_cfg = cb_cfg.reduce_lr
    if getattr(lr_cfg, 'enabled', True):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=getattr(lr_cfg, 'monitor', 'val_mae'),
            factor=float(getattr(lr_cfg, 'factor', 0.5)),
            patience=int(getattr(lr_cfg, 'patience', 5)),
            min_lr=float(getattr(lr_cfg, 'min_lr', 1e-7)),
            verbose=1,
        ))
        logger.info("ReduceLROnPlateau enabled")

    # CSVLogger
    csv_cfg = cb_cfg.csv_logger
    if getattr(csv_cfg, 'enabled', True):
        history_dir = Path(paths_cfg.history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(history_dir / 'training_log.csv')
        callbacks.append(keras.callbacks.CSVLogger(csv_path, append=False))
        logger.info(f"CSVLogger → {csv_path}")

    # TensorBoard (optional)
    tb_cfg = cb_cfg.tensorboard
    if getattr(tb_cfg, 'enabled', False):
        tb_dir = str(Path(paths_cfg.output_dir) / 'tensorboard')
        callbacks.append(keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1))
        logger.info(f"TensorBoard → {tb_dir}")

    return callbacks
