"""
Training orchestrator for Fractal Classifier v0.2.
"""

import logging
import json
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from ..config import Config
from ..data import DataIndexer, DataPipeline
from ..models import build_model_from_config
from ..utils import setup_gpu
from .callbacks import build_callbacks

logger = logging.getLogger(__name__)


class FractalRegressorTrainer:
    """
    End-to-end training orchestrator.

    Usage::

        config = load_config('config/default_config.yaml')
        trainer = FractalRegressorTrainer(config)
        history = trainer.train()
    """

    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> keras.callbacks.History:
        """Run the full training pipeline and return the History object."""
        self._setup_gpu()
        self._prepare_output_dirs()
        train_ds, val_ds = self._build_datasets()
        self.model = self._build_model()
        self._compile_model()
        callbacks = build_callbacks(self.config)

        logger.info("Starting training")
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.training.epochs,
            steps_per_epoch=self._steps_per_epoch,
            validation_steps=self._val_steps,
            callbacks=callbacks,
            verbose=1,
        )
        self._save_history()
        logger.info("Training complete")
        return self.history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_gpu(self):
        info = setup_gpu(self.config)
        logger.info(f"GPU setup: {info}")

    def _prepare_output_dirs(self):
        paths_cfg = self.config.paths
        for attr in ('output_dir', 'checkpoint_dir', 'history_dir'):
            p = Path(getattr(paths_cfg, attr, 'outputs'))
            p.mkdir(parents=True, exist_ok=True)

    def _build_datasets(self):
        data_cfg = self.config.data
        split_cfg = data_cfg.split

        self._indexer = DataIndexer(
            dataset_dir=data_cfg.dataset_dir,
            resolutions=[int(r) for r in data_cfg.resolutions],
            split_ratios=(
                float(split_cfg.train_ratio),
                float(split_cfg.val_ratio),
                float(split_cfg.test_ratio),
            ),
            random_seed=int(getattr(split_cfg, 'random_seed', 42)),
        )
        self._indexer.build()
        logger.info("\n" + self._indexer.summary())

        batch_size = int(self.config.training.batch_size)
        native_res = int(getattr(data_cfg, 'native_resolution', 128))
        self._steps_per_epoch = self._indexer.steps_per_epoch('train', batch_size, native_res)
        self._val_steps = self._indexer.steps_per_epoch('val', batch_size, native_res)
        logger.info(f"Steps per epoch — train: {self._steps_per_epoch}, val: {self._val_steps}")

        train_pipeline = DataPipeline.from_config(self._indexer, self.config, training=True)
        val_pipeline   = DataPipeline.from_config(self._indexer, self.config, training=False)

        return (
            train_pipeline.build_train_dataset().repeat(),
            val_pipeline.build_val_dataset(),
        )

    def _build_model(self) -> keras.Model:
        model = build_model_from_config(self.config)
        model.summary(print_fn=logger.info)
        return model

    def _compile_model(self):
        train_cfg = self.config.training
        optimizer = self._build_optimizer(train_cfg)
        loss = self._build_loss(train_cfg)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse'),
            ],
        )
        logger.info(f"Model compiled — loss={train_cfg.loss}, optimizer={train_cfg.optimizer}")

    @staticmethod
    def _build_optimizer(train_cfg) -> keras.optimizers.Optimizer:
        lr = float(train_cfg.learning_rate)
        opt_name = str(getattr(train_cfg, 'optimizer', 'adam')).lower()
        opt_params = train_cfg.optimizer_params.to_dict() if hasattr(train_cfg, 'optimizer_params') else {}

        if opt_name == 'adam':
            return keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=float(opt_params.get('beta_1', 0.9)),
                beta_2=float(opt_params.get('beta_2', 0.999)),
                epsilon=float(opt_params.get('epsilon', 1e-7)),
            )
        elif opt_name == 'sgd':
            return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        elif opt_name == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    @staticmethod
    def _build_loss(train_cfg) -> keras.losses.Loss:
        loss_name = str(getattr(train_cfg, 'loss', 'huber')).lower()
        if loss_name == 'huber':
            delta = float(getattr(train_cfg, 'huber_delta', 1.0))
            return keras.losses.Huber(delta=delta)
        elif loss_name == 'mse':
            return keras.losses.MeanSquaredError()
        elif loss_name == 'mae':
            return keras.losses.MeanAbsoluteError()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    def _save_history(self):
        if self.history is None:
            return
        history_dir = Path(self.config.paths.history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)
        out_path = history_dir / 'training_history.json'
        with open(out_path, 'w') as f:
            json.dump({k: [float(v) for v in vals]
                       for k, vals in self.history.history.items()}, f, indent=2)
        logger.info(f"Training history saved to {out_path}")
