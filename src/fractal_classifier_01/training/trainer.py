"""
Training orchestration for Fractal Classifier.

This module provides the main Trainer class that coordinates model training,
including data loading, model building, callback setup, and training execution.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, callbacks

from ..config import Config
from ..data import DataIndexer, DataPipeline
from ..models import build_model_from_config, save_model, print_model_summary
from ..utils import setup_gpu

logger = logging.getLogger(__name__)


class FractalClassifierTrainer:
    """
    Main trainer class for fractal classification model.
    
    Orchestrates the complete training workflow including:
    - GPU setup and configuration
    - Data loading and pipeline creation
    - Model building and compilation
    - Callback setup
    - Training execution
    - Model saving and history export
    """
    
    def __init__(self, 
                 config: Config,
                 verbose: bool = True):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
            verbose: Whether to print detailed information
        """
        self.config = config
        self.verbose = verbose
        self.model = None
        self.history = None
        self.data_indexer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        logger.info("Initializing FractalClassifierTrainer")
    
    def setup(self) -> None:
        """
        Set up GPU, data, and model.
        
        This method should be called before training to initialize
        all necessary components.
        """
        logger.info("="*80)
        logger.info("FRACTAL CLASSIFIER TRAINER - SETUP")
        logger.info("="*80)
        
        # Step 1: Setup GPU
        if self.verbose:
            print("\n[1/3] Setting up GPU configuration...")
        self.gpu_info = setup_gpu(self.config)
        
        # Step 2: Setup data
        if self.verbose:
            print("\n[2/3] Loading and preparing data...")
        self._setup_data()
        
        # Step 3: Build model
        if self.verbose:
            print("\n[3/3] Building model...")
        self._build_model()
        
        logger.info("Setup complete!")
        logger.info("="*80)
    
    def _setup_data(self) -> None:
        """Set up data loading and pipeline."""
        # Initialize data indexer
        h5_path = self.config.data.h5_path
        self.data_indexer = DataIndexer(h5_path)
        
        # Build train/val/test splits
        split_config = self.config.data.split
        train_idx, val_idx, test_idx = self.data_indexer.build_train_val_test_splits(
            train_ratio=split_config.train_ratio,
            val_ratio=split_config.val_ratio,
            test_ratio=split_config.test_ratio,
            random_state=split_config.random_state,
            stratify=split_config.stratify
        )
        
        if self.verbose:
            print(f"  Dataset: {h5_path}")
            print(f"  Total samples: {self.data_indexer.num_samples}")
            print(f"  Train: {len(train_idx)} samples")
            print(f"  Validation: {len(val_idx)} samples")
            print(f"  Test: {len(test_idx)} samples")
            print(f"  Classes: {self.data_indexer.num_classes} ({self.data_indexer.label_names})")
        
        # Create data pipeline
        pipeline = DataPipeline(
            h5_path=h5_path,
            label2class=self.data_indexer.label2class,
            input_shape=tuple(self.config.data.input_shape)
        )
        
        # Build datasets
        train_subset = self.data_indexer.get_subset(train_idx)
        val_subset = self.data_indexer.get_subset(val_idx)
        test_subset = self.data_indexer.get_subset(test_idx)
        
        # Training dataset (with repeat for fitting)
        self.train_dataset = pipeline.build_dataset_from_config(
            train_subset, self.config, training=True
        )
        
        # Validation dataset (no repeat, no augmentation)
        self.val_dataset = pipeline.build_dataset_from_config(
            val_subset, self.config, training=False
        )
        
        # Test dataset (no repeat, no augmentation)
        self.test_dataset = pipeline.build_dataset_from_config(
            test_subset, self.config, training=False
        )
        
        # Calculate steps per epoch
        batch_size = self.config.training.batch_size
        self.steps_per_epoch = max(1, len(train_idx) // batch_size)
        self.validation_steps = max(1, len(val_idx) // batch_size)
        
        if self.verbose:
            print(f"  Batch size: {batch_size}")
            print(f"  Steps per epoch: {self.steps_per_epoch}")
            print(f"  Validation steps: {self.validation_steps}")
    
    def _build_model(self) -> None:
        """Build and compile the model."""
        # Build model architecture
        self.model = build_model_from_config(
            self.config,
            num_classes=self.data_indexer.num_classes
        )
        
        if self.verbose:
            print(f"  Model built with {self.model.count_params():,} parameters")
        
        # Compile model
        self._compile_model()
        
        if self.verbose:
            print("  Model compiled successfully")
    
    def _compile_model(self) -> None:
        """Compile the model with optimizer, loss, and metrics."""
        training_config = self.config.training
        
        # Setup optimizer
        optimizer_name = training_config.optimizer.name.lower()
        lr = training_config.learning_rate
        
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = optimizers.SGD(learning_rate=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=lr)
        else:
            logger.warning(f"Unknown optimizer '{optimizer_name}', defaulting to Adam")
            optimizer = optimizers.Adam(learning_rate=lr)
        
        # Setup loss
        loss_name = training_config.loss
        if loss_name == 'sparse_categorical_crossentropy':
            loss_fn = losses.SparseCategoricalCrossentropy()
        elif loss_name == 'categorical_crossentropy':
            loss_fn = losses.CategoricalCrossentropy()
        else:
            logger.warning(f"Unknown loss '{loss_name}', defaulting to sparse categorical crossentropy")
            loss_fn = losses.SparseCategoricalCrossentropy()
        
        # Setup metrics
        metric_list = []
        for metric_name in training_config.metrics:
            if metric_name == 'sparse_categorical_accuracy':
                metric_list.append(metrics.SparseCategoricalAccuracy())
            elif metric_name == 'accuracy':
                metric_list.append(metrics.Accuracy())
            else:
                logger.warning(f"Unknown metric '{metric_name}', skipping")
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metric_list
        )
        
        logger.info(f"Model compiled with optimizer={optimizer_name}, lr={lr}, loss={loss_name}")
    
    def _setup_callbacks(self) -> list:
        """Set up training callbacks."""
        callback_list = []
        callbacks_config = self.config.training.callbacks
        paths_config = self.config.paths
        
        # Ensure output directories exist
        checkpoint_dir = Path(paths_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = Path(paths_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ModelCheckpoint
        if callbacks_config.model_checkpoint.enabled:
            ckpt_path = checkpoint_dir / paths_config.best_model
            callback_list.append(callbacks.ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor=callbacks_config.model_checkpoint.monitor,
                save_best_only=callbacks_config.model_checkpoint.save_best_only,
                verbose=callbacks_config.model_checkpoint.verbose
            ))
            logger.info(f"ModelCheckpoint: saving best model to {ckpt_path}")
        
        # EarlyStopping
        if callbacks_config.early_stopping.enabled:
            callback_list.append(callbacks.EarlyStopping(
                monitor=callbacks_config.early_stopping.monitor,
                patience=callbacks_config.early_stopping.patience,
                restore_best_weights=callbacks_config.early_stopping.restore_best_weights,
                verbose=callbacks_config.early_stopping.verbose
            ))
            logger.info(f"EarlyStopping: patience={callbacks_config.early_stopping.patience}")
        
        # ReduceLROnPlateau
        if callbacks_config.reduce_lr.enabled:
            callback_list.append(callbacks.ReduceLROnPlateau(
                monitor=callbacks_config.reduce_lr.monitor,
                factor=callbacks_config.reduce_lr.factor,
                patience=callbacks_config.reduce_lr.patience,
                min_lr=callbacks_config.reduce_lr.min_lr,
                verbose=callbacks_config.reduce_lr.verbose
            ))
            logger.info(f"ReduceLROnPlateau: factor={callbacks_config.reduce_lr.factor}, "
                       f"patience={callbacks_config.reduce_lr.patience}")
        
        # CSVLogger
        if callbacks_config.csv_logger.enabled:
            history_path = output_dir / paths_config.history_csv
            callback_list.append(callbacks.CSVLogger(
                filename=str(history_path),
                append=callbacks_config.csv_logger.append
            ))
            logger.info(f"CSVLogger: logging to {history_path}")
        
        # TensorBoard
        if callbacks_config.tensorboard.enabled:
            tensorboard_dir = Path(paths_config.tensorboard_log_dir)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            callback_list.append(callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                write_graph=callbacks_config.tensorboard.write_graph,
                write_images=callbacks_config.tensorboard.write_images,
                update_freq=callbacks_config.tensorboard.update_freq,
                profile_batch=callbacks_config.tensorboard.profile_batch,
                histogram_freq=callbacks_config.tensorboard.histogram_freq
            ))
            logger.info(f"TensorBoard: logging to {tensorboard_dir}")
        
        # Kernel Export Callback
        if callbacks_config.kernel_export.enabled:
            from .callbacks import KernelExportCallback
            
            kernel_weights_dir = Path(paths_config.kernel_weights_dir)
            kernel_weights_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine frequency
            frequency = callbacks_config.kernel_export.get('frequency', 'epoch')
            export_epochs = callbacks_config.kernel_export.get('export_epochs', None)
            export_format = callbacks_config.kernel_export.get('format', 'hdf5')
            
            callback_list.append(KernelExportCallback(
                output_dir=kernel_weights_dir,
                frequency=frequency,
                export_epochs=export_epochs,
                format=export_format,
                verbose=1
            ))
            logger.info(f"KernelExport: saving kernels to {kernel_weights_dir} (format: {export_format})")
        
        return callback_list
    
    def train(self) -> tf.keras.callbacks.History:
        """
        Execute training.
        
        Returns:
            Training history object
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call setup() first.")
        
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        
        # Setup callbacks
        callback_list = self._setup_callbacks()
        
        # Training configuration
        epochs = self.config.training.epochs
        
        if self.verbose:
            print(f"\nTraining for {epochs} epochs")
            print(f"Batch size: {self.config.training.batch_size}")
            print(f"Learning rate: {self.config.training.learning_rate}")
            print(f"Steps per epoch: {self.steps_per_epoch}")
            print(f"Validation steps: {self.validation_steps}")
            print(f"Callbacks: {len(callback_list)} enabled")
            print("\n" + "="*80 + "\n")
        
        # Train the model
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_dataset,
            validation_steps=self.validation_steps,
            callbacks=callback_list,
            verbose=1 if self.verbose else 0
        )
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return self.history
    
    def save_history(self, output_path: Optional[str] = None) -> None:
        """
        Save training history to CSV.
        
        Args:
            output_path: Path to save CSV. If None, uses config path.
        """
        if self.history is None:
            logger.warning("No training history to save")
            return
        
        if output_path is None:
            output_dir = Path(self.config.paths.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / self.config.paths.history_csv
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.history.history)
        history_df.index.name = 'epoch'
        
        # Save to CSV
        history_df.to_csv(output_path)
        logger.info(f"Training history saved to {output_path}")
        
        if self.verbose:
            print(f"\nTraining history saved to {output_path}")
    
    def evaluate(self, dataset: str = 'test') -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Which dataset to evaluate ('train', 'val', or 'test')
        
        Returns:
            Dictionary of metric names and values
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call setup() first.")
        
        # Select dataset
        if dataset == 'train':
            ds = self.train_dataset
        elif dataset == 'val':
            ds = self.val_dataset
        elif dataset == 'test':
            ds = self.test_dataset
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        
        logger.info(f"Evaluating on {dataset} set...")
        results = self.model.evaluate(ds, verbose=1 if self.verbose else 0)
        
        # Convert to dictionary
        metric_names = ['loss'] + [m.name for m in self.model.metrics]
        results_dict = dict(zip(metric_names, results))
        
        logger.info(f"Evaluation results: {results_dict}")
        
        return results_dict
    
    def print_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            logger.warning("Model not built yet")
            return
        
        print_model_summary(self.model)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.model.load_weights(checkpoint_path)
        logger.info("Checkpoint loaded successfully")
