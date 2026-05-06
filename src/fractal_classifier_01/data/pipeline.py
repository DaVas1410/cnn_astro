"""
TensorFlow data pipeline for Fractal Classifier.

This module provides utilities to build tf.data pipelines with lazy loading,
augmentation, batching, and prefetching for efficient training.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Callable
import logging

from .loader import read_image

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Builds and manages TensorFlow data pipelines for fractal classification.
    
    Wraps dataset construction with configurable augmentation, batching,
    and prefetching for efficient GPU utilization.
    """
    
    def __init__(self, 
                 h5_path: str,
                 label2class: dict,
                 input_shape: Tuple[int, int, int] = (512, 512, 1)):
        """
        Initialize DataPipeline.
        
        Args:
            h5_path: Path to HDF5 dataset file
            label2class: Dictionary mapping k-value to class id
            input_shape: Expected input shape (height, width, channels)
        """
        self.h5_path = h5_path
        self.label2class = label2class
        self.input_shape = input_shape
    
    def _generator_from_index(self, 
                             index_subset: List[Tuple[str, int, int]],
                             preprocess_fn: Optional[Callable] = None):
        """
        Create a Python generator that yields images and labels.
        
        Args:
            index_subset: List of (group, idx, k_value) tuples
            preprocess_fn: Optional preprocessing function to apply to images
        
        Yields:
            Tuple of (image, label) where image is (H, W, 1) float32
        """
        def gen():
            for group, idx, k_val in index_subset:
                # Read image from HDF5
                img = read_image(self.h5_path, group, idx)
                
                # Apply preprocessing if provided
                if preprocess_fn is not None:
                    img = preprocess_fn(img)
                
                # Ensure channel dimension exists
                if img.ndim == 2:
                    img = img[..., np.newaxis]
                
                # Get class label
                label = self.label2class[k_val]
                
                yield img, label
        
        return gen
    
    def build_dataset(self,
                     index_subset: List[Tuple[str, int, int]],
                     batch_size: int = 32,
                     shuffle: bool = True,
                     training: bool = True,
                     repeat: bool = False,
                     augment: bool = True,
                     shuffle_buffer: int = 512,
                     num_parallel_calls: int = 2,
                     prefetch: int = 1,
                     shuffle_seed: int = 42,
                     preprocess_fn: Optional[Callable] = None) -> tf.data.Dataset:
        """
        Build a tf.data.Dataset from an index subset.
        
        Args:
            index_subset: List of (group, idx, k_value) tuples
            batch_size: Batch size for training
            shuffle: Whether to shuffle the dataset
            training: Whether this is a training dataset (affects augmentation)
            repeat: Whether to repeat the dataset indefinitely
            augment: Whether to apply data augmentation (only if training=True)
            shuffle_buffer: Buffer size for shuffling
            num_parallel_calls: Number of parallel map operations
            prefetch: Number of batches to prefetch
            shuffle_seed: Random seed for shuffling
            preprocess_fn: Optional preprocessing function
        
        Returns:
            tf.data.Dataset ready for training/evaluation
        """
        # Define output signature for the generator
        output_signature = (
            tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        
        # Create dataset from generator
        ds = tf.data.Dataset.from_generator(
            self._generator_from_index(index_subset, preprocess_fn),
            output_signature=output_signature
        )
        
        # Shuffle if requested (typically for training)
        if shuffle and training:
            buffer_size = min(shuffle_buffer, len(index_subset))
            ds = ds.shuffle(buffer_size=buffer_size, seed=shuffle_seed)
            logger.debug(f"Shuffling with buffer size {buffer_size}")
        
        # Cast to float32 (redundant but explicit)
        ds = ds.map(
            lambda im, lab: (tf.cast(im, tf.float32), lab),
            num_parallel_calls=num_parallel_calls
        )
        
        # Apply augmentation only during training
        if training and augment:
            ds = self._apply_augmentation(ds, num_parallel_calls)
        
        # Batch the data
        ds = ds.batch(batch_size)
        
        # Repeat if requested
        if repeat:
            ds = ds.repeat()
        
        # Prefetch for performance
        ds = ds.prefetch(prefetch)
        
        logger.info(f"Built dataset: {len(index_subset)} samples, "
                   f"batch_size={batch_size}, shuffle={shuffle}, "
                   f"training={training}, augment={augment}")
        
        return ds
    
    def _apply_augmentation(self, 
                           ds: tf.data.Dataset,
                           num_parallel_calls: int = 2) -> tf.data.Dataset:
        """
        Apply data augmentation transformations.
        
        Current augmentations:
        - Random horizontal flip
        - Random vertical flip
        
        Args:
            ds: Input dataset
            num_parallel_calls: Number of parallel map operations
        
        Returns:
            Dataset with augmentation applied
        """
        # Random horizontal flip
        ds = ds.map(
            lambda im, lab: (tf.image.random_flip_left_right(im), lab),
            num_parallel_calls=num_parallel_calls
        )
        
        # Random vertical flip
        ds = ds.map(
            lambda im, lab: (tf.image.random_flip_up_down(im), lab),
            num_parallel_calls=num_parallel_calls
        )
        
        logger.debug("Applied augmentation: horizontal flip, vertical flip")
        
        return ds
    
    def build_dataset_from_config(self,
                                 index_subset: List[Tuple[str, int, int]],
                                 config,
                                 training: bool = True) -> tf.data.Dataset:
        """
        Build dataset using configuration object.
        
        Args:
            index_subset: List of (group, idx, k_value) tuples
            config: Configuration object with training and pipeline settings
            training: Whether this is a training dataset
        
        Returns:
            tf.data.Dataset ready for training/evaluation
        """
        # Extract parameters from config
        batch_size = config.training.batch_size
        pipeline_config = config.training.pipeline
        augment_config = config.training.augmentation
        
        # Determine if augmentation should be applied
        augment = (training and 
                  augment_config.horizontal_flip and 
                  augment_config.vertical_flip)
        
        return self.build_dataset(
            index_subset=index_subset,
            batch_size=batch_size,
            shuffle=training,
            training=training,
            repeat=training,  # Repeat only for training
            augment=augment,
            shuffle_buffer=pipeline_config.shuffle_buffer,
            num_parallel_calls=pipeline_config.num_parallel_calls,
            prefetch=pipeline_config.prefetch,
            shuffle_seed=pipeline_config.shuffle_seed
        )


def build_tf_dataset(h5_path: str,
                    index_subset: List[Tuple[str, int, int]],
                    label2class: dict,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    training: bool = True,
                    input_shape: Tuple[int, int, int] = (512, 512, 1),
                    **kwargs) -> tf.data.Dataset:
    """
    Convenience function to build a tf.data.Dataset.
    
    Args:
        h5_path: Path to HDF5 dataset file
        index_subset: List of (group, idx, k_value) tuples
        label2class: Dictionary mapping k-value to class id
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        training: Whether this is a training dataset
        input_shape: Expected input shape (height, width, channels)
        **kwargs: Additional arguments passed to DataPipeline.build_dataset
    
    Returns:
        tf.data.Dataset ready for training/evaluation
    """
    pipeline = DataPipeline(h5_path, label2class, input_shape)
    return pipeline.build_dataset(
        index_subset=index_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        training=training,
        **kwargs
    )
