"""
GPU configuration utilities for Fractal Classifier v0.2.
Adapted from v0.1 — same interface, compatible with the new config schema.
"""

import os
import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras import mixed_precision

logger = logging.getLogger(__name__)


def setup_gpu(config=None, *, allocator=None, memory_growth=None, mixed_precision_policy=None) -> dict:
    """Configure GPU: allocator → memory growth → mixed precision."""
    if config is not None:
        gpu_cfg = config.gpu
        allocator = gpu_cfg.get('allocator', 'cuda_malloc_async')
        memory_growth = gpu_cfg.get('memory_growth', True)
        mp = gpu_cfg.get('mixed_precision', {})
        if isinstance(mp, dict):
            enabled = mp.get('enabled', True)
            mixed_precision_policy = mp.get('policy', 'mixed_float16') if enabled else 'float32'
        else:
            mixed_precision_policy = 'float32'
    else:
        allocator = allocator or 'cuda_malloc_async'
        memory_growth = memory_growth if memory_growth is not None else True
        mixed_precision_policy = mixed_precision_policy or 'mixed_float16'

    # 1. Allocator
    if 'TF_GPU_ALLOCATOR' not in os.environ:
        os.environ['TF_GPU_ALLOCATOR'] = allocator

    # 2. Memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, bool(memory_growth))
            logger.info(f"Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth: {e}")
    else:
        logger.warning("No GPUs found — training on CPU")

    # 3. Mixed precision
    try:
        mixed_precision.set_global_policy(mixed_precision_policy)
        logger.info(f"Mixed precision policy: {mixed_precision_policy}")
    except Exception as e:
        logger.warning(f"Could not set mixed precision: {e}")

    return {
        'num_gpus': len(gpus),
        'allocator': allocator,
        'memory_growth': memory_growth,
        'mixed_precision': mixed_precision_policy,
    }
