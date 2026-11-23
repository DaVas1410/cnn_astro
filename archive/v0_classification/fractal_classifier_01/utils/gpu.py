"""
GPU configuration utilities for Fractal Classifier.

This module provides functions to configure TensorFlow GPU settings,
enable memory growth, and set up mixed precision training.
"""

import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def set_gpu_allocator(allocator: str = 'cuda_malloc_async') -> None:
    """
    Set TensorFlow GPU memory allocator.
    
    The cuda_malloc_async allocator reduces memory fragmentation on newer
    CUDA versions (CUDA 11.2+).
    
    Args:
        allocator: GPU allocator type ('cuda_malloc_async' recommended)
    
    Note:
        This must be called before any TensorFlow operations.
        Environment variables can only be set once before TF initialization.
    """
    if 'TF_GPU_ALLOCATOR' not in os.environ:
        os.environ['TF_GPU_ALLOCATOR'] = allocator
        logger.info(f"Set TF_GPU_ALLOCATOR to '{allocator}'")
    else:
        logger.warning(f"TF_GPU_ALLOCATOR already set to '{os.environ['TF_GPU_ALLOCATOR']}'")


def set_memory_growth(enable: bool = True) -> None:
    """
    Enable or disable GPU memory growth.
    
    When enabled, TensorFlow will allocate GPU memory as needed rather than
    pre-allocating all available memory at startup. This is useful when
    sharing GPUs or running multiple experiments.
    
    Args:
        enable: Whether to enable memory growth
    
    Note:
        This must be called before any TensorFlow operations that use the GPU.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        logger.warning("No GPUs found. Memory growth setting will have no effect.")
        return
    
    try:
        for gpu in gpus:
            if enable:
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_memory_growth(gpu, False)
        
        status = "enabled" if enable else "disabled"
        logger.info(f"Memory growth {status} for {len(gpus)} GPU(s): {gpus}")
    
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(f"Failed to set memory growth: {e}")
        logger.error("Memory growth must be set before any GPU operations.")
        raise


def set_mixed_precision_policy(policy: str = 'mixed_float16') -> None:
    """
    Set mixed precision training policy.
    
    Mixed precision uses float16 for computation (faster on modern GPUs)
    and float32 for numerical stability where needed. This reduces memory
    usage and can speed up training significantly.
    
    Args:
        policy: Mixed precision policy name
                'mixed_float16' - float16 compute, float32 variables (recommended for GPUs)
                'mixed_bfloat16' - bfloat16 compute (for TPUs)
                'float32' - no mixed precision (default)
    
    Note:
        When using mixed precision, ensure the output layer has dtype='float32'
        for numerical stability with softmax/sigmoid activations.
    """
    try:
        mixed_precision.set_global_policy(policy)
        current_policy = mixed_precision.global_policy()
        logger.info(f"Mixed precision policy set to: {current_policy}")
        
        if policy == 'mixed_float16':
            logger.info("Using float16 for computation, float32 for variables")
            logger.info("Ensure output layers use dtype='float32' for stability")
    
    except Exception as e:
        logger.error(f"Failed to set mixed precision policy '{policy}': {e}")
        raise


def setup_gpu(config=None,
             allocator: Optional[str] = None,
             memory_growth: Optional[bool] = None,
             mixed_precision_policy: Optional[str] = None) -> dict:
    """
    Configure GPU settings for training.
    
    This is a convenience function that sets up all GPU-related configurations
    in the correct order. It should be called before building the model.
    
    Args:
        config: Configuration object (if provided, other args are ignored)
        allocator: GPU allocator type (default: 'cuda_malloc_async')
        memory_growth: Enable memory growth (default: True)
        mixed_precision_policy: Mixed precision policy (default: 'mixed_float16')
    
    Returns:
        Dictionary with GPU setup information
    
    Example:
        >>> from fractal_classifier.utils import setup_gpu
        >>> from fractal_classifier.config import load_config
        >>> 
        >>> config = load_config('my_config.yaml')
        >>> gpu_info = setup_gpu(config)
        >>> print(gpu_info)
    """
    # Extract parameters from config if provided
    if config is not None:
        gpu_config = config.gpu
        allocator = gpu_config.get('allocator', 'cuda_malloc_async')
        memory_growth = gpu_config.get('memory_growth', True)
        
        mixed_precision_config = gpu_config.get('mixed_precision', {})
        if isinstance(mixed_precision_config, dict):
            mp_enabled = mixed_precision_config.get('enabled', True)
            mixed_precision_policy = mixed_precision_config.get('policy', 'mixed_float16') if mp_enabled else 'float32'
        else:
            mixed_precision_policy = 'float32'
    else:
        # Use provided arguments or defaults
        allocator = allocator or 'cuda_malloc_async'
        memory_growth = memory_growth if memory_growth is not None else True
        mixed_precision_policy = mixed_precision_policy or 'mixed_float16'
    
    logger.info("="*60)
    logger.info("Setting up GPU configuration")
    logger.info("="*60)
    
    # Step 1: Set GPU allocator (must be first)
    set_gpu_allocator(allocator)
    
    # Step 2: Set memory growth
    set_memory_growth(memory_growth)
    
    # Step 3: Set mixed precision policy
    set_mixed_precision_policy(mixed_precision_policy)
    
    # Gather GPU information
    gpus = tf.config.list_physical_devices('GPU')
    gpu_info = {
        'num_gpus': len(gpus),
        'gpus': [gpu.name for gpu in gpus],
        'allocator': allocator,
        'memory_growth': memory_growth,
        'mixed_precision': mixed_precision_policy,
    }
    
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
    else:
        logger.warning("No GPUs found - training will use CPU")
    
    logger.info("="*60)
    
    return gpu_info


def get_gpu_info() -> dict:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information including:
        - num_gpus: Number of available GPUs
        - gpu_names: List of GPU device names
        - memory_info: Memory information (if available)
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        'num_gpus': len(gpus),
        'gpu_names': [gpu.name for gpu in gpus],
    }
    
    # Try to get memory information
    if gpus:
        try:
            # This requires GPU to be initialized
            tf.config.experimental.get_memory_info('GPU:0')
            # If successful, could add detailed memory info here
        except Exception:
            pass
    
    return info


def print_gpu_info() -> None:
    """Print detailed GPU information."""
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs found")
    else:
        print(f"Number of GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu.name}")
            print(f"  Type: {gpu.device_type}")
    
    # Print current mixed precision policy
    policy = mixed_precision.global_policy()
    print(f"\nMixed Precision Policy: {policy}")
    print(f"  Compute dtype: {policy.compute_dtype}")
    print(f"  Variable dtype: {policy.variable_dtype}")
    
    print("="*60 + "\n")
