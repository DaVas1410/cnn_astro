"""Utility functions for Fractal Classifier."""

from .gpu import (
    setup_gpu,
    set_gpu_allocator,
    set_memory_growth,
    set_mixed_precision_policy,
    get_gpu_info,
    print_gpu_info
)

from .kernels import (
    extract_conv_kernels,
    extract_all_layer_weights,
    save_kernels_hdf5,
    save_kernels_npz,
    load_kernels_hdf5,
    load_kernels_npz,
    compute_kernel_statistics,
    compute_kernel_similarity,
    compare_kernel_evolution,
    export_model_kernels
)

__all__ = [
    # GPU utilities
    'setup_gpu',
    'set_gpu_allocator',
    'set_memory_growth',
    'set_mixed_precision_policy',
    'get_gpu_info',
    'print_gpu_info',
    # Kernel utilities
    'extract_conv_kernels',
    'extract_all_layer_weights',
    'save_kernels_hdf5',
    'save_kernels_npz',
    'load_kernels_hdf5',
    'load_kernels_npz',
    'compute_kernel_statistics',
    'compute_kernel_similarity',
    'compare_kernel_evolution',
    'export_model_kernels'
]
