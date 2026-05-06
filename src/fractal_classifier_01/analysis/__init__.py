"""Analysis utilities for fractal classifier models."""

from .kernel_analysis import (
    plot_kernel_statistics_evolution,
    plot_kernel_similarity_matrix,
    plot_kernel_convergence_trajectory,
    compare_kernels_with_reference,
    visualize_kernel_difference,
    analyze_kernel_convergence
)

__all__ = [
    'plot_kernel_statistics_evolution',
    'plot_kernel_similarity_matrix',
    'plot_kernel_convergence_trajectory',
    'compare_kernels_with_reference',
    'visualize_kernel_difference',
    'analyze_kernel_convergence'
]
