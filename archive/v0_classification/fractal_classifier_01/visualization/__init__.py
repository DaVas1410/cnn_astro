"""Visualization utilities for Fractal Classifier."""

from .plots import (
    plot_training_history,
    plot_comprehensive_history,
    plot_confusion_matrix,
    plot_centroid_analysis,
    set_plot_style,
    smooth_curve
)
from .activations import visualize_activations, save_intermediate_outputs
from .kernels import visualize_kernels, visualize_first_layer_kernels

__all__ = [
    'plot_training_history',
    'plot_comprehensive_history',
    'plot_confusion_matrix',
    'plot_centroid_analysis',
    'set_plot_style',
    'smooth_curve',
    'visualize_activations',
    'save_intermediate_outputs',
    'visualize_kernels',
    'visualize_first_layer_kernels'
]
