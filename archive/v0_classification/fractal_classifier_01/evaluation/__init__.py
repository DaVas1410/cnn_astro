"""Evaluation utilities for Fractal Classifier."""

from .evaluator import FractalEvaluator
from .metrics import (
    compute_centroid_k,
    compute_centroid_error,
    compute_classification_accuracy,
    compute_per_class_accuracy,
    compute_top_k_accuracy,
    compute_confidence_metrics,
    compute_all_metrics
)

__all__ = [
    'FractalEvaluator',
    'compute_centroid_k',
    'compute_centroid_error',
    'compute_classification_accuracy',
    'compute_per_class_accuracy',
    'compute_top_k_accuracy',
    'compute_confidence_metrics',
    'compute_all_metrics'
]
