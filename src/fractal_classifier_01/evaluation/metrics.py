"""
Custom metrics for Fractal Classifier evaluation.

This module provides metric functions specific to fractal classification,
including centroid-based k-value prediction and error metrics.
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def compute_centroid_k(probabilities: np.ndarray, 
                      k_values: np.ndarray) -> np.ndarray:
    """
    Compute centroid k-value prediction from class probabilities.
    
    The centroid is calculated as the weighted average of k-values
    using the predicted class probabilities as weights:
    
        centroid_k = sum(prob_i * k_i) for all classes i
    
    Or equivalently: centroid_k = probabilities @ k_values
    
    This provides a continuous prediction that can interpolate between
    the discrete training k-values.
    
    Args:
        probabilities: Array of shape (N, num_classes) with class probabilities
        k_values: Array of shape (num_classes,) with k-values for each class
    
    Returns:
        Array of shape (N,) with centroid k-value for each sample
    
    Example:
        >>> probs = np.array([[0.1, 0.3, 0.6], [0.8, 0.15, 0.05]])
        >>> k_vals = np.array([2, 4, 8])
        >>> compute_centroid_k(probs, k_vals)
        array([5.6, 2.6])
    """
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must have 2 dimensions, got {probabilities.ndim}")
    
    if k_values.ndim != 1:
        raise ValueError(f"k_values must have 1 dimension, got {k_values.ndim}")
    
    if probabilities.shape[1] != len(k_values):
        raise ValueError(f"probabilities shape {probabilities.shape} incompatible with "
                        f"k_values shape {k_values.shape}")
    
    # Compute weighted average
    centroid = probabilities @ k_values
    
    return centroid


def compute_centroid_error(centroid_k: np.ndarray,
                          true_k: np.ndarray,
                          metric: str = 'mae') -> float:
    """
    Compute error between predicted centroid k-values and true k-values.
    
    Args:
        centroid_k: Predicted centroid k-values
        true_k: True k-values
        metric: Error metric to use ('mae', 'mse', 'rmse', 'mape')
    
    Returns:
        Error value
    
    Raises:
        ValueError: If metric is unknown
    """
    centroid_k = np.asarray(centroid_k)
    true_k = np.asarray(true_k)
    
    if centroid_k.shape != true_k.shape:
        raise ValueError(f"Shape mismatch: centroid_k {centroid_k.shape} vs true_k {true_k.shape}")
    
    metric = metric.lower()
    
    if metric == 'mae':
        # Mean Absolute Error
        return np.mean(np.abs(centroid_k - true_k))
    
    elif metric == 'mse':
        # Mean Squared Error
        return np.mean((centroid_k - true_k) ** 2)
    
    elif metric == 'rmse':
        # Root Mean Squared Error
        return np.sqrt(np.mean((centroid_k - true_k) ** 2))
    
    elif metric == 'mape':
        # Mean Absolute Percentage Error
        # Avoid division by zero
        mask = true_k != 0
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((centroid_k[mask] - true_k[mask]) / true_k[mask])) * 100
    
    else:
        raise ValueError(f"Unknown metric: {metric}. "
                        f"Choose from 'mae', 'mse', 'rmse', 'mape'")


def compute_classification_accuracy(predictions: np.ndarray,
                                   true_labels: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted class indices
        true_labels: True class indices
    
    Returns:
        Accuracy (fraction of correct predictions)
    """
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    
    if predictions.shape != true_labels.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs "
                        f"true_labels {true_labels.shape}")
    
    return np.mean(predictions == true_labels)


def compute_per_class_accuracy(predictions: np.ndarray,
                               true_labels: np.ndarray,
                               num_classes: int = None) -> Dict[int, float]:
    """
    Compute accuracy for each class.
    
    Args:
        predictions: Predicted class indices
        true_labels: True class indices
        num_classes: Number of classes (if None, inferred from labels)
    
    Returns:
        Dictionary mapping class index to accuracy
    """
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    
    if num_classes is None:
        num_classes = max(true_labels.max(), predictions.max()) + 1
    
    per_class_acc = {}
    for c in range(num_classes):
        mask = true_labels == c
        if np.sum(mask) > 0:
            per_class_acc[c] = np.mean(predictions[mask] == true_labels[mask])
        else:
            per_class_acc[c] = np.nan
    
    return per_class_acc


def compute_top_k_accuracy(probabilities: np.ndarray,
                          true_labels: np.ndarray,
                          k: int = 3) -> float:
    """
    Compute top-k accuracy.
    
    A prediction is considered correct if the true label is among the
    top k predictions with highest probability.
    
    Args:
        probabilities: Array of shape (N, num_classes) with class probabilities
        true_labels: Array of shape (N,) with true class indices
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy
    """
    probabilities = np.asarray(probabilities)
    true_labels = np.asarray(true_labels)
    
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must have 2 dimensions, got {probabilities.ndim}")
    
    # Get top-k predicted classes
    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = np.array([true_labels[i] in top_k_preds[i] for i in range(len(true_labels))])
    
    return np.mean(correct)


def compute_confidence_metrics(probabilities: np.ndarray) -> Dict[str, float]:
    """
    Compute confidence-related metrics from prediction probabilities.
    
    Args:
        probabilities: Array of shape (N, num_classes) with class probabilities
    
    Returns:
        Dictionary with confidence metrics:
        - mean_confidence: Average max probability
        - median_confidence: Median max probability
        - min_confidence: Minimum max probability
        - std_confidence: Standard deviation of max probabilities
    """
    probabilities = np.asarray(probabilities)
    
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must have 2 dimensions, got {probabilities.ndim}")
    
    # Get maximum probability for each prediction (confidence)
    confidences = probabilities.max(axis=1)
    
    return {
        'mean_confidence': float(np.mean(confidences)),
        'median_confidence': float(np.median(confidences)),
        'min_confidence': float(np.min(confidences)),
        'max_confidence': float(np.max(confidences)),
        'std_confidence': float(np.std(confidences))
    }


def compute_all_metrics(probabilities: np.ndarray,
                       true_labels: np.ndarray,
                       k_values: np.ndarray,
                       true_k_values: np.ndarray = None) -> Dict[str, any]:
    """
    Compute all available metrics.
    
    Args:
        probabilities: Array of shape (N, num_classes) with class probabilities
        true_labels: Array of shape (N,) with true class indices
        k_values: Array of shape (num_classes,) with k-values for each class
        true_k_values: Optional array of true k-values for centroid error
    
    Returns:
        Dictionary with all computed metrics
    """
    results = {}
    
    # Classification metrics
    predictions = probabilities.argmax(axis=1)
    results['accuracy'] = compute_classification_accuracy(predictions, true_labels)
    results['top3_accuracy'] = compute_top_k_accuracy(probabilities, true_labels, k=3)
    
    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(predictions, true_labels)
    for class_idx, acc in per_class_acc.items():
        results[f'class_{class_idx}_accuracy'] = acc
    
    # Centroid predictions
    centroid_k = compute_centroid_k(probabilities, k_values)
    results['mean_centroid_k'] = float(np.mean(centroid_k))
    results['std_centroid_k'] = float(np.std(centroid_k))
    
    # Centroid errors (if true k-values provided)
    if true_k_values is not None:
        results['centroid_mae'] = compute_centroid_error(centroid_k, true_k_values, 'mae')
        results['centroid_rmse'] = compute_centroid_error(centroid_k, true_k_values, 'rmse')
        results['centroid_mape'] = compute_centroid_error(centroid_k, true_k_values, 'mape')
    
    # Confidence metrics
    confidence_metrics = compute_confidence_metrics(probabilities)
    results.update(confidence_metrics)
    
    return results
