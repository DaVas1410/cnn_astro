"""
Regression evaluator for Fractal Classifier v0.2.

Computes MAE, RMSE, R² on the test set and exports per-sample predictions.
Standalone metric functions (compute_regression_metrics, compute_per_kmin_metrics)
have no TensorFlow dependency and can be used independently.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone metric functions (no TensorFlow dependency)
# ---------------------------------------------------------------------------

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Returns:
        Dict with keys: mae, rmse, r2, mean_bias, std_error.
    """
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return {
        'mae':       mae,
        'rmse':      rmse,
        'r2':        r2,
        'mean_bias': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
    }


def compute_per_kmin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Compute MAE and RMSE separately for each integer k_min value.

    Returns:
        Dict mapping integer k_min → {'mae': …, 'rmse': …, 'n': …}.
    """
    results = {}
    for k in sorted(np.unique(np.round(y_true).astype(int))):
        mask = np.round(y_true).astype(int) == k
        if not mask.any():
            continue
        yt, yp = y_true[mask], y_pred[mask]
        err = yp - yt
        results[int(k)] = {
            'mae':  float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'n':    int(mask.sum()),
        }
    return results


# ---------------------------------------------------------------------------
# Full evaluator (requires TensorFlow at runtime via lazy import)
# ---------------------------------------------------------------------------

class FractalEvaluator:
    """
    Evaluate a trained regression model on the test split.

    Args:
        model:  Trained Keras model.
        config: Config object (used for paths and evaluation settings).
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def evaluate(self, indexer) -> Dict[str, float]:
        """
        Run evaluation on the test split and return a metrics dictionary.

        Args:
            indexer: A built DataIndexer (its test_index is used).

        Returns:
            Dict with keys: mae, rmse, r2, mean_bias, std_error.
        """
        y_true, y_pred = self._predict_all(indexer)
        metrics = compute_regression_metrics(y_true, y_pred)
        self._log_metrics(metrics)
        if getattr(self.config.evaluation, 'export_predictions', True):
            self._export_predictions(indexer.test_index, y_pred)
        return metrics

    def _predict_all(self, indexer) -> Tuple[np.ndarray, np.ndarray]:
        """Collect ground-truth and predicted k_min over the full test set."""
        from ..data import DataPipeline
        pipeline = DataPipeline.from_config(indexer, self.config, training=False)
        test_ds = pipeline.build_test_dataset()

        y_true_list, y_pred_list = [], []
        for batch_images, batch_labels in test_ds:
            y_pred_list.append(self.model.predict(batch_images, verbose=0).flatten())
            y_true_list.append(batch_labels.numpy().flatten())

        return np.concatenate(y_true_list), np.concatenate(y_pred_list)

    @staticmethod
    def _log_metrics(metrics: Dict[str, float]):
        logger.info("=" * 50)
        logger.info("Evaluation results")
        logger.info("=" * 50)
        for name, value in metrics.items():
            logger.info(f"  {name:15s}: {value:.4f}")
        logger.info("=" * 50)

    def _export_predictions(self, test_index: np.ndarray, y_pred: np.ndarray):
        eval_dir = Path(self.config.paths.evaluation_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / 'predictions.csv'
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'resolution', 'image_idx', 'k_min_true', 'k_min_pred', 'error'])
            for record, pred in zip(test_index, y_pred):
                error = float(pred) - float(record['k_min'])
                writer.writerow([
                    record['file_path'], record['resolution'], record['image_idx'],
                    f"{record['k_min']:.4f}", f"{pred:.4f}", f"{error:.4f}",
                ])
        logger.info(f"Predictions exported to {out_path}")
