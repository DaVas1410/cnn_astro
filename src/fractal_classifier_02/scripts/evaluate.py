#!/usr/bin/env python3
"""
Evaluate a trained Fractal Classifier v0.2 model.

Usage:
    python scripts/evaluate.py \\
        --config config/default_config.yaml \\
        --checkpoint outputs/checkpoints/best.keras
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
from fractal_classifier_02.config import load_config
from fractal_classifier_02.data import DataIndexer
from fractal_classifier_02.evaluation import FractalEvaluator, compute_per_kmin_metrics
from fractal_classifier_02.visualization import (
    plot_prediction_scatter, plot_residuals, plot_error_by_kmin
)
from fractal_classifier_02.utils import setup_gpu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fractal Classifier v0.2')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to saved .keras model file')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_gpu(config)

    logger.info(f"Loading model from {args.checkpoint}")
    model = tf.keras.models.load_model(args.checkpoint)

    data_cfg = config.data
    split_cfg = data_cfg.split
    indexer = DataIndexer(
        dataset_dir=data_cfg.dataset_dir,
        resolutions=[int(r) for r in data_cfg.resolutions],
        split_ratios=(
            float(split_cfg.train_ratio),
            float(split_cfg.val_ratio),
            float(split_cfg.test_ratio),
        ),
        random_seed=int(getattr(split_cfg, 'random_seed', 42)),
    )
    indexer.build()

    evaluator = FractalEvaluator(model, config)
    metrics = evaluator.evaluate(indexer)

    # Per-class metrics and visualisation require collecting predictions
    from fractal_classifier_02.data import DataPipeline
    import numpy as np

    pipeline = DataPipeline.from_config(indexer, config, training=False)
    test_ds = pipeline.build_test_dataset()
    y_true_list, y_pred_list = [], []
    for imgs, lbls in test_ds:
        y_pred_list.append(model.predict(imgs, verbose=0).flatten())
        y_true_list.append(lbls.numpy().flatten())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    plots_dir = config.paths.plots_dir
    fmts = list(config.visualization.formats)
    dpi = int(config.visualization.dpi)

    plot_prediction_scatter(y_true, y_pred, plots_dir, fmts, dpi)
    plot_residuals(y_true, y_pred, plots_dir, fmts, dpi)

    per_kmin = compute_per_kmin_metrics(y_true, y_pred)
    plot_error_by_kmin(per_kmin, plots_dir, fmts, dpi)

    logger.info("Evaluation complete.")


if __name__ == '__main__':
    main()
