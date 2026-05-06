"""
Visualization utilities for Fractal Classifier v0.2.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _save_fig(fig, output_path: Path, formats: Sequence[str], dpi: int):
    import matplotlib.pyplot as plt
    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved: {path}")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: str = 'outputs/plots',
    smooth_window: int = 5,
    formats: Sequence[str] = ('png',),
    dpi: int = 150,
) -> None:
    """
    Plot training and validation loss and MAE curves.

    Args:
        history:      Dict from keras History.history (or loaded JSON).
        output_dir:   Directory to save plots.
        smooth_window: Rolling average window for smoothing (1 = disabled).
        formats:      Output image formats.
        dpi:          Resolution.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def smooth(values, w):
        if w <= 1 or len(values) < w:
            return values
        kernel = np.ones(w) / w
        padded = np.pad(values, (w // 2, w // 2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')[:len(values)]

    metrics_to_plot = [
        ('loss', 'val_loss', 'Loss'),
        ('mae', 'val_mae', 'MAE (k_min scale)'),
    ]

    for train_key, val_key, ylabel in metrics_to_plot:
        if train_key not in history:
            continue
        epochs = range(1, len(history[train_key]) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        train_vals = np.array(history[train_key], dtype=float)
        ax.plot(epochs, train_vals, alpha=0.3, color='steelblue', label='train (raw)')
        ax.plot(epochs, smooth(train_vals, smooth_window), color='steelblue', label='train (smooth)')
        if val_key in history:
            val_vals = np.array(history[val_key], dtype=float)
            ax.plot(epochs, val_vals, alpha=0.3, color='tomato', label='val (raw)')
            ax.plot(epochs, smooth(val_vals, smooth_window), color='tomato', label='val (smooth)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Training — {ylabel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save_fig(fig, output_dir / train_key, formats, dpi)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = 'outputs/plots',
    formats: Sequence[str] = ('png',),
    dpi: int = 150,
) -> None:
    """
    Scatter plot of predicted vs true k_min values.

    A perfect model would lie on the diagonal (y = x).
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color='steelblue', rasterized=True)

    lim_min = min(y_true.min(), y_pred.min()) - 1
    lim_max = max(y_true.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, label='ideal')

    mae = float(np.mean(np.abs(y_pred - y_true)))
    r2  = 1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    ax.set_xlabel('True k_min')
    ax.set_ylabel('Predicted k_min')
    ax.set_title(f'Prediction scatter  |  MAE={mae:.3f}  R²={r2:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    _save_fig(fig, output_dir / 'scatter', formats, dpi)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = 'outputs/plots',
    formats: Sequence[str] = ('png',),
    dpi: int = 150,
) -> None:
    """
    Residual plot: (predicted - true) vs true k_min.

    A well-calibrated model shows residuals centred on zero with no
    systematic trend across k_min values.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_true, residuals, alpha=0.3, s=8, color='steelblue', rasterized=True)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('True k_min')
    ax.set_ylabel('Residual (pred − true)')
    ax.set_title('Residuals vs True k_min')
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_dir / 'residuals', formats, dpi)


def plot_error_by_kmin(
    per_kmin_metrics: Dict[int, Dict[str, float]],
    output_dir: str = 'outputs/plots',
    formats: Sequence[str] = ('png',),
    dpi: int = 150,
) -> None:
    """
    Bar chart of MAE per k_min class.

    Args:
        per_kmin_metrics: Output of evaluation.compute_per_kmin_metrics().
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_vals = sorted(per_kmin_metrics.keys())
    maes   = [per_kmin_metrics[k]['mae'] for k in k_vals]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(k_vals, maes, color='steelblue', edgecolor='white', width=0.8)
    ax.set_xlabel('k_min')
    ax.set_ylabel('MAE')
    ax.set_title('MAE per k_min value')
    ax.set_xticks(k_vals)
    ax.grid(True, axis='y', alpha=0.3)

    _save_fig(fig, output_dir / 'error_by_kmin', formats, dpi)
