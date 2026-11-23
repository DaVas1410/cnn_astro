"""
Training history and analysis plotting for Fractal Classifier.

This module provides functions to visualize training metrics, including
loss curves, accuracy curves, learning rate schedules, and overfitting analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def set_plot_style(style: str = 'seaborn-v0_8-darkgrid') -> None:
    """
    Set matplotlib style for plots.
    
    Args:
        style: Matplotlib style name
    """
    try:
        plt.style.use(style)
    except Exception as e:
        logger.warning(f"Could not set style '{style}': {e}")
        logger.warning("Using default style")


def smooth_curve(values: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Smooth a curve using a moving average window.
    
    Args:
        values: Array of values to smooth
        window: Window size for moving average
    
    Returns:
        Smoothed array (same length as input)
    """
    if window <= 1:
        return values
    
    values = np.asarray(values)
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    
    # Pad to match original length
    pad_size = len(values) - len(smoothed)
    if pad_size > 0:
        smoothed = np.pad(smoothed, (pad_size, 0), mode='edge')
    
    return smoothed


def plot_training_history(history_df: pd.DataFrame,
                          output_path: Optional[str] = None,
                          metrics: Optional[List[str]] = None,
                          dpi: int = 300,
                          formats: List[str] = ['png'],
                          smooth_window: int = 1,
                          figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot basic training history (loss and accuracy).
    
    Args:
        history_df: DataFrame with training history (columns: loss, val_loss, etc.)
        output_path: Path to save figure (without extension)
        metrics: List of metric names to plot (default: loss and accuracy)
        dpi: DPI for saved figures
        formats: List of output formats ('png', 'pdf', etc.)
        smooth_window: Window size for smoothing (1 = no smoothing)
        figsize: Figure size (width, height)
    """
    if metrics is None:
        # Auto-detect metrics from DataFrame columns
        if 'loss' in history_df.columns:
            metrics = ['loss']
            if any('accuracy' in col for col in history_df.columns):
                # Find the accuracy metric
                acc_cols = [col for col in history_df.columns if 'accuracy' in col and not col.startswith('val_')]
                if acc_cols:
                    metrics.append(acc_cols[0])
        else:
            raise ValueError("No 'loss' column found in history DataFrame")
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    epochs = np.arange(1, len(history_df) + 1)
    
    for ax, metric in zip(axes, metrics):
        # Training metric
        train_values = history_df[metric].values
        if smooth_window > 1:
            train_values = smooth_curve(train_values, smooth_window)
        ax.plot(epochs, train_values, label=f'Training', linewidth=2)
        
        # Validation metric (if available)
        val_metric = f'val_{metric}'
        if val_metric in history_df.columns:
            val_values = history_df[val_metric].values
            if smooth_window > 1:
                val_values = smooth_curve(val_values, smooth_window)
            ax.plot(epochs, val_values, label=f'Validation', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file(s)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            save_path = output_path.with_suffix(f'.{fmt}')
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
    
    plt.close(fig)


def plot_comprehensive_history(history_df: pd.DataFrame,
                               output_path: Optional[str] = None,
                               dpi: int = 300,
                               formats: List[str] = ['png'],
                               smooth_window: int = 3,
                               figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot comprehensive training history with multiple subplots.
    
    Creates a 2x2 grid with:
    - Loss curves
    - Accuracy curves  
    - Learning rate schedule
    - Train-val gap analysis
    
    Args:
        history_df: DataFrame with training history
        output_path: Path to save figure (without extension)
        dpi: DPI for saved figures
        formats: List of output formats
        smooth_window: Window size for smoothing
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    epochs = np.arange(1, len(history_df) + 1)
    
    # Find accuracy metric name
    acc_metric = None
    for col in history_df.columns:
        if 'accuracy' in col and not col.startswith('val_'):
            acc_metric = col
            break
    
    # Subplot 1: Loss
    ax = axes[0, 0]
    train_loss = history_df['loss'].values
    val_loss = history_df['val_loss'].values if 'val_loss' in history_df.columns else None
    
    if smooth_window > 1:
        train_loss_smooth = smooth_curve(train_loss, smooth_window)
        ax.plot(epochs, train_loss_smooth, label='Train (smoothed)', linewidth=2)
        ax.plot(epochs, train_loss, alpha=0.3, linewidth=1, label='Train (raw)')
        
        if val_loss is not None:
            val_loss_smooth = smooth_curve(val_loss, smooth_window)
            ax.plot(epochs, val_loss_smooth, label='Val (smoothed)', linewidth=2)
            ax.plot(epochs, val_loss, alpha=0.3, linewidth=1, label='Val (raw)')
    else:
        ax.plot(epochs, train_loss, label='Train', linewidth=2)
        if val_loss is not None:
            ax.plot(epochs, val_loss, label='Validation', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    ax = axes[0, 1]
    if acc_metric:
        train_acc = history_df[acc_metric].values
        val_acc = history_df[f'val_{acc_metric}'].values if f'val_{acc_metric}' in history_df.columns else None
        
        if smooth_window > 1:
            train_acc_smooth = smooth_curve(train_acc, smooth_window)
            ax.plot(epochs, train_acc_smooth, label='Train (smoothed)', linewidth=2)
            ax.plot(epochs, train_acc, alpha=0.3, linewidth=1, label='Train (raw)')
            
            if val_acc is not None:
                val_acc_smooth = smooth_curve(val_acc, smooth_window)
                ax.plot(epochs, val_acc_smooth, label='Val (smoothed)', linewidth=2)
                ax.plot(epochs, val_acc, alpha=0.3, linewidth=1, label='Val (raw)')
        else:
            ax.plot(epochs, train_acc, label='Train', linewidth=2)
            if val_acc is not None:
                ax.plot(epochs, val_acc, label='Validation', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No accuracy metric found', 
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # Subplot 3: Learning Rate (if available)
    ax = axes[1, 0]
    if 'lr' in history_df.columns:
        lr = history_df['lr'].values
        ax.semilogy(epochs, lr, linewidth=2, color='green')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate (log scale)', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No learning rate data', 
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # Subplot 4: Train-Val Gap
    ax = axes[1, 1]
    if val_loss is not None and acc_metric and f'val_{acc_metric}' in history_df.columns:
        # Compute gaps
        loss_gap = train_loss - val_loss
        acc_gap = train_acc - val_acc
        
        if smooth_window > 1:
            loss_gap = smooth_curve(loss_gap, smooth_window)
            acc_gap = smooth_curve(acc_gap, smooth_window)
        
        ax.plot(epochs, loss_gap, label='Loss Gap (Train - Val)', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(epochs, acc_gap, label='Acc Gap (Train - Val)', 
                linewidth=2, color='orange', linestyle='--')
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss Gap', fontsize=11)
        ax2.set_ylabel('Accuracy Gap', fontsize=11)
        ax.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for gap analysis', 
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save to file(s)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            save_path = output_path.with_suffix(f'.{fmt}')
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved comprehensive plot to {save_path}")
    
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         output_path: Optional[str] = None,
                         dpi: int = 300,
                         normalize: bool = False,
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names for axis labels
        output_path: Path to save figure
        dpi: DPI for saved figure
        normalize: Whether to normalize by true class counts
        figsize: Figure size
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'Blues'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Proportion' if normalize else 'Count'},
               ax=ax)
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close(fig)


def plot_centroid_analysis(df: pd.DataFrame,
                          output_path: Optional[str] = None,
                          dpi: int = 300,
                          figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Plot centroid prediction analysis.
    
    Creates visualizations showing:
    - Centroid predictions vs true k-values
    - Distribution of centroid errors
    
    Args:
        df: DataFrame with columns 'true_k' and 'centroid_k'
        output_path: Path to save figure
        dpi: DPI for saved figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Subplot 1: Centroid predictions vs true k
    ax = axes[0]
    for true_k in sorted(df['true_k'].unique()):
        mask = df['true_k'] == true_k
        centroids = df.loc[mask, 'centroid_k']
        ax.scatter([true_k] * len(centroids), centroids, alpha=0.5, s=20)
    
    # Plot ideal line
    k_range = [df['true_k'].min(), df['true_k'].max()]
    ax.plot(k_range, k_range, 'r--', linewidth=2, label='Ideal (y=x)')
    
    ax.set_xlabel('True k-value', fontsize=12)
    ax.set_ylabel('Predicted Centroid k-value', fontsize=12)
    ax.set_title('Centroid Predictions vs True k-values', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Error distribution
    ax = axes[1]
    df['centroid_error'] = np.abs(df['centroid_k'] - df['true_k'])
    
    for true_k in sorted(df['true_k'].unique()):
        mask = df['true_k'] == true_k
        errors = df.loc[mask, 'centroid_error']
        ax.hist(errors, bins=20, alpha=0.5, label=f'k={true_k}')
    
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Centroid Errors', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved centroid analysis to {output_path}")
    
    plt.close(fig)
