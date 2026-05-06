#!/usr/bin/env python3
"""
Analysis script for Fractal Classifier.

This script provides a command-line interface for analyzing training results,
generating visualizations, and examining model behavior.
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fractal_classifier.visualization import (
    plot_training_history,
    plot_comprehensive_history,
    visualize_activations,
    visualize_kernels,
    set_plot_style
)
from fractal_classifier.models import load_model
from fractal_classifier.data import read_image
import numpy as np


def setup_logging(level: str = 'INFO') -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze fractal classifier training and model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--history', '-hist',
        type=str,
        help='Path to training history CSV file'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to trained model file (.h5)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to HDF5 dataset (for activation visualization)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='analysis_results',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--plot-style',
        type=str,
        default='seaborn-v0_8-darkgrid',
        help='Matplotlib style for plots'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved plots'
    )
    
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['png', 'pdf'],
        help='Output formats for plots'
    )
    
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=3,
        help='Smoothing window for history plots (1 = no smoothing)'
    )
    
    parser.add_argument(
        '--simple-history',
        action='store_true',
        help='Generate simple history plot (instead of comprehensive)'
    )
    
    parser.add_argument(
        '--visualize-kernels',
        action='store_true',
        help='Visualize convolutional kernels'
    )
    
    parser.add_argument(
        '--visualize-activations',
        action='store_true',
        help='Visualize layer activations'
    )
    
    parser.add_argument(
        '--num-activation-samples',
        type=int,
        default=3,
        help='Number of images for activation visualization'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def load_sample_images(h5_path: str, num_samples: int = 3) -> np.ndarray:
    """Load sample images from HDF5 dataset."""
    import h5py
    
    images = []
    with h5py.File(h5_path, 'r') as f:
        groups = sorted([g for g in f.keys() if g.startswith('k_')],
                       key=lambda s: int(s.split('_')[1]))
        
        for group in groups[:num_samples]:
            img_dataset = f[f'{group}/images']
            img = np.array(img_dataset[0], dtype=np.float32)
            if img.ndim == 2:
                img = img[..., np.newaxis]
            images.append(img)
            
            if len(images) >= num_samples:
                break
    
    return np.stack(images, axis=0)


def main():
    """Main analysis function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FRACTAL CLASSIFIER - ANALYSIS")
    logger.info("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Set plot style
    set_plot_style(args.plot_style)
    
    # Analyze training history
    if args.history:
        logger.info(f"Analyzing training history from {args.history}")
        
        try:
            history_df = pd.read_csv(args.history)
            logger.info(f"Loaded {len(history_df)} epochs of training history")
            
            # Generate plots
            if args.simple_history:
                output_path = output_dir / 'training_history_simple'
                plot_training_history(
                    history_df,
                    output_path=str(output_path),
                    dpi=args.dpi,
                    formats=args.formats,
                    smooth_window=args.smooth_window
                )
                logger.info(f"Simple history plot saved")
            else:
                output_path = output_dir / 'training_history_full'
                plot_comprehensive_history(
                    history_df,
                    output_path=str(output_path),
                    dpi=args.dpi,
                    formats=args.formats,
                    smooth_window=args.smooth_window
                )
                logger.info(f"Comprehensive history plot saved")
            
            # Print summary statistics
            print("\nTraining Summary:")
            print(f"  Total epochs: {len(history_df)}")
            print(f"  Final train loss: {history_df['loss'].iloc[-1]:.4f}")
            if 'val_loss' in history_df.columns:
                print(f"  Final val loss: {history_df['val_loss'].iloc[-1]:.4f}")
                print(f"  Best val loss: {history_df['val_loss'].min():.4f} "
                      f"(epoch {history_df['val_loss'].idxmin() + 1})")
            
            # Check for accuracy columns
            acc_cols = [col for col in history_df.columns if 'accuracy' in col and not col.startswith('val_')]
            if acc_cols:
                acc_col = acc_cols[0]
                print(f"  Final train accuracy: {history_df[acc_col].iloc[-1]:.4f}")
                val_acc_col = f'val_{acc_col}'
                if val_acc_col in history_df.columns:
                    print(f"  Final val accuracy: {history_df[val_acc_col].iloc[-1]:.4f}")
                    print(f"  Best val accuracy: {history_df[val_acc_col].max():.4f} "
                          f"(epoch {history_df[val_acc_col].idxmax() + 1})")
            
        except Exception as e:
            logger.error(f"Failed to analyze history: {e}", exc_info=True)
    
    # Analyze model
    if args.model:
        logger.info(f"Loading model from {args.model}")
        
        try:
            model = load_model(args.model)
            logger.info(f"Model loaded: {model.count_params():,} parameters")
            
            # Print model summary
            print("\nModel Architecture:")
            model.summary()
            
            # Visualize kernels
            if args.visualize_kernels:
                logger.info("Visualizing convolutional kernels...")
                kernels_dir = output_dir / 'kernels'
                visualize_kernels(
                    model,
                    output_dir=str(kernels_dir),
                    dpi=args.dpi
                )
                logger.info(f"Kernel visualizations saved to {kernels_dir}")
            
            # Visualize activations
            if args.visualize_activations:
                if not args.data:
                    logger.error("--data is required for activation visualization")
                else:
                    logger.info("Visualizing layer activations...")
                    
                    # Load sample images
                    images = load_sample_images(
                        args.data,
                        num_samples=args.num_activation_samples
                    )
                    logger.info(f"Loaded {len(images)} sample images")
                    
                    # Visualize activations
                    activations_dir = output_dir / 'activations'
                    visualize_activations(
                        model,
                        images,
                        output_dir=str(activations_dir),
                        dpi=args.dpi
                    )
                    logger.info(f"Activation visualizations saved to {activations_dir}")
            
        except Exception as e:
            logger.error(f"Failed to analyze model: {e}", exc_info=True)
    
    logger.info("="*80)
    logger.info("Analysis completed!")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
