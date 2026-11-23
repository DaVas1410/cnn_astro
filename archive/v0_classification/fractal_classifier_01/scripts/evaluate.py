#!/usr/bin/env python3
"""
Evaluation script for Fractal Classifier.

This script provides a command-line interface for evaluating trained models
on test datasets and generating prediction reports.
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fractal_classifier.config import load_config
from fractal_classifier.models import load_model
from fractal_classifier.evaluation import FractalEvaluator
from fractal_classifier.visualization import plot_confusion_matrix, plot_centroid_analysis


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
        description='Evaluate fractal cloud classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model file (.h5)'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to HDF5 dataset file for evaluation'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[2, 4, 8, 16, 32, 64],
        help='K-values corresponding to model classes (in order)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for prediction'
    )
    
    parser.add_argument(
        '--intermediate',
        action='store_true',
        help='Evaluate on intermediate k-values (generalization test)'
    )
    
    parser.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Generate confusion matrix'
    )
    
    parser.add_argument(
        '--confusion-samples',
        type=int,
        default=20,
        help='Samples per class for confusion matrix'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum samples per group to evaluate (for quick tests)'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FRACTAL CLASSIFIER - EVALUATION")
    logger.info("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load model
    try:
        logger.info(f"Loading model from {args.model}")
        model = load_model(args.model)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Initialize evaluator
    k_values = np.array(args.k_values)
    logger.info(f"K-values: {k_values.tolist()}")
    
    try:
        evaluator = FractalEvaluator(
            model=model,
            k_values=k_values,
            batch_size=args.batch_size
        )
        logger.info("Evaluator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1
    
    # Perform evaluation
    if args.intermediate:
        # Intermediate k-value evaluation (generalization test)
        logger.info("Performing intermediate k-value evaluation...")
        try:
            results = evaluator.evaluate_intermediate_k(
                h5_path=args.data,
                output_dir=str(output_dir)
            )
            
            # Print summary
            evaluator.print_evaluation_summary(results['per_image'])
            
            # Generate plots if requested
            if args.plots:
                logger.info("Generating plots...")
                plot_centroid_analysis(
                    results['per_image'],
                    output_path=str(output_dir / 'centroid_analysis.png')
                )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return 1
    
    else:
        # Standard evaluation
        logger.info("Performing standard evaluation...")
        try:
            df = evaluator.evaluate_on_hdf5(
                h5_path=args.data,
                max_samples_per_group=args.max_samples
            )
            
            # Save results
            output_csv = output_dir / 'evaluation_results.csv'
            df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")
            
            # Print summary
            evaluator.print_evaluation_summary(df)
            
            # Generate plots if requested
            if args.plots:
                logger.info("Generating plots...")
                plot_centroid_analysis(
                    df,
                    output_path=str(output_dir / 'centroid_analysis.png')
                )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return 1
    
    # Generate confusion matrix if requested
    if args.confusion_matrix:
        logger.info("Computing confusion matrix...")
        try:
            cm, y_true, y_pred = evaluator.compute_confusion_matrix(
                h5_path=args.data,
                samples_per_class=args.confusion_samples
            )
            
            # Save confusion matrix
            cm_path = output_dir / 'confusion_matrix.npy'
            np.save(cm_path, cm)
            logger.info(f"Confusion matrix saved to {cm_path}")
            
            # Plot confusion matrix if plots enabled
            if args.plots:
                class_names = [f'k={k}' for k in k_values]
                plot_confusion_matrix(
                    cm,
                    class_names=class_names,
                    output_path=str(output_dir / 'confusion_matrix.png'),
                    normalize=False
                )
                plot_confusion_matrix(
                    cm,
                    class_names=class_names,
                    output_path=str(output_dir / 'confusion_matrix_normalized.png'),
                    normalize=True
                )
            
        except Exception as e:
            logger.error(f"Confusion matrix failed: {e}", exc_info=True)
    
    logger.info("="*80)
    logger.info("Evaluation completed!")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
