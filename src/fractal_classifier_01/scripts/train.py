#!/usr/bin/env python3
"""
Training script for Fractal Classifier.

This script provides a command-line interface for training the fractal
classification model using configuration files.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import fractal_classifier
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fractal_classifier.config import load_config, save_config
from fractal_classifier.training import FractalClassifierTrainer


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
        description='Train a fractal cloud classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file (uses defaults if not provided)'
    )
    
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        help='Path to HDF5 dataset file (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Checkpoint directory (overrides config)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU (use CPU only)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save the final config (with overrides) to this path and exit'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FRACTAL CLASSIFIER - TRAINING")
    logger.info("="*80)
    
    # Build configuration overrides
    overrides = {}
    
    if args.data_path:
        overrides['data'] = overrides.get('data', {})
        overrides['data']['h5_path'] = args.data_path
    
    if args.output_dir:
        overrides['paths'] = overrides.get('paths', {})
        overrides['paths']['output_dir'] = args.output_dir
    
    if args.checkpoint_dir:
        overrides['paths'] = overrides.get('paths', {})
        overrides['paths']['checkpoint_dir'] = args.checkpoint_dir
    
    if args.epochs:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['epochs'] = args.epochs
    
    if args.batch_size:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['batch_size'] = args.batch_size
    
    if args.learning_rate:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['learning_rate'] = args.learning_rate
    
    if args.no_gpu:
        overrides['gpu'] = overrides.get('gpu', {})
        overrides['gpu']['mixed_precision'] = {'enabled': False}
    
    # Load configuration
    try:
        config = load_config(args.config, overrides=overrides if overrides else None)
        logger.info(f"Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # If --save-config is specified, save and exit
    if args.save_config:
        try:
            save_config(config, args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")
            return 0
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return 1
    
    # Display configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  Dataset: {config.data.h5_path}")
    print(f"  Output dir: {config.paths.output_dir}")
    print(f"  Checkpoint dir: {config.paths.checkpoint_dir}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Mixed precision: {config.gpu.mixed_precision.enabled}")
    print()
    
    # Initialize trainer
    try:
        trainer = FractalClassifierTrainer(config, verbose=True)
        logger.info("Trainer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return 1
    
    # Setup (GPU, data, model)
    try:
        trainer.setup()
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1
    
    # Load checkpoint if resuming
    if args.resume_from:
        try:
            trainer.load_checkpoint(args.resume_from)
            logger.info(f"Resumed from checkpoint: {args.resume_from}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 1
    
    # Display model summary
    trainer.print_summary()
    
    # Train
    try:
        history = trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    
    # Save training history
    try:
        trainer.save_history()
        logger.info("Training history saved")
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")
    
    # Evaluate on test set
    try:
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate('test')
        print(f"Test results: {test_results}")
    except Exception as e:
        logger.warning(f"Test evaluation failed: {e}")
    
    logger.info("="*80)
    logger.info("Training pipeline completed!")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
