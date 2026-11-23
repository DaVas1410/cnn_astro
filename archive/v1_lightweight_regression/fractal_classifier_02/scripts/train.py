#!/usr/bin/env python3
"""
Train the Fractal Classifier v0.2 regression model.

Usage:
    python scripts/train.py --config config/default_config.yaml
    python scripts/train.py  # uses default config
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fractal_classifier_02.config import load_config
from fractal_classifier_02.training import FractalRegressorTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():
    parser = argparse.ArgumentParser(description='Train Fractal Classifier v0.2')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (default: built-in defaults)')
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = FractalRegressorTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
