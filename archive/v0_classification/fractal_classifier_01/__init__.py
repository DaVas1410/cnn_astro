"""
Fractal Classifier v0.1

A modular deep learning framework for classifying fractal cloud patterns
based on turbulence parameters.

Main Components:
    - config: Configuration management with YAML files
    - data: HDF5 data loading and TensorFlow pipelines
    - models: CNN model architecture
    - training: Training orchestration and callbacks
    - evaluation: Model evaluation with centroid metrics
    - visualization: Plotting and visualization utilities
    - utils: GPU configuration and utilities
"""

__version__ = '0.1.0'
__author__ = 'Fractal Cloud Research Team'

# Import main classes for convenient access
from .config import Config, load_config
from .training import FractalClassifierTrainer
from .evaluation import FractalEvaluator
from .models import build_fractal_cnn, load_model
from .data import DataIndexer, DataPipeline
from .utils import setup_gpu

__all__ = [
    '__version__',
    'Config',
    'load_config',
    'FractalClassifierTrainer',
    'FractalEvaluator',
    'build_fractal_cnn',
    'load_model',
    'DataIndexer',
    'DataPipeline',
    'setup_gpu',
]
