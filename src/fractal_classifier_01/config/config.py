"""
Configuration management for Fractal Classifier.

This module provides utilities to load, validate, and merge configuration files
with default values. Configurations are stored in YAML format.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration container for Fractal Classifier.
    
    Provides dictionary-like access to nested configuration values with
    dot notation support (e.g., config.data.batch_size).
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
        """
        self._config = config_dict
        self._make_nested_accessible(config_dict)
    
    def _make_nested_accessible(self, config_dict: Dict[str, Any], parent_key: str = ''):
        """Recursively make nested dictionaries accessible as attributes."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration values."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Dictionary-style setting of configuration values."""
        self._config[key] = value
        setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a standard dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: Optional[Union[str, Path]] = None, 
                overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from YAML file and merge with defaults.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses default config.
        overrides: Dictionary of values to override in the loaded config.
    
    Returns:
        Config object with merged configuration values.
    
    Raises:
        FileNotFoundError: If config_path is specified but doesn't exist.
        ValueError: If configuration validation fails.
    """
    # Load default configuration
    default_config_path = Path(__file__).parent / 'default_config.yaml'
    with open(default_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Load user configuration if provided
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge user config with defaults
        config_dict = deep_merge(config_dict, user_config)
    else:
        logger.info("Using default configuration")
    
    # Apply overrides if provided
    if overrides:
        logger.info(f"Applying configuration overrides: {overrides}")
        config_dict = deep_merge(config_dict, overrides)
    
    # Validate configuration
    validate_config(config_dict)
    
    return Config(config_dict)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Values in 'override' take precedence over values in 'base'.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
    
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config_dict: Dict[str, Any]) -> None:
    """
    Validate configuration values.
    
    Args:
        config_dict: Configuration dictionary to validate
    
    Raises:
        ValueError: If validation fails
    """
    # Validate required top-level sections
    required_sections = ['data', 'model', 'training', 'gpu', 'paths']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data configuration
    data_config = config_dict['data']
    if 'input_shape' not in data_config:
        raise ValueError("Missing required 'input_shape' in data configuration")
    if len(data_config['input_shape']) != 3:
        raise ValueError("input_shape must have 3 dimensions (height, width, channels)")
    
    # Validate split ratios sum to 1.0
    split_config = data_config.get('split', {})
    if split_config:
        train = split_config.get('train_ratio', 0)
        val = split_config.get('val_ratio', 0)
        test = split_config.get('test_ratio', 0)
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # Validate model configuration
    model_config = config_dict['model']
    if 'filters' not in model_config:
        raise ValueError("Missing required 'filters' in model configuration")
    if not isinstance(model_config['filters'], list):
        raise ValueError("'filters' must be a list of integers")
    
    # Validate training configuration
    training_config = config_dict['training']
    if training_config.get('epochs', 0) <= 0:
        raise ValueError("epochs must be positive")
    if training_config.get('batch_size', 0) <= 0:
        raise ValueError("batch_size must be positive")
    if training_config.get('learning_rate', 0) <= 0:
        raise ValueError("learning_rate must be positive")
    
    logger.info("Configuration validation passed")


def save_config(config: Union[Config, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Config object or dictionary to save
        output_path: Path where to save the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict() if isinstance(config, Config) else config
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {output_path}")


def create_example_config(output_path: Union[str, Path], 
                         style: str = 'minimal') -> None:
    """
    Create an example configuration file.
    
    Args:
        output_path: Path where to save the example configuration
        style: 'minimal' for basic config, 'full' for all options
    """
    if style == 'minimal':
        example = {
            'data': {
                'h5_path': 'path/to/your/fractal_dataset.h5',
                'batch_size': 4,
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.0001,
            },
            'paths': {
                'output_dir': 'outputs',
                'checkpoint_dir': 'checkpoints',
            }
        }
    else:
        # Load full default config as example
        config = load_config()
        example = config.to_dict()
    
    save_config(example, output_path)
    logger.info(f"Example configuration created at {output_path}")
