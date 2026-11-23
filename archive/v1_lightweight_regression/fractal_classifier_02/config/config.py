"""
Configuration management for Fractal Classifier v0.2.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """Nested configuration container with dot-notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        for key, value in config_dict.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        self._config[key] = value
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config({self._config})"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (override wins on conflict)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(cfg: Dict[str, Any]) -> None:
    """Lightweight validation of required fields."""
    for section in ('data', 'model', 'training', 'gpu', 'paths'):
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")

    data = cfg['data']
    if 'dataset_dir' not in data:
        raise ValueError("data.dataset_dir is required")
    if len(data.get('input_shape', data.get('model', {}).get('input_shape', []))) == 0:
        pass  # checked below via model section

    model = cfg['model']
    if 'input_shape' not in model:
        raise ValueError("model.input_shape is required")
    if len(model['input_shape']) != 3:
        raise ValueError("model.input_shape must have 3 elements: [H, W, C]")

    train = cfg['training']
    for param in ('epochs', 'batch_size', 'learning_rate'):
        if train.get(param, 0) <= 0:
            raise ValueError(f"training.{param} must be positive")

    split = data.get('split', {})
    if split:
        total = split.get('train_ratio', 0) + split.get('val_ratio', 0) + split.get('test_ratio', 0)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total:.4f})")

    logger.debug("Config validation passed")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load config from YAML, merging with defaults.

    Args:
        config_path: Path to user YAML file. None → use defaults only.
        overrides:   Dict of values to apply on top of the merged config.

    Returns:
        Config object.
    """
    default_path = Path(__file__).parent / 'default_config.yaml'
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        cfg = deep_merge(cfg, user_cfg or {})
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.info("Using default config")

    if overrides:
        cfg = deep_merge(cfg, overrides)

    validate_config(cfg)
    return Config(cfg)


def save_config(config: Union[Config, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Save a Config (or plain dict) to YAML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = config.to_dict() if isinstance(config, Config) else config
    with open(output_path, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to {output_path}")
