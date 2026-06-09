# scripts/hpc_ensemble/config.py
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def output_dir(cfg: dict, member_id: int = None) -> Path:
    base = Path(cfg['hpc']['project_dir']) / cfg['output']['dir']
    if member_id is not None:
        return base / f'member_{member_id}'
    return base
