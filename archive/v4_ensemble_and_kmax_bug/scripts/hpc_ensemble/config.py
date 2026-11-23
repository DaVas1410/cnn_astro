# scripts/hpc_ensemble/config.py
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def output_dir(cfg: dict, member_id: int = None) -> Path:
    out = Path(cfg['output']['dir'])
    if not out.is_absolute():
        out = Path(cfg['hpc']['project_dir']) / out
    if member_id is not None:
        return out / f'member_{member_id}'
    return out
