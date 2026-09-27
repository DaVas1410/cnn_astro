#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

TARGET_RANGES: dict[str, list[float]] = {
    "k_min": [1.0, 32.0],
    "k_max": [34.0, 64.0],
    "sigma": [0.0, 5.0],
    "beta": [-3.0, -1.0],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate thin single-target YAML files that inherit one base configuration."
    )
    repository = Path(__file__).resolve().parent
    parser.add_argument("--base-config", type=Path, default=repository / "configs" / "config_base.yaml")
    parser.add_argument("--output-dir", type=Path, default=repository / "configs")
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGET_RANGES), default=list(TARGET_RANGES))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def payload(base_reference: str, target: str) -> dict[str, Any]:
    return {
        "extends": base_reference,
        "experiment": {"name": f"single_{target}"},
        "task": {
            "mode": "single_target",
            "targets": [target],
            "target_ranges": {target: TARGET_RANGES[target]},
        },
        "search": {"selection": {"target": target}},
        "candidate_training": {"selection": {"target": target}},
        "interpretability": {"enabled": False},
        "stages": {
            "enabled": [
                "validate_data", "grid_search", "select_candidates",
                "retrain_candidates", "select_final", "train_ensemble",
                "evaluate_ensemble", "report",
            ]
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    base = args.base_config.expanduser().resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        base_reference = str(base.relative_to(output))
    except ValueError:
        base_reference = str(base)
    for target in args.targets:
        destination = output / f"config_single_{target}.yaml"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite.")
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload(base_reference, target), handle, sort_keys=False)
        print(destination)


if __name__ == "__main__":
    main()
