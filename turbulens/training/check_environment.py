#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_lib.config import configuration_count, load_and_resolve_config, selection_group_count
from pipeline_lib.data import inspect_dataset

REQUIRED_IMPORTS = (
    "yaml",
    "numpy",
    "h5py",
    "matplotlib",
    "sklearn",
    "torch",
    "torchvision",
    "tqdm",
    "threadpoolctl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight validation for the local fractal-regression pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_multitask.yaml"),
        help="Pipeline YAML to resolve and validate.",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Validate software/configuration but do not require dataset directories to exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    failures: list[str] = []

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or newer is required.")

    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            print(f"OK import {name:<13}: {getattr(module, '__version__', 'n/a')}")
        except Exception as exc:  # pragma: no cover - environment dependent
            failures.append(f"import {name}: {exc}")
            print(f"FAIL import {name:<11}: {exc}")

    config = None
    try:
        config = load_and_resolve_config(args.config)
        grid_runs = configuration_count(config)
        selection = config["search"]["selection"]
        groups = selection_group_count(config, selection)
        selected_candidates = groups * int(selection["top_per_group"])
        candidate_runs = (
            selected_candidates
            * int(config["candidate_training"]["repetitions_per_configuration"])
        )
        print("\nConfiguration     : OK")
        print(f"Source            : {Path(args.config).resolve()}")
        print(f"Run root          : {config['paths']['run_root']}")
        print(f"Task              : {config['task']['mode']} {config['task']['targets']}")
        print(f"Grid runs         : {grid_runs}")
        print(f"Grid candidates   : {selected_candidates}")
        print(f"Candidate runs    : {candidate_runs}")
        print(f"Ensemble members  : {config['ensemble']['members']}")
        print(f"Learning rates    : {config['search']['space']['learning_rate']}")
        print(f"Batch sizes       : {config['search']['space']['batch_size']}")
        print(f"Pretrained        : {config['model']['pretrained']}")
        print(f"Weight recipes    : {config['model']['pretrained_weights']}")
        print(f"Config hash       : {config['_meta']['resolved_config_hash']}")
    except Exception as exc:
        failures.append(f"config: {exc}")
        print(f"\nFAIL configuration: {exc}")

    if config is not None and not args.skip_data_check:
        print("\nDataset validation")
        for key in ("small_root", "large_root"):
            path = Path(config["data"][key])
            if not path.is_dir():
                print(f"MISSING {key:<10}: {path}")
                failures.append(f"missing {key}: {path}")
                continue
            try:
                _, infos, manifest = inspect_dataset(
                    path,
                    config["data"]["split_directories"],
                    config["task"]["targets"],
                    config["task"]["target_ranges"],
                    augmentation=bool(config["training"]["augmentation"]),
                    require_square_for_augmentation=bool(
                        config["data"]["require_square_for_augmentation"]
                    ),
                    enforce_disjoint_files=bool(config["data"]["enforce_disjoint_files"]),
                )
                counts = {
                    split: sum(item.n_samples for item in split_infos)
                    for split, split_infos in infos.items()
                }
                file_counts = {split: len(split_infos) for split, split_infos in infos.items()}
                print(f"OK      {key:<10}: {path}")
                print(
                    f"        files={file_counts} samples={counts} "
                    f"shape={manifest['image_shape']} channels={manifest['stored_channels']}"
                )
                print(f"        fingerprint={manifest['dataset_fingerprint']}")
            except Exception as exc:
                failures.append(f"dataset {key}: {exc}")
                print(f"FAIL    {key:<10}: {exc}")

    smi = shutil.which("nvidia-smi")
    print("\nGPU visibility")
    if smi:
        result = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,memory.total,compute_mode",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        print(f"nvidia-smi        : {'OK' if result.returncode == 0 else 'FAILED'}")
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode != 0 and config is not None and config["runtime"]["device"] == "cuda":
            failures.append("nvidia-smi failed while runtime.device=cuda.")
    else:
        print("nvidia-smi        : not found")
        if config is not None and config["runtime"]["device"] == "cuda":
            failures.append("nvidia-smi not found while runtime.device=cuda.")

    if failures:
        print("\nPREFLIGHT FAILED")
        for item in failures:
            print(f" - {item}")
        return 2

    print("\nPREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
