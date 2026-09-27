#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Interpret the latent representations learned by a final regression ensemble.

The script loads every completed member created by ``run_final_ensemble.sh``,
extracts embeddings from the trained network, and performs:

1. PCA of each member embedding.
2. Correlation of embedding PCs with true and predicted physical parameters.
3. Canonical Correlation Analysis (CCA) between embeddings and true physical
   parameters.
4. Optional UMAP visualization colored by k_min, k_max, sigma, and beta.
5. A consensus analysis based on concatenated, member-wise standardized
   embeddings.

Raw embeddings from independently trained networks are NOT averaged. Their
coordinate systems can differ by rotations, sign changes, and permutations.
The consensus representation instead standardizes each member embedding using
validation data and concatenates the standardized blocks before PCA, CCA, and
UMAP.

PCA, CCA preprocessing, and UMAP are fitted on the validation split by default.
The test split is transformed afterward and used for held-out interpretation.
This avoids using the test set to determine the latent axes.

Typical command
---------------
python -u interpret_final_ensemble.py \
    --project-root /home/hernan.morales/projects/Fractal_Images \
    --ensemble-root /home/hernan.morales/projects/Fractal_Images/results/joint_regression_final_ensemble \
    --experiments-root /home/hernan.morales/projects/Fractal_Images/results/joint_regression_final_ensemble/experiments \
    --data-root /home/hernan.morales/projects/Fractal_Images/data/data_large \
    --training-script /home/hernan.morales/projects/Fractal_Images/code/joint_regression_multifile_ensemble.py \
    --output-dir /home/hernan.morales/projects/Fractal_Images/results/joint_regression_final_ensemble/interpretability \
    --expected-members 5 \
    --splits validation test \
    --fit-split validation \
    --plot-split test \
    --embedding-layer shared \
    --batch-size 256 \
    --num-workers 18 \
    --prefetch-factor 2 \
    --device auto \
    --amp \
    --pca-components 20 \
    --pc-correlation-components 10 \
    --cca-components 4 \
    --cca-pca-components 50 \
    --cca-pca-variance 0.95 \
    --umap-policy optional \
    --umap-neighbors 30 \
    --umap-min-dist 0.1 \
    --umap-fit-max-samples 20000 \
    --umap-transform-batch-size 10000 \
    --umap-representations consensus \
    --max-plot-points 30000 \
    --color-source both \
    --consensus \
    --save-embeddings \
    --overwrite

Dependencies
------------
Required: numpy, torch, torchvision, h5py, matplotlib, scikit-learn.
Optional: umap-learn. With ``--umap-policy optional``, UMAP is skipped if the
package is unavailable. Use ``--umap-policy require`` to make it mandatory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass

# Limit BLAS/OpenMP threads BEFORE importing NumPy/SciPy/scikit-learn/Torch.
# On the target server OpenBLAS and scikit-learn otherwise initialize with
# ~32 threads each, which can cause severe oversubscription and make PCA/CCA
# appear to hang. Users may override the default with INTERPRET_LINALG_THREADS.
_PREIMPORT_LINALG_THREADS = str(max(1, int(os.environ.get("INTERPRET_LINALG_THREADS", "4"))))
for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = _PREIMPORT_LINALG_THREADS
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import IncrementalPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

try:
    from threadpoolctl import threadpool_limits
except Exception:
    @contextlib.contextmanager
    def threadpool_limits(limits=None):
        yield


TARGET_NAMES: Tuple[str, ...] = ("k_min", "k_max", "sigma", "beta")
DEFAULT_PROJECT_ROOT = Path("/home/hernan.morales/projects/Fractal_Images")
DEFAULT_ENSEMBLE_NAME = "joint_regression_final_ensemble"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract final-ensemble embeddings and analyze them with PCA, "
            "PC-parameter correlations, CCA, and UMAP."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Fractal_Images project root.",
    )
    paths.add_argument(
        "--ensemble-root",
        type=Path,
        default=None,
        help=(
            "Root created by run_final_ensemble.sh. Default: "
            "<project-root>/results/joint_regression_final_ensemble."
        ),
    )
    paths.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Directory containing completed ensemble-member experiment folders.",
    )
    paths.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Dataset root containing train/, val/, and test/. Default: "
            "<project-root>/data/data_large."
        ),
    )
    paths.add_argument(
        "--training-script",
        type=Path,
        default=None,
        help=(
            "The exact joint_regression_multifile_ensemble.py used for training. "
            "It is imported to reuse the model and dataset definitions."
        ),
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Interpretability output directory. Default: <ensemble-root>/interpretability.",
    )

    discovery = parser.add_argument_group("ensemble discovery")
    discovery.add_argument(
        "--expected-members",
        type=int,
        default=5,
        help="Expected number of completed ensemble members.",
    )
    discovery.add_argument(
        "--allow-partial",
        action="store_true",
        help="Analyze available completed members when fewer than expected exist.",
    )

    extraction = parser.add_argument_group("embedding extraction")
    extraction.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=["validation", "test"],
        help="Dataset splits from which embeddings are extracted.",
    )
    extraction.add_argument(
        "--fit-split",
        choices=("validation", "test"),
        default="validation",
        help="Split used to fit PCA, CCA preprocessing, and UMAP.",
    )
    extraction.add_argument(
        "--plot-split",
        choices=("validation", "test"),
        default="test",
        help="Split used for the principal interpretability plots and summaries.",
    )
    extraction.add_argument(
        "--embedding-layer",
        choices=("shared", "backbone"),
        default="shared",
        help=(
            "Embedding to analyze. 'shared' is the 256-dimensional joint feature "
            "feeding the four regression heads. 'backbone' is the pooled ResNet feature."
        ),
    )
    extraction.add_argument("--batch-size", type=int, default=256)
    extraction.add_argument("--num-workers", type=int, default=18)
    extraction.add_argument("--prefetch-factor", type=int, default=2)
    extraction.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or cuda:N.",
    )
    extraction.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA mixed precision during embedding extraction.",
    )
    extraction.add_argument(
        "--max-analysis-samples",
        type=int,
        default=50000,
        help=(
            "Maximum samples retained per split for interpretability. The memory-safe "
            "default is 50000. Set to 0 only when the full split comfortably fits in RAM."
        ),
    )
    extraction.add_argument(
        "--sample-seed",
        type=int,
        default=2026,
        help="Seed for optional sample limiting and plot subsampling.",
    )
    extraction.add_argument(
        "--save-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-member embeddings and predictions as compressed NPZ files.",
    )

    pca_group = parser.add_argument_group("PCA and correlation")
    pca_group.add_argument(
        "--pca-components",
        type=int,
        default=20,
        help="Number of PCA coordinates saved for each representation.",
    )
    pca_group.add_argument(
        "--pc-correlation-components",
        type=int,
        default=10,
        help="Number of PCs correlated with physical parameters.",
    )
    pca_group.add_argument(
        "--pca-fit-max-samples",
        type=int,
        default=10000,
        help=(
            "Maximum validation samples used to fit each PCA model. All retained "
            "validation/test samples are transformed afterward. Zero means all."
        ),
    )
    pca_group.add_argument(
        "--pca-backend",
        choices=("auto", "torch", "incremental"),
        default="incremental",
        help=(
            "PCA implementation. incremental is the server-safe default and uses "
            "bounded CPU batches. auto/torch remain available for optional testing."
        ),
    )
    pca_group.add_argument(
        "--pca-device",
        default="cpu",
        help="Device used for torch PCA: auto, cpu, cuda, or cuda:N.",
    )
    pca_group.add_argument("--pca-oversamples", type=int, default=10)
    pca_group.add_argument("--pca-power-iterations", type=int, default=3)
    pca_group.add_argument("--pca-cpu-batch-size", type=int, default=4096)
    pca_group.add_argument("--pca-transform-batch-size", type=int, default=10000)
    pca_group.add_argument(
        "--linear-algebra-threads",
        type=int,
        default=4,
        help="Maximum BLAS/OpenMP threads used by CPU PCA/CCA linear algebra.",
    )

    pca_group.add_argument(
        "--pca-standardize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Standardize embedding dimensions before per-member PCA. Centering only "
            "is the default so PCA describes embedding variance itself."
        ),
    )

    cca_group = parser.add_argument_group("CCA")
    cca_group.add_argument("--cca-components", type=int, default=4)
    cca_group.add_argument(
        "--cca-fit-max-samples",
        type=int,
        default=5000,
        help=(
            "Maximum validation samples used to fit CCA and its PCA preprocessing. "
            "Held-out test scores are still calculated on all retained test samples. "
            "Zero means all."
        ),
    )
    cca_group.add_argument(
        "--cca-pca-components",
        type=int,
        default=50,
        help="Maximum standardized embedding PCs supplied to CCA.",
    )
    cca_group.add_argument(
        "--cca-pca-variance",
        type=float,
        default=0.95,
        help="Target cumulative variance retained before CCA, subject to the component cap.",
    )
    cca_group.add_argument("--cca-max-iter", type=int, default=1000)
    cca_group.add_argument("--cca-tol", type=float, default=1e-5)

    umap_group = parser.add_argument_group("UMAP")
    umap_group.add_argument(
        "--umap-policy",
        choices=("require", "optional", "skip"),
        default="optional",
        help="Require UMAP, run it when available, or skip it.",
    )
    umap_group.add_argument("--umap-neighbors", type=int, default=30)
    umap_group.add_argument("--umap-min-dist", type=float, default=0.1)
    umap_group.add_argument("--umap-metric", type=str, default="euclidean")
    umap_group.add_argument(
        "--umap-pca-components",
        type=int,
        default=50,
        help="PCA pre-reduction dimension before UMAP.",
    )
    umap_group.add_argument(
        "--umap-fit-max-samples",
        type=int,
        default=10000,
        help="Maximum validation samples used to fit each UMAP model. Zero means all.",
    )
    umap_group.add_argument(
        "--umap-transform-batch-size",
        type=int,
        default=10000,
        help="Number of samples transformed per UMAP batch to limit peak memory.",
    )
    umap_group.add_argument(
        "--umap-representations",
        choices=("all", "consensus", "none"),
        default="consensus",
        help=(
            "Run UMAP for every member, only for the consensus representation, or not "
            "at all. Consensus-only is the default because repeated UMAP fits are slow."
        ),
    )

    output = parser.add_argument_group("plots and output")
    output.add_argument(
        "--color-source",
        choices=("true", "predicted", "both"),
        default="both",
        help="Color PCA and UMAP plots by true values, predicted values, or both.",
    )
    output.add_argument(
        "--max-plot-points",
        type=int,
        default=30000,
        help="Maximum observations displayed in each scatter plot.",
    )
    output.add_argument(
        "--consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Analyze a consensus representation formed by concatenating member-wise "
            "standardized embeddings."
        ),
    )
    output.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace a previous interpretability output directory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.expected_members < 1:
        raise ValueError("--expected-members must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.num_workers > 0 and args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive when workers are enabled.")
    if args.max_analysis_samples < 0:
        raise ValueError("--max-analysis-samples must be non-negative.")
    if args.pca_fit_max_samples < 0:
        raise ValueError("--pca-fit-max-samples must be non-negative.")
    if args.cca_fit_max_samples < 0:
        raise ValueError("--cca-fit-max-samples must be non-negative.")
    if args.umap_transform_batch_size < 1:
        raise ValueError("--umap-transform-batch-size must be positive.")
    if args.pca_components < 2:
        raise ValueError("--pca-components must be at least 2.")
    if args.pca_oversamples < 0:
        raise ValueError("--pca-oversamples must be non-negative.")
    if args.pca_power_iterations < 1:
        raise ValueError("--pca-power-iterations must be positive.")
    if args.pca_cpu_batch_size < 1 or args.pca_transform_batch_size < 1:
        raise ValueError("PCA batch sizes must be positive.")
    if args.linear_algebra_threads < 1:
        raise ValueError("--linear-algebra-threads must be positive.")
    if args.pc_correlation_components < 1:
        raise ValueError("--pc-correlation-components must be positive.")
    if args.cca_components < 1 or args.cca_components > len(TARGET_NAMES):
        raise ValueError("--cca-components must lie between 1 and 4.")
    if args.cca_pca_components < args.cca_components:
        raise ValueError("--cca-pca-components cannot be smaller than --cca-components.")
    if not 0.0 < args.cca_pca_variance <= 1.0:
        raise ValueError("--cca-pca-variance must lie in (0,1].")
    if args.cca_max_iter < 1 or args.cca_tol <= 0.0:
        raise ValueError("CCA iteration and tolerance settings must be positive.")
    if args.umap_neighbors < 2:
        raise ValueError("--umap-neighbors must be at least 2.")
    if not 0.0 <= args.umap_min_dist <= 1.0:
        raise ValueError("--umap-min-dist must lie in [0,1].")
    if args.umap_pca_components < 2:
        raise ValueError("--umap-pca-components must be at least 2.")
    if args.umap_fit_max_samples < 0:
        raise ValueError("--umap-fit-max-samples must be non-negative.")
    if args.max_plot_points < 1:
        raise ValueError("--max-plot-points must be positive.")
    args.splits = list(dict.fromkeys(args.splits))
    if args.fit_split not in args.splits:
        raise ValueError("--fit-split must be included in --splits.")
    if args.plot_split not in args.splits:
        raise ValueError("--plot-split must be included in --splits.")


def resolve_paths(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.expanduser().resolve()
    if args.ensemble_root is None:
        args.ensemble_root = args.project_root / "results" / DEFAULT_ENSEMBLE_NAME
    args.ensemble_root = args.ensemble_root.expanduser().resolve()
    if args.experiments_root is None:
        args.experiments_root = args.ensemble_root / "experiments"
    args.experiments_root = args.experiments_root.expanduser().resolve()
    if args.data_root is None:
        args.data_root = args.project_root / "data" / "data_large"
    args.data_root = args.data_root.expanduser().resolve()
    if args.training_script is None:
        args.training_script = args.project_root / "code" / "joint_regression_multifile_ensemble.py"
    args.training_script = args.training_script.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.ensemble_root / "interpretability"
    args.output_dir = args.output_dir.expanduser().resolve()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def atomic_csv_write(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: to_jsonable(value) for key, value in row.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def atomic_npz_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def completion_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return load_json(path).get("status") == "completed"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def parse_seed(configuration: Mapping[str, Any], experiment_name: str) -> int:
    value = configuration.get("seed")
    if value is not None:
        return int(value)
    match = re.search(r"seed([0-9]+)", experiment_name)
    if match:
        return int(match.group(1))
    raise KeyError(f"Could not determine the seed for {experiment_name}.")


def import_training_module(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Training script not found: {path}")
    module_name = "_joint_regression_ensemble_training"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    required = (
        "TARGET_NAMES",
        "TARGET_RANGES",
        "NormalizationStats",
        "JointHDF5Dataset",
        "JointResNetRegressor",
        "discover_split_files",
        "inspect_h5",
        "load_target_matrix",
        "normalize_targets",
        "denormalize_targets",
        "create_dataloader",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Training module is missing required objects: {missing}")
    if tuple(module.TARGET_NAMES) != TARGET_NAMES:
        raise ValueError(
            f"Training target order {tuple(module.TARGET_NAMES)} does not match {TARGET_NAMES}."
        )
    return module


def discover_members(experiments_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    if not experiments_root.is_dir():
        raise NotADirectoryError(f"Experiments directory not found: {experiments_root}")
    members: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for experiment_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        try:
            if not completion_is_valid(experiment_dir / "COMPLETED.json"):
                raise RuntimeError("no valid COMPLETED.json marker")
            config_path = experiment_dir / "configuration.json"
            checkpoint_path = experiment_dir / "checkpoints" / "best_checkpoint.pt"
            split_path = experiment_dir / "split_indices.npz"
            manifest_path = experiment_dir / "data_manifest.json"
            if not config_path.is_file():
                raise FileNotFoundError("missing configuration.json")
            if not checkpoint_path.is_file():
                raise FileNotFoundError("missing checkpoints/best_checkpoint.pt")
            if not split_path.is_file():
                raise FileNotFoundError("missing split_indices.npz")
            if not manifest_path.is_file():
                raise FileNotFoundError("missing data_manifest.json")
            configuration = load_json(config_path)
            members.append(
                {
                    "name": experiment_dir.name,
                    "experiment_dir": experiment_dir.resolve(),
                    "configuration": configuration,
                    "seed": parse_seed(configuration, experiment_dir.name),
                    "checkpoint": checkpoint_path.resolve(),
                    "split_file": split_path.resolve(),
                    "manifest": load_json(manifest_path),
                }
            )
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
            skipped.append({"directory": str(experiment_dir), "reason": str(exc)})
    members.sort(key=lambda item: (int(item["seed"]), str(item["name"])))
    seeds = [int(item["seed"]) for item in members]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate ensemble seeds found: {seeds}")
    return members, skipped


def canonical_configuration(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {
        "seed",
        "output_dir",
        "resume",
        "device",
        "overwrite",
        "cudnn_benchmark",
        "deterministic",
    }
    return {
        str(key): to_jsonable(value)
        for key, value in configuration.items()
        if key not in ignored
    }


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")
    required = {"model_state", "model_config", "normalization_stats", "target_ranges"}
    missing = sorted(required.difference(checkpoint.keys()))
    if missing:
        raise KeyError(f"Checkpoint {path} is missing keys: {missing}")
    return checkpoint


def verify_members(members: Sequence[MutableMapping[str, Any]]) -> Dict[str, Any]:
    if not members:
        raise ValueError("No completed ensemble members were found.")
    reference_config = canonical_configuration(members[0]["configuration"])
    reference_manifest = to_jsonable(members[0]["manifest"])
    reference_checkpoint = load_checkpoint(Path(members[0]["checkpoint"]))
    reference_model_config = to_jsonable(reference_checkpoint["model_config"])
    reference_normalization = to_jsonable(reference_checkpoint["normalization_stats"])
    reference_ranges = to_jsonable(reference_checkpoint["target_ranges"])
    members[0]["checkpoint_metadata"] = {
        "model_config": reference_model_config,
        "normalization_stats": reference_normalization,
        "target_ranges": reference_ranges,
    }

    mismatches: List[Dict[str, Any]] = []
    for member in members[1:]:
        checkpoint = load_checkpoint(Path(member["checkpoint"]))
        member_metadata = {
            "model_config": to_jsonable(checkpoint["model_config"]),
            "normalization_stats": to_jsonable(checkpoint["normalization_stats"]),
            "target_ranges": to_jsonable(checkpoint["target_ranges"]),
        }
        member["checkpoint_metadata"] = member_metadata
        differences: Dict[str, Any] = {}
        candidate_config = canonical_configuration(member["configuration"])
        if candidate_config != reference_config:
            differences["configuration"] = {
                "reference": reference_config,
                "candidate": candidate_config,
            }
        if to_jsonable(member["manifest"]) != reference_manifest:
            differences["data_manifest"] = "different"
        if member_metadata != members[0]["checkpoint_metadata"]:
            differences["checkpoint_metadata"] = {
                "reference": members[0]["checkpoint_metadata"],
                "candidate": member_metadata,
            }
        if differences:
            mismatches.append({"member": member["name"], "differences": differences})
    if mismatches:
        raise ValueError(
            "Ensemble members are not directly comparable. First mismatch: "
            f"{mismatches[0]}"
        )
    return {
        "identical_hyperparameters": True,
        "identical_data_manifest": True,
        "identical_model_configuration": True,
        "identical_normalization": True,
        "canonical_configuration": reference_config,
        "model_config": reference_model_config,
        "normalization_stats": reference_normalization,
        "target_ranges": reference_ranges,
    }


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        index = 0 if device.index is None else int(device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        raise ValueError("Only CPU and CUDA devices are supported.")
    return device


def prepare_output_dir(path: Path, overwrite: bool, experiments_root: Path) -> None:
    resolved = path.resolve()
    if resolved == experiments_root.resolve():
        raise ValueError("Output directory cannot equal the experiments directory.")
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {resolved}")
        shutil.rmtree(resolved)
    for subdir in (
        "embeddings",
        "coordinates",
        "metrics",
        "plots",
        "models",
    ):
        (resolved / subdir).mkdir(parents=True, exist_ok=True)


def load_split_indices(path: Path, split: str, expected_size: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if split not in data:
            raise KeyError(f"{path} does not contain split {split!r}.")
        indices = np.asarray(data[split], dtype=np.int64)
    if indices.ndim != 1 or indices.size != expected_size:
        raise ValueError(
            f"Split {split!r} in {path} has shape {indices.shape}; expected ({expected_size},)."
        )
    if not np.array_equal(indices, np.arange(expected_size, dtype=np.int64)):
        raise ValueError(
            f"Split indices for {split!r} are not the expected logical multifile order."
        )
    return indices


def deterministic_subset(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or indices.size <= maximum:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(indices.size, size=maximum, replace=False))
    return np.asarray(indices[positions], dtype=np.int64)


def build_datasets(
    module: Any,
    args: argparse.Namespace,
    member: Mapping[str, Any],
    consistency: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    split_files = module.discover_split_files(args.data_root)
    infos = {
        split: [module.inspect_h5(path) for path in paths]
        for split, paths in split_files.items()
    }
    config = member["configuration"]
    model_config = consistency["model_config"]
    normalization = module.NormalizationStats(**consistency["normalization_stats"])
    in_channels = int(model_config["in_channels"])
    input_standardization = str(config.get("input_standardization", "imagenet"))

    datasets: Dict[str, Any] = {}
    targets_physical: Dict[str, np.ndarray] = {}
    targets_normalized: Dict[str, np.ndarray] = {}
    retained_indices: Dict[str, np.ndarray] = {}
    shared_epoch = mp.Value("i", 0)

    for offset, split in enumerate(args.splits):
        paths = split_files[split]
        total = int(sum(info.n_samples for info in infos[split]))
        full_indices = load_split_indices(Path(member["split_file"]), split, total)
        selected = deterministic_subset(
            full_indices,
            args.max_analysis_samples,
            args.sample_seed + 1009 * (offset + 1),
        )
        physical_all = np.asarray(module.load_target_matrix(paths), dtype=np.float32)
        normalized_all = np.asarray(module.normalize_targets(physical_all), dtype=np.float32)
        retained_indices[split] = selected
        targets_physical[split] = np.asarray(physical_all[selected], dtype=np.float32)
        targets_normalized[split] = np.asarray(normalized_all[selected], dtype=np.float32)
        datasets[split] = module.JointHDF5Dataset(
            h5_paths=paths,
            file_infos=infos[split],
            indices=selected,
            labels_normalized_all=normalized_all,
            normalization=normalization,
            in_channels=in_channels,
            input_standardization=input_standardization,
            augment=False,
            augmentation_seed=0,
            shared_epoch=shared_epoch,
        )
    return datasets, targets_physical, targets_normalized, retained_indices


def build_model(module: Any, checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    config = checkpoint["model_config"]
    model = module.JointResNetRegressor(
        architecture=str(config["architecture"]),
        in_channels=int(config["in_channels"]),
        pretrained=False,
        dropout=float(config["dropout"]),
        preserve_resolution=bool(config["preserve_resolution"]),
        n_outputs=int(config.get("n_outputs", len(TARGET_NAMES))),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def extract_member_split(
    module: Any,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    embedding_layer: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embeddings: List[np.ndarray] = []
    predictions_raw: List[np.ndarray] = []
    predictions_clamped: List[np.ndarray] = []
    targets_seen: List[np.ndarray] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            autocast_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_context:
                backbone_features = model.backbone(images)
                shared_features = model.shared(backbone_features)
                raw = torch.cat(
                    [model.heads[target](shared_features) for target in TARGET_NAMES],
                    dim=1,
                )
            selected_embedding = shared_features if embedding_layer == "shared" else backbone_features
            embeddings.append(selected_embedding.detach().float().cpu().numpy())
            raw_np = raw.detach().float().cpu().numpy()
            predictions_raw.append(raw_np)
            predictions_clamped.append(np.clip(raw_np, 0.0, 1.0).astype(np.float32))
            targets_seen.append(targets.detach().float().cpu().numpy())
    embedding_array = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    raw_array = np.concatenate(predictions_raw, axis=0).astype(np.float32, copy=False)
    clamped_array = np.concatenate(predictions_clamped, axis=0).astype(np.float32, copy=False)
    target_array = np.concatenate(targets_seen, axis=0).astype(np.float32, copy=False)
    return embedding_array, raw_array, clamped_array, target_array



@dataclass
class RobustPCA:
    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    n_components_: int
    n_features_in_: int
    n_samples_fit_: int
    backend_: str
    device_: str
    fit_seconds_: float

    def transform(self, values: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"PCA transform expected (*,{self.n_features_in_}), got {tuple(array.shape)}"
            )
        batch_size = max(1, int(batch_size))
        output = np.empty((array.shape[0], self.n_components_), dtype=np.float32)
        mean = np.asarray(self.mean_, dtype=np.float32)
        components_t = np.asarray(self.components_.T, dtype=np.float32)
        for start in range(0, array.shape[0], batch_size):
            stop = min(array.shape[0], start + batch_size)
            output[start:stop] = (np.asarray(array[start:stop], dtype=np.float32) - mean) @ components_t
        return output


def resolve_pca_device(value: str, extraction_device: torch.device) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return extraction_device if extraction_device.type == "cuda" else torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda:0")
    return torch.device(value)


def fit_robust_pca(
    values: np.ndarray,
    n_components: int,
    args: argparse.Namespace,
    seed: int,
    fit_max_samples: int,
    label: str,
    extraction_device: torch.device,
) -> Tuple[RobustPCA, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
        raise ValueError(f"PCA requires a 2-D matrix with >=3 samples and >=2 features; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: PCA input contains NaN or infinite values.")
    fit_indices = choose_plot_indices(array.shape[0], fit_max_samples, seed + 104729) if fit_max_samples > 0 else np.arange(array.shape[0], dtype=np.int64)
    fit = np.asarray(array[fit_indices], dtype=np.float32, order="C")
    n_components = min(int(n_components), fit.shape[0] - 1, fit.shape[1])
    if n_components < 2:
        raise ValueError(f"{label}: fewer than 2 PCA components are available.")
    pca_device = resolve_pca_device(args.pca_device, extraction_device)
    backend = str(args.pca_backend)
    if backend == "auto":
        backend = "torch" if pca_device.type == "cuda" and torch.cuda.is_available() else "incremental"

    print(
        f"    [{label}] PCA fit subset {len(fit_indices)}/{len(array)}; "
        f"matrix={fit.shape[0]}x{fit.shape[1]}; backend={backend}; device={pca_device}",
        flush=True,
    )
    started = time.time()
    if backend == "torch":
        try:
            if pca_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for torch PCA")
            q = min(fit.shape[1], n_components + max(0, int(args.pca_oversamples)))
            if pca_device.type == "cuda":
                free_bytes, total_bytes = torch.cuda.mem_get_info(pca_device)
                print(
                    f"    [{label}] CUDA PCA memory free={free_bytes / 2**30:.2f} GiB, "
                    f"matrix={fit.nbytes / 2**30:.2f} GiB",
                    flush=True,
                )
            with torch.random.fork_rng(devices=[pca_device] if pca_device.type == "cuda" else []):
                torch.manual_seed(int(seed))
                if pca_device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                tensor = torch.from_numpy(fit).to(pca_device, dtype=torch.float32)
                mean_t = tensor.mean(dim=0)
                centered = tensor - mean_t
                _, singular_values, vectors = torch.pca_lowrank(
                    centered,
                    q=q,
                    center=False,
                    niter=max(1, int(args.pca_power_iterations)),
                )
                vectors = vectors[:, :n_components]
                singular_values = singular_values[:n_components]
                explained = singular_values.square() / max(1, fit.shape[0] - 1)
                total_variance = centered.var(dim=0, unbiased=True).sum()
                ratio = explained / torch.clamp(total_variance, min=torch.finfo(torch.float32).eps)
                model = RobustPCA(
                    mean_=mean_t.detach().cpu().numpy().astype(np.float32),
                    components_=vectors.T.detach().cpu().numpy().astype(np.float32),
                    explained_variance_=explained.detach().cpu().numpy().astype(np.float32),
                    explained_variance_ratio_=ratio.detach().cpu().numpy().astype(np.float32),
                    singular_values_=singular_values.detach().cpu().numpy().astype(np.float32),
                    n_components_=n_components,
                    n_features_in_=fit.shape[1],
                    n_samples_fit_=fit.shape[0],
                    backend_="torch_lowrank",
                    device_=str(pca_device),
                    fit_seconds_=float(time.time() - started),
                )
                del tensor, centered, mean_t, vectors, singular_values, explained, total_variance, ratio
            if pca_device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"    [{label}] torch PCA finished in {model.fit_seconds_:.2f} s", flush=True)
            return model, fit_indices
        except RuntimeError as exc:
            if args.pca_backend != "auto":
                raise
            print(
                f"    [{label}] torch PCA failed: {type(exc).__name__}: {exc}; "
                "falling back to IncrementalPCA",
                flush=True,
            )
            if pca_device.type == "cuda":
                torch.cuda.empty_cache()
            backend = "incremental"

    if backend != "incremental":
        raise ValueError(f"Unsupported PCA backend: {backend}")
    batch_size = max(int(args.pca_cpu_batch_size), 4 * n_components, n_components + 1)
    n_batches = max(1, fit.shape[0] // batch_size)
    n_batches = min(n_batches, max(1, fit.shape[0] // n_components))
    chunks = np.array_split(np.arange(fit.shape[0], dtype=np.int64), n_batches)
    incremental = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    with threadpool_limits(limits=int(args.linear_algebra_threads)):
        for batch_index, idx in enumerate(chunks, start=1):
            incremental.partial_fit(np.asarray(fit[idx], dtype=np.float32))
            print(
                f"    [{label}] IncrementalPCA batch {batch_index}/{len(chunks)} complete "
                f"({len(idx)} samples)",
                flush=True,
            )
    model = RobustPCA(
        mean_=np.asarray(incremental.mean_, dtype=np.float32),
        components_=np.asarray(incremental.components_, dtype=np.float32),
        explained_variance_=np.asarray(incremental.explained_variance_, dtype=np.float32),
        explained_variance_ratio_=np.asarray(incremental.explained_variance_ratio_, dtype=np.float32),
        singular_values_=np.asarray(incremental.singular_values_, dtype=np.float32),
        n_components_=n_components,
        n_features_in_=fit.shape[1],
        n_samples_fit_=fit.shape[0],
        backend_="incremental_cpu",
        device_="cpu",
        fit_seconds_=float(time.time() - started),
    )
    print(f"    [{label}] incremental PCA finished in {model.fit_seconds_:.2f} s", flush=True)
    return model, fit_indices

def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return safe_pearson(average_ranks(a[mask]), average_ranks(b[mask]))


def choose_plot_indices(n_samples: int, maximum: int, seed: int) -> np.ndarray:
    if n_samples <= maximum:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=maximum, replace=False)).astype(np.int64)


def fit_projection_pca(
    fit_embedding: np.ndarray,
    requested_components: int,
    standardize: bool,
    seed: int,
    fit_max_samples: int,
    args: argparse.Namespace,
    label: str,
    extraction_device: torch.device,
) -> Tuple[Optional[StandardScaler], RobustPCA, np.ndarray]:
    x_all = np.asarray(fit_embedding, dtype=np.float32)
    scaler: Optional[StandardScaler]
    if standardize:
        # Fit the scaler on the same deterministic PCA fit subset to prevent a full
        # float64 copy of all embeddings.
        scale_indices = choose_plot_indices(x_all.shape[0], fit_max_samples, seed + 104729) if fit_max_samples > 0 else np.arange(x_all.shape[0], dtype=np.int64)
        scaler = StandardScaler(copy=True).fit(np.asarray(x_all[scale_indices], dtype=np.float32))
        scaled = np.asarray(scaler.transform(x_all), dtype=np.float32)
    else:
        scaler = None
        scaled = x_all
    pca, fit_indices = fit_robust_pca(
        scaled,
        requested_components,
        args=args,
        seed=seed,
        fit_max_samples=fit_max_samples,
        label=label,
        extraction_device=extraction_device,
    )
    return scaler, pca, fit_indices


def transform_projection_pca(
    embedding: np.ndarray,
    scaler: Optional[StandardScaler],
    pca: RobustPCA,
    batch_size: int,
) -> np.ndarray:
    x = np.asarray(embedding, dtype=np.float32)
    if scaler is not None:
        x = np.asarray(scaler.transform(x), dtype=np.float32)
    return pca.transform(x, batch_size=batch_size)

def plot_embedding_colored(
    coordinates: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    path: Path,
    maximum_points: int,
    seed: int,
) -> None:
    indices = choose_plot_indices(coordinates.shape[0], maximum_points, seed)
    xy = coordinates[indices]
    color = np.asarray(values)[indices]
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color,
        s=8,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_explained_variance(pca: Any, title: str, path: Path) -> None:
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    x = np.arange(1, ratios.size + 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.bar(x, ratios * 100.0, label="Individual")
    ax.plot(x, np.cumsum(ratios) * 100.0, marker="o", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(title)
    ax.set_ylim(0.0, 105.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    path: Path,
) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    fig_width = max(7.0, 1.0 + 1.15 * len(column_labels))
    fig_height = max(5.0, 1.5 + 0.48 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if abs(value) < 0.65 else "white",
                )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_cca_model(
    fit_embedding: np.ndarray,
    fit_parameters: np.ndarray,
    args: argparse.Namespace,
    label: str,
    extraction_device: torch.device,
) -> Dict[str, Any]:
    x_all = np.asarray(fit_embedding, dtype=np.float32)
    y_all = np.asarray(fit_parameters, dtype=np.float32)
    if x_all.shape[0] != y_all.shape[0]:
        raise ValueError("CCA embedding and parameter sample counts differ.")
    fit_indices = choose_plot_indices(
        x_all.shape[0], args.cca_fit_max_samples, args.sample_seed + 4201
    ) if args.cca_fit_max_samples > 0 else np.arange(x_all.shape[0], dtype=np.int64)
    x_fit = np.asarray(x_all[fit_indices], dtype=np.float32)
    y_fit = np.asarray(y_all[fit_indices], dtype=np.float32)
    x_scaler = StandardScaler(copy=True).fit(x_fit)
    y_scaler = StandardScaler(copy=True).fit(y_fit)
    x_scaled = np.asarray(x_scaler.transform(x_fit), dtype=np.float32)
    y_scaled = np.asarray(y_scaler.transform(y_fit), dtype=np.float32)
    maximum_pca = min(args.cca_pca_components, x_scaled.shape[0] - 1, x_scaled.shape[1])
    if maximum_pca < args.cca_components:
        raise ValueError(f"Only {maximum_pca} embedding dimensions are available before CCA.")
    print(f"    [{label}] CCA preprocessing PCA...", flush=True)
    pca, _ = fit_robust_pca(
        x_scaled,
        maximum_pca,
        args=args,
        seed=args.sample_seed + 4201,
        fit_max_samples=0,
        label=f"{label}/CCA-PCA",
        extraction_device=extraction_device,
    )
    x_pca_full = pca.transform(x_scaled, batch_size=args.pca_transform_batch_size)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    variance_count = int(np.searchsorted(cumulative, args.cca_pca_variance, side="left") + 1)
    retained = max(args.cca_components, min(maximum_pca, variance_count))
    x_pca = x_pca_full[:, :retained]
    n_components = min(args.cca_components, retained, y_scaled.shape[1])
    cca = CCA(n_components=n_components, scale=False, max_iter=args.cca_max_iter, tol=args.cca_tol)
    caught: List[str] = []
    print(
        f"    [{label}] CCA solver fit: samples={len(x_pca)}, features={retained}, "
        f"outputs={y_scaled.shape[1]}, max_iter={args.cca_max_iter}, tol={args.cca_tol}",
        flush=True,
    )
    with threadpool_limits(limits=int(args.linear_algebra_threads)):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", category=ConvergenceWarning)
            cca.fit(x_pca, y_scaled)
            caught = [str(record.message) for record in records]
    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "pca": pca,
        "retained_pca_components": retained,
        "cca": cca,
        "n_components": n_components,
        "warnings": caught,
        "fit_samples_total": int(x_all.shape[0]),
        "fit_samples_used": int(fit_indices.size),
        "pca_backend": pca.backend_,
        "pca_device": pca.device_,
        "pca_fit_seconds": pca.fit_seconds_,
    }


def transform_cca(
    model: Mapping[str, Any],
    embedding: np.ndarray,
    parameters: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_scaled = np.asarray(model["x_scaler"].transform(np.asarray(embedding, dtype=np.float32)), dtype=np.float32)
    x_pca = model["pca"].transform(x_scaled, batch_size=batch_size)[:, : int(model["retained_pca_components"])]
    y_scaled = np.asarray(model["y_scaler"].transform(np.asarray(parameters, dtype=np.float32)), dtype=np.float32)
    x_scores, y_scores = model["cca"].transform(x_pca, y_scaled)
    return np.asarray(x_scores, dtype=np.float32), np.asarray(y_scores, dtype=np.float32)

def plot_cca_correlations(
    split_correlations: Mapping[str, Sequence[float]],
    title: str,
    path: Path,
) -> None:
    splits = list(split_correlations)
    n_components = max(len(split_correlations[split]) for split in splits)
    x = np.arange(1, n_components + 1)
    width = 0.8 / max(1, len(splits))
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for index, split in enumerate(splits):
        values = np.asarray(split_correlations[split], dtype=np.float64)
        offset = (index - 0.5 * (len(splits) - 1)) * width
        ax.bar(x[: values.size] + offset, values, width=width, label=split)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("corr(embedding variate, parameter variate)")
    ax.set_title(title)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cca_pair(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    component: int,
    title: str,
    path: Path,
    maximum_points: int,
    seed: int,
) -> None:
    indices = choose_plot_indices(x_scores.shape[0], maximum_points, seed)
    x = x_scores[indices, component]
    y = y_scores[indices, component]
    correlation = safe_pearson(x, y)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(x, y, s=8, alpha=0.65, linewidths=0, rasterized=True)
    ax.set_xlabel(f"Embedding canonical variate U{component + 1}")
    ax.set_ylabel(f"Physical-parameter canonical variate V{component + 1}")
    ax.set_title(f"{title}\nPearson r = {correlation:.3f}")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def import_umap(policy: str) -> Tuple[Optional[Any], Optional[str]]:
    if policy == "skip":
        return None, "UMAP disabled by --umap-policy skip."
    try:
        import umap  # type: ignore

        return umap, None
    except ImportError as exc:
        message = (
            "umap-learn is not installed. Install it in the active environment with "
            "'python -m pip install umap-learn'."
        )
        if policy == "require":
            raise ImportError(message) from exc
        return None, message


def fit_umap_coordinates(
    umap_module: Any,
    embeddings: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    label: str,
    extraction_device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    fit_embedding = np.asarray(embeddings[args.fit_split], dtype=np.float32)
    fit_indices = choose_plot_indices(
        fit_embedding.shape[0], args.umap_fit_max_samples, args.sample_seed + 7103
    ) if args.umap_fit_max_samples > 0 else np.arange(fit_embedding.shape[0], dtype=np.int64)
    scaler = StandardScaler(copy=True).fit(np.asarray(fit_embedding[fit_indices], dtype=np.float32))
    fit_scaled = np.asarray(scaler.transform(np.asarray(fit_embedding[fit_indices], dtype=np.float32)), dtype=np.float32)
    maximum_pca = min(args.umap_pca_components, fit_scaled.shape[0] - 1, fit_scaled.shape[1])
    pca: Optional[RobustPCA]
    if maximum_pca >= 2 and maximum_pca < fit_scaled.shape[1]:
        print(f"    [{label}] UMAP preprocessing PCA...", flush=True)
        pca, _ = fit_robust_pca(
            fit_scaled,
            maximum_pca,
            args=args,
            seed=args.sample_seed + 7103,
            fit_max_samples=0,
            label=f"{label}/UMAP-PCA",
            extraction_device=extraction_device,
        )
        fit_reduced = pca.transform(fit_scaled, batch_size=args.pca_transform_batch_size)
    else:
        pca = None
        fit_reduced = fit_scaled

    n_neighbors = min(args.umap_neighbors, max(2, fit_reduced.shape[0] - 1))
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.sample_seed,
        transform_seed=args.sample_seed,
        low_memory=True,
        n_jobs=1,
    )
    reducer.fit(fit_reduced)

    coordinates: Dict[str, np.ndarray] = {}
    for split, embedding in embeddings.items():
        chunks: List[np.ndarray] = []
        for start in range(0, embedding.shape[0], args.umap_transform_batch_size):
            stop = min(embedding.shape[0], start + args.umap_transform_batch_size)
            x = np.asarray(scaler.transform(np.asarray(embedding[start:stop], dtype=np.float32)), dtype=np.float32)
            if pca is not None:
                x = pca.transform(x, batch_size=args.pca_transform_batch_size)
            chunks.append(np.asarray(reducer.transform(x), dtype=np.float32))
            print(
                f"    [{label}] UMAP transform {split}: {stop}/{embedding.shape[0]}",
                flush=True,
            )
        coordinates[split] = np.concatenate(chunks, axis=0)
    metadata = {
        "fit_split": args.fit_split,
        "fit_samples_total": int(fit_embedding.shape[0]),
        "fit_samples_used": int(fit_indices.size),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(args.umap_min_dist),
        "metric": args.umap_metric,
        "transform_batch_size": int(args.umap_transform_batch_size),
        "pca_pre_reduction_components": None if pca is None else int(maximum_pca),
        "pca_backend": None if pca is None else pca.backend_,
    }
    return coordinates, metadata

def analyze_representation(
    name: str,
    display_name: str,
    embeddings: Mapping[str, np.ndarray],
    predictions_physical: Mapping[str, np.ndarray],
    targets_physical: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    output_dir: Path,
    umap_module: Optional[Any],
    prestandardized: bool,
) -> Dict[str, Any]:
    representation_dir = output_dir / "plots" / name
    representation_dir.mkdir(parents=True, exist_ok=True)
    coordinate_dir = output_dir / "coordinates" / name
    coordinate_dir.mkdir(parents=True, exist_ok=True)

    representation_start = time.time()
    fit_shape = embeddings[args.fit_split].shape
    estimated_mib = float(np.prod(fit_shape) * 4 / (1024 ** 2))
    print(
        f"  [{name}] samples={fit_shape[0]}, dimensions={fit_shape[1]}, "
        f"embedding_matrix={estimated_mib:.1f} MiB",
        flush=True,
    )
    maximum_requested = max(args.pca_components, args.pc_correlation_components, 2)
    stage_start = time.time()
    print(
        f"  [{name}] PCA: fitting on at most {args.pca_fit_max_samples or 'all'} "
        f"{args.fit_split} samples...",
        flush=True,
    )
    pca_scaler, pca_model, pca_fit_indices = fit_projection_pca(
        embeddings[args.fit_split],
        requested_components=maximum_requested,
        standardize=(False if prestandardized else args.pca_standardize),
        seed=args.sample_seed,
        fit_max_samples=args.pca_fit_max_samples,
        args=args,
        label=name,
        extraction_device=resolve_device(args.device),
    )
    pca_scores: Dict[str, np.ndarray] = {}
    for split, embedding in embeddings.items():
        pca_scores[split] = transform_projection_pca(embedding, pca_scaler, pca_model, args.pca_transform_batch_size)
        atomic_npz_save(
            coordinate_dir / f"{split}_pca_coordinates.npz",
            pca_coordinates=pca_scores[split],
            target_physical=np.asarray(targets_physical[split], dtype=np.float32),
            prediction_physical=np.asarray(predictions_physical[split], dtype=np.float32),
            target_names=np.asarray(TARGET_NAMES),
            explained_variance_ratio=np.asarray(pca_model.explained_variance_ratio_, dtype=np.float32),
        )
    print(
        f"  [{name}] PCA completed in {time.time() - stage_start:.1f} s "
        f"using {pca_fit_indices.size} fit samples.",
        flush=True,
    )

    explained_rows: List[Dict[str, Any]] = []
    cumulative = np.cumsum(pca_model.explained_variance_ratio_)
    for index, (ratio, cumulative_ratio) in enumerate(
        zip(pca_model.explained_variance_ratio_, cumulative), start=1
    ):
        explained_rows.append(
            {
                "representation": name,
                "display_name": display_name,
                "pc": index,
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance_ratio": float(cumulative_ratio),
            }
        )
    atomic_csv_write(explained_rows, output_dir / "metrics" / f"{name}_pca_explained_variance.csv")
    plot_explained_variance(
        pca_model,
        title=f"{display_name}: PCA explained variance",
        path=representation_dir / "pca_explained_variance.png",
    )

    correlation_rows: List[Dict[str, Any]] = []
    n_corr = min(args.pc_correlation_components, pca_scores[args.fit_split].shape[1])
    for split in args.splits:
        for source, matrix in (
            ("true", targets_physical[split]),
            ("predicted", predictions_physical[split]),
        ):
            pearson_matrix = np.empty((n_corr, len(TARGET_NAMES)), dtype=np.float64)
            for pc_index in range(n_corr):
                for target_index, target in enumerate(TARGET_NAMES):
                    pearson = safe_pearson(
                        pca_scores[split][:, pc_index],
                        matrix[:, target_index],
                    )
                    spearman = safe_spearman(
                        pca_scores[split][:, pc_index],
                        matrix[:, target_index],
                    )
                    pearson_matrix[pc_index, target_index] = pearson
                    correlation_rows.append(
                        {
                            "representation": name,
                            "display_name": display_name,
                            "split": split,
                            "parameter_source": source,
                            "pc": pc_index + 1,
                            "parameter": target,
                            "pearson_r": pearson,
                            "spearman_rho": spearman,
                            "explained_variance_ratio": float(
                                pca_model.explained_variance_ratio_[pc_index]
                            ),
                        }
                    )
            plot_correlation_heatmap(
                pearson_matrix,
                row_labels=[f"PC{i + 1}" for i in range(n_corr)],
                column_labels=list(TARGET_NAMES),
                title=f"{display_name}: {split} PC correlations with {source} parameters",
                path=representation_dir / f"{split}_pc_parameter_correlations_{source}.png",
            )
    atomic_csv_write(
        correlation_rows,
        output_dir / "metrics" / f"{name}_pc_parameter_correlations.csv",
    )

    plot_split = args.plot_split
    sources: List[str]
    if args.color_source == "both":
        sources = ["true", "predicted"]
    else:
        sources = [args.color_source]
    pc1_variance = 100.0 * float(pca_model.explained_variance_ratio_[0])
    pc2_variance = 100.0 * float(pca_model.explained_variance_ratio_[1])
    for source in sources:
        matrix = (
            targets_physical[plot_split]
            if source == "true"
            else predictions_physical[plot_split]
        )
        for target_index, target in enumerate(TARGET_NAMES):
            plot_embedding_colored(
                pca_scores[plot_split][:, :2],
                matrix[:, target_index],
                title=(
                    f"{display_name}: {plot_split} PCA colored by "
                    f"{source} {target}"
                ),
                xlabel=f"PC1 ({pc1_variance:.2f}% variance)",
                ylabel=f"PC2 ({pc2_variance:.2f}% variance)",
                colorbar_label=f"{source} {target}",
                path=representation_dir / f"{plot_split}_pca_{source}_{target}.png",
                maximum_points=args.max_plot_points,
                seed=args.sample_seed + 101 * (target_index + 1) + (0 if source == "true" else 5000),
            )

    stage_start = time.time()
    print(
        f"  [{name}] CCA: fitting on at most {args.cca_fit_max_samples or 'all'} "
        f"{args.fit_split} samples...",
        flush=True,
    )
    cca_model = fit_cca_model(
        embeddings[args.fit_split],
        targets_physical[args.fit_split],
        args,
        name,
        resolve_device(args.device),
    )
    print(
        f"  [{name}] CCA fit completed in {time.time() - stage_start:.1f} s "
        f"using {cca_model['fit_samples_used']} samples.",
        flush=True,
    )
    cca_rows: List[Dict[str, Any]] = []
    cca_loading_rows: List[Dict[str, Any]] = []
    cca_scores_by_split: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    split_correlations: Dict[str, List[float]] = {}
    for split in args.splits:
        x_scores, y_scores = transform_cca(
            cca_model,
            embeddings[split],
            targets_physical[split],
            args.pca_transform_batch_size,
        )
        cca_scores_by_split[split] = (x_scores, y_scores)
        correlations: List[float] = []
        for component in range(int(cca_model["n_components"])):
            correlation = safe_pearson(x_scores[:, component], y_scores[:, component])
            correlations.append(correlation)
            cca_rows.append(
                {
                    "representation": name,
                    "display_name": display_name,
                    "split": split,
                    "canonical_component": component + 1,
                    "canonical_correlation": correlation,
                    "fit_split": args.fit_split,
                    "pca_components_before_cca": int(cca_model["retained_pca_components"]),
                }
            )
            for target_index, target in enumerate(TARGET_NAMES):
                cca_loading_rows.append(
                    {
                        "representation": name,
                        "display_name": display_name,
                        "split": split,
                        "canonical_component": component + 1,
                        "parameter": target,
                        "corr_embedding_variate_with_parameter": safe_pearson(
                            x_scores[:, component], targets_physical[split][:, target_index]
                        ),
                        "corr_parameter_variate_with_parameter": safe_pearson(
                            y_scores[:, component], targets_physical[split][:, target_index]
                        ),
                    }
                )
        split_correlations[split] = correlations
        atomic_npz_save(
            coordinate_dir / f"{split}_cca_coordinates.npz",
            embedding_canonical_scores=x_scores,
            parameter_canonical_scores=y_scores,
            target_physical=np.asarray(targets_physical[split], dtype=np.float32),
            target_names=np.asarray(TARGET_NAMES),
        )
    atomic_csv_write(cca_rows, output_dir / "metrics" / f"{name}_cca_correlations.csv")
    atomic_csv_write(
        cca_loading_rows,
        output_dir / "metrics" / f"{name}_cca_parameter_loadings.csv",
    )
    atomic_json_dump(
        {
            "representation": name,
            "fit_split": args.fit_split,
            "retained_pca_components": int(cca_model["retained_pca_components"]),
            "cca_components": int(cca_model["n_components"]),
            "convergence_warnings": list(cca_model["warnings"]),
            "fit_samples_total": int(cca_model["fit_samples_total"]),
            "fit_samples_used": int(cca_model["fit_samples_used"]),
        },
        output_dir / "models" / f"{name}_cca_metadata.json",
    )
    plot_cca_correlations(
        split_correlations,
        title=f"{display_name}: CCA canonical correlations",
        path=representation_dir / "cca_canonical_correlations.png",
    )
    plot_x_scores, plot_y_scores = cca_scores_by_split[plot_split]
    for component in range(int(cca_model["n_components"])):
        plot_cca_pair(
            plot_x_scores,
            plot_y_scores,
            component=component,
            title=f"{display_name}: {plot_split} CCA component {component + 1}",
            path=representation_dir / f"{plot_split}_cca_component_{component + 1}.png",
            maximum_points=args.max_plot_points,
            seed=args.sample_seed + 9001 + component,
        )

    umap_metadata: Dict[str, Any]
    run_umap = (
        umap_module is not None
        and args.umap_representations != "none"
        and (args.umap_representations == "all" or name == "consensus")
    )
    if not run_umap:
        reason = "module unavailable or disabled"
        if umap_module is not None and args.umap_representations == "consensus" and name != "consensus":
            reason = "configured for consensus representation only"
        umap_metadata = {"status": "skipped", "reason": reason}
    else:
        stage_start = time.time()
        print(f"  [{name}] UMAP: fitting and transforming in batches...", flush=True)
        umap_coordinates, details = fit_umap_coordinates(umap_module, embeddings, args, name, resolve_device(args.device))
        print(f"  [{name}] UMAP completed in {time.time() - stage_start:.1f} s.", flush=True)
        umap_metadata = {"status": "completed", **details}
        for split, coordinates in umap_coordinates.items():
            atomic_npz_save(
                coordinate_dir / f"{split}_umap_coordinates.npz",
                umap_coordinates=coordinates,
                target_physical=np.asarray(targets_physical[split], dtype=np.float32),
                prediction_physical=np.asarray(predictions_physical[split], dtype=np.float32),
                target_names=np.asarray(TARGET_NAMES),
            )
        for source in sources:
            matrix = (
                targets_physical[plot_split]
                if source == "true"
                else predictions_physical[plot_split]
            )
            for target_index, target in enumerate(TARGET_NAMES):
                plot_embedding_colored(
                    umap_coordinates[plot_split],
                    matrix[:, target_index],
                    title=(
                        f"{display_name}: {plot_split} UMAP colored by "
                        f"{source} {target}"
                    ),
                    xlabel="UMAP1",
                    ylabel="UMAP2",
                    colorbar_label=f"{source} {target}",
                    path=representation_dir / f"{plot_split}_umap_{source}_{target}.png",
                    maximum_points=args.max_plot_points,
                    seed=args.sample_seed + 211 * (target_index + 1) + (0 if source == "true" else 6000),
                )

    plot_correlations = [
        row
        for row in correlation_rows
        if row["split"] == plot_split and row["parameter_source"] == "true"
    ]
    strongest = sorted(
        plot_correlations,
        key=lambda row: abs(float(row["pearson_r"]))
        if math.isfinite(float(row["pearson_r"]))
        else -1.0,
        reverse=True,
    )[:12]
    print(
        f"  [{name}] representation analysis completed in "
        f"{time.time() - representation_start:.1f} s.",
        flush=True,
    )
    return {
        "name": name,
        "display_name": display_name,
        "embedding_dimension": int(embeddings[args.fit_split].shape[1]),
        "sample_counts": {split: int(value.shape[0]) for split, value in embeddings.items()},
        "pca_components": int(pca_model.n_components_),
        "pca_explained_variance_ratio": [
            float(value) for value in pca_model.explained_variance_ratio_
        ],
        "strongest_test_or_plot_split_pc_associations": strongest,
        "cca": {
            "fit_split": args.fit_split,
            "retained_pca_components": int(cca_model["retained_pca_components"]),
            "canonical_correlations": split_correlations,
            "convergence_warnings": list(cca_model["warnings"]),
            "fit_samples_total": int(cca_model["fit_samples_total"]),
            "fit_samples_used": int(cca_model["fit_samples_used"]),
        },
        "umap": umap_metadata,
        "runtime_seconds": float(time.time() - representation_start),
        "pca_fit_samples_used": int(pca_fit_indices.size),
        "pca_backend": pca_model.backend_,
        "pca_device": pca_model.device_,
        "pca_fit_seconds": pca_model.fit_seconds_,
    }


def build_consensus_embeddings(
    member_embeddings: Mapping[str, Mapping[str, np.ndarray]],
    fit_split: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    member_names = list(member_embeddings)
    if not member_names:
        raise ValueError("Cannot construct consensus without member embeddings.")
    scalers: Dict[str, StandardScaler] = {}
    transformed: Dict[str, List[np.ndarray]] = {
        split: [] for split in member_embeddings[member_names[0]]
    }
    dimensions: Dict[str, int] = {}
    for member_name in member_names:
        split_map = member_embeddings[member_name]
        scaler = StandardScaler()
        scaler.fit(np.asarray(split_map[fit_split], dtype=np.float32))
        scalers[member_name] = scaler
        dimensions[member_name] = int(split_map[fit_split].shape[1])
        for split, values in split_map.items():
            transformed[split].append(
                scaler.transform(np.asarray(values, dtype=np.float32)).astype(np.float32)
            )
    consensus = {
        split: np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        for split, blocks in transformed.items()
    }
    metadata = {
        "method": "concatenation_of_member_wise_standardized_embeddings",
        "fit_split_for_standardization": fit_split,
        "member_order": member_names,
        "member_dimensions": dimensions,
        "consensus_dimension": int(consensus[fit_split].shape[1]),
        "raw_embedding_average_used": False,
    }
    return consensus, metadata


def markdown_report(
    args: argparse.Namespace,
    members: Sequence[Mapping[str, Any]],
    consistency: Mapping[str, Any],
    representation_summaries: Sequence[Mapping[str, Any]],
    umap_message: Optional[str],
) -> str:
    lines: List[str] = [
        "# Final Ensemble Interpretability Report",
        "",
        "## Analysis design",
        "",
        f"- Embedding layer: `{args.embedding_layer}`.",
        f"- PCA/CCA/UMAP fit split: `{args.fit_split}`.",
        f"- Primary plot split: `{args.plot_split}`.",
        f"- Ensemble members: {len(members)}.",
        "- PCA axes are fitted independently for each network.",
        "- Raw embeddings are not averaged across independently trained networks.",
        "- The consensus representation concatenates member-wise standardized embeddings.",
        "- CCA is fitted against true physical parameters, not predictions.",
        "- Predicted-parameter coloring is provided as a separate diagnostic.",
        "",
        "## Ensemble members",
        "",
    ]
    for member in members:
        lines.append(f"- `{member['name']}` (seed {member['seed']})")
    if umap_message:
        lines.extend(["", "## UMAP status", "", f"- {umap_message}"])

    lines.extend(["", "## Representation summaries", ""])
    for summary in representation_summaries:
        ratios = summary["pca_explained_variance_ratio"]
        first_two = 100.0 * float(sum(ratios[:2]))
        lines.extend(
            [
                f"### {summary['display_name']}",
                "",
                f"- Embedding dimension: {summary['embedding_dimension']}.",
                f"- Variance explained by PC1 + PC2: {first_two:.2f}%.",
                "- Strongest PC associations on the primary plot split:",
            ]
        )
        for row in summary["strongest_test_or_plot_split_pc_associations"][:8]:
            lines.append(
                "  - PC{pc} with {parameter}: Pearson r = {pearson_r:.3f}, "
                "Spearman rho = {spearman_rho:.3f}.".format(**row)
            )
        lines.append("- CCA canonical correlations:")
        for split, values in summary["cca"]["canonical_correlations"].items():
            formatted = ", ".join(f"{float(value):.3f}" for value in values)
            lines.append(f"  - {split}: {formatted}")
        lines.append(f"- UMAP status: {summary['umap']['status']}.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- A large absolute PC-parameter correlation indicates that the corresponding physical variable is aligned with a major variance direction in the embedding.",
            "- PCA signs are arbitrary. A positive or negative sign may flip between independently trained members without changing the represented structure.",
            "- High held-out CCA correlations indicate that linear combinations of embedding features strongly encode combinations of the four physical parameters.",
            "- UMAP is nonlinear and primarily descriptive. Distances and cluster shapes should not be interpreted as calibrated physical scales.",
            "- Agreement across individual members and the consensus analysis is stronger evidence than any single visualization.",
            "",
            "## Configuration consistency",
            "",
            f"- Identical hyperparameters: {consistency['identical_hyperparameters']}.",
            f"- Identical data manifest: {consistency['identical_data_manifest']}.",
            f"- Identical normalization: {consistency['identical_normalization']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    resolve_paths(args)

    module = import_training_module(args.training_script)
    if to_jsonable(module.TARGET_RANGES) is None:
        raise RuntimeError("Could not read target ranges from training module.")
    members, skipped = discover_members(args.experiments_root)
    if len(members) != args.expected_members and not args.allow_partial:
        raise RuntimeError(
            f"Found {len(members)} completed ensemble members, expected "
            f"{args.expected_members}. Use --allow-partial to continue."
        )
    if not members:
        raise RuntimeError("No completed ensemble members are available.")
    consistency = verify_members(members)

    prepare_output_dir(args.output_dir, args.overwrite, args.experiments_root)
    atomic_json_dump(skipped, args.output_dir / "skipped_experiments.json")
    atomic_json_dump(consistency, args.output_dir / "configuration_consistency.json")

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Device: {device}", flush=True)
    print(f"Members: {len(members)}", flush=True)
    print(f"Embedding layer: {args.embedding_layer}", flush=True)

    datasets, targets_physical, targets_normalized, retained_indices = build_datasets(
        module,
        args,
        members[0],
        consistency,
    )
    dataloaders = {
        split: module.create_dataloader(
            dataset=datasets[split],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=(device.type == "cuda"),
            sampler=None,
            shuffle=False,
            seed=args.sample_seed + index,
        )
        for index, split in enumerate(args.splits)
    }

    member_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    member_predictions_physical: Dict[str, Dict[str, np.ndarray]] = {}
    member_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    for member_index, member in enumerate(members, start=1):
        member_name = str(member["name"])
        print(
            f"[{member_index}/{len(members)}] Loading {member_name} "
            f"(seed {member['seed']})",
            flush=True,
        )
        checkpoint = load_checkpoint(Path(member["checkpoint"]), map_location="cpu")
        model = build_model(module, checkpoint, device)
        split_embeddings: Dict[str, np.ndarray] = {}
        split_predictions: Dict[str, np.ndarray] = {}
        for split in args.splits:
            print(f"  Extracting {split} embeddings...", flush=True)
            embedding, prediction_raw, prediction_clamped, targets_seen = extract_member_split(
                module,
                model,
                dataloaders[split],
                device,
                amp_enabled,
                args.embedding_layer,
            )
            if targets_seen.shape != targets_normalized[split].shape or not np.allclose(
                targets_seen,
                targets_normalized[split],
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Target order mismatch while extracting {member_name}/{split}."
                )
            prediction_physical = np.asarray(
                module.denormalize_targets(prediction_clamped, clamp=True),
                dtype=np.float32,
            )
            split_embeddings[split] = embedding
            split_predictions[split] = prediction_physical
            if args.save_embeddings:
                atomic_npz_save(
                    args.output_dir / "embeddings" / f"{member_name}_{split}.npz",
                    embedding=embedding,
                    prediction_normalized_raw=prediction_raw,
                    prediction_normalized_clamped=prediction_clamped,
                    prediction_physical=prediction_physical,
                    target_normalized=targets_normalized[split],
                    target_physical=targets_physical[split],
                    retained_global_indices=retained_indices[split],
                    target_names=np.asarray(TARGET_NAMES),
                    seed=np.asarray([int(member["seed"])], dtype=np.int64),
                )
        member_embeddings[member_name] = split_embeddings
        member_predictions_physical[member_name] = split_predictions
        member_rows.append(
            {
                "member": member_name,
                "seed": int(member["seed"]),
                "checkpoint": str(member["checkpoint"]),
                "embedding_dimension": int(split_embeddings[args.fit_split].shape[1]),
                **{
                    f"{split}_samples": int(split_embeddings[split].shape[0])
                    for split in args.splits
                },
            }
        )
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_csv_write(member_rows, args.output_dir / "ensemble_members.csv")

    ensemble_predictions = {
        split: np.mean(
            np.stack(
                [member_predictions_physical[name][split] for name in member_embeddings],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
        for split in args.splits
    }

    umap_module, umap_message = import_umap(args.umap_policy)
    if umap_message:
        print(f"UMAP: {umap_message}", flush=True)

    representation_summaries: List[Dict[str, Any]] = []
    all_pca_rows: List[Mapping[str, Any]] = []
    all_cca_rows: List[Mapping[str, Any]] = []
    for member in members:
        name = str(member["name"])
        print(f"Analyzing representation: {name}", flush=True)
        summary = analyze_representation(
            name=name,
            display_name=f"Ensemble member seed {member['seed']}",
            embeddings=member_embeddings[name],
            predictions_physical=member_predictions_physical[name],
            targets_physical=targets_physical,
            args=args,
            output_dir=args.output_dir,
            umap_module=umap_module,
            prestandardized=False,
        )
        representation_summaries.append(summary)

    consensus_metadata: Optional[Dict[str, Any]] = None
    if args.consensus:
        print("Building and analyzing consensus representation...", flush=True)
        consensus_embeddings, consensus_metadata = build_consensus_embeddings(
            member_embeddings,
            fit_split=args.fit_split,
        )
        atomic_json_dump(
            consensus_metadata,
            args.output_dir / "models" / "consensus_representation.json",
        )
        if args.save_embeddings:
            for split in args.splits:
                atomic_npz_save(
                    args.output_dir / "embeddings" / f"consensus_{split}.npz",
                    embedding=consensus_embeddings[split],
                    prediction_physical=ensemble_predictions[split],
                    target_physical=targets_physical[split],
                    retained_global_indices=retained_indices[split],
                    target_names=np.asarray(TARGET_NAMES),
                )
        summary = analyze_representation(
            name="consensus",
            display_name="Consensus ensemble representation",
            embeddings=consensus_embeddings,
            predictions_physical=ensemble_predictions,
            targets_physical=targets_physical,
            args=args,
            output_dir=args.output_dir,
            umap_module=umap_module,
            prestandardized=True,
        )
        representation_summaries.append(summary)

    # Merge representation-specific CSV tables into convenient global tables.
    for summary in representation_summaries:
        name = str(summary["name"])
        for filename, collector in (
            (f"{name}_pc_parameter_correlations.csv", all_pca_rows),
            (f"{name}_cca_correlations.csv", all_cca_rows),
        ):
            path = args.output_dir / "metrics" / filename
            if path.is_file():
                with path.open("r", newline="", encoding="utf-8") as handle:
                    collector.extend(list(csv.DictReader(handle)))
    atomic_csv_write(all_pca_rows, args.output_dir / "metrics" / "all_pc_parameter_correlations.csv")
    atomic_csv_write(all_cca_rows, args.output_dir / "metrics" / "all_cca_correlations.csv")

    summary_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.time() - start_time),
        "arguments": vars(args),
        "members": member_rows,
        "configuration_consistency": consistency,
        "sample_counts": {
            split: int(targets_physical[split].shape[0]) for split in args.splits
        },
        "embedding_layer": args.embedding_layer,
        "fit_split": args.fit_split,
        "plot_split": args.plot_split,
        "raw_embedding_average_used": False,
        "consensus": consensus_metadata,
        "umap_message": umap_message,
        "representations": representation_summaries,
    }
    atomic_json_dump(summary_payload, args.output_dir / "FINAL_INTERPRETABILITY_SUMMARY.json")
    atomic_text_write(
        markdown_report(
            args,
            members,
            consistency,
            representation_summaries,
            umap_message,
        ),
        args.output_dir / "interpretability_report.md",
    )
    atomic_text_write(
        "\n".join(str(member["checkpoint"]) for member in members) + "\n",
        args.output_dir / "member_checkpoint_paths.txt",
    )
    print(f"Interpretability analysis completed: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
