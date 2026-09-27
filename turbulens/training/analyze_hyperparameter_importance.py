#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc hyperparameter analysis for completed multitask regression runs.

This utility reads ONLY artifacts already produced by the pipeline. It does not
train, modify, delete, or evaluate any neural network. The intended use is to
inspect a completed multitask grid search and candidate-retraining stage before
choosing a refined hyperparameter grid for a new experiment version.

The analysis is validation-only, matching the scientific selection policy of
this repository. For each stage it reconstructs the exact configuration-level
selection score used by the pipeline from validation MSE/RMSE/R2 and the
selection settings stored in the resolved configuration.

Outputs include:
  * configuration-level metric tables for grid and candidate retraining;
  * marginal effect plots/tables for every varied hyperparameter;
  * descriptive main-effect eta-squared values;
  * cross-validated random-forest permutation importance for the grid;
  * per-target validation MAE/RMSE/R2 summaries;
  * exact selected-grid versus candidate-retraining comparisons;
  * training/validation loss curves for every candidate run;
  * matching grid curves for the same configurations;
  * architecture-level grid learning-curve envelopes;
  * a Markdown report with evidence-based grid-refinement flags.

Important interpretation notes
------------------------------
The grid is a designed factorial experiment, so marginal effects are useful for
screening. They are still descriptive rather than causal because interactions
can exist. Candidate retraining contains a validation-selected subset of the
original grid, so candidate marginal effects are especially subject to
selection bias and should be used as confirmation, not as an independent
hyperparameter search.

Typical command
---------------
python -u analyze_hyperparameter_importance.py \
    --run-root /path/to/turbulens/training/outputs/multitask/local_v2 \
    --overwrite

The default output is <run-root>/hyperparameter_analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Keep numerical helper libraries modest on shared servers. The script is CPU-only.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:  # pragma: no cover - handled cleanly at runtime
    ColumnTransformer = None
    RandomForestRegressor = None
    permutation_importance = None
    KFold = None
    Pipeline = None
    OneHotEncoder = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


FACTOR_SPECS: tuple[tuple[str, str, str], ...] = (
    ("architecture", "model", "architecture"),
    ("in_channels", "model", "in_channels"),
    ("pretrained", "model", "pretrained"),
    ("dropout", "model", "dropout"),
    ("batch_size", "optimization", "batch_size"),
    ("learning_rate", "optimization", "learning_rate"),
    ("weight_decay", "optimization", "weight_decay"),
    ("loss", "optimization", "loss"),
    ("smooth_l1_beta", "optimization", "smooth_l1_beta"),
)

NUMERIC_FACTORS = {
    "in_channels",
    "dropout",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "smooth_l1_beta",
}

# Only these numeric hyperparameters have a meaningful lower/upper search boundary.
# in_channels is numeric in the matrix but semantically categorical (1 vs 3 channels).
BOUNDARY_REFINABLE_FACTORS = {
    "dropout",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "smooth_l1_beta",
}

METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("selection_score", "Selection score", "lower"),
    ("validation_mse", "Validation normalized MSE", "lower"),
    ("validation_rmse", "Validation normalized RMSE", "lower"),
    ("validation_mae", "Validation normalized MAE", "lower"),
    ("validation_r2", "Validation normalized R2", "higher"),
    ("validation_loss", "Validation training objective", "lower"),
)

TARGET_METRICS = ("mae", "rmse", "r2")


@dataclass(frozen=True)
class StageData:
    name: str
    root: Path
    selection: Mapping[str, Any]
    run_rows: list[dict[str, Any]]
    config_rows: list[dict[str, Any]]
    histories: dict[str, dict[str, list[float]]]
    representative_runs: dict[str, list[Path]]
    scaling: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed grid-search and candidate-retraining hyperparameters without retraining."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Completed multitask run root containing grid_search/ and candidate_retraining/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <run-root>/hyperparameter_analysis.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing analysis directory.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG output resolution.")
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--rf-folds", type=int, default=5)
    parser.add_argument("--rf-permutation-repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Skip random-forest permutation importance. Marginal/eta-squared analysis still runs.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Read runs with summary/history files even if COMPLETED.json is absent. Not recommended.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temp.replace(path)


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


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


def valid_completion(directory: Path) -> bool:
    marker = directory / "COMPLETED.json"
    if not marker.is_file():
        return False
    try:
        payload = load_json(marker)
    except Exception:
        return False
    return payload.get("status") == "completed"


def load_resolved_config(run_root: Path) -> dict[str, Any]:
    json_path = run_root / "resolved_config.json"
    if json_path.is_file():
        payload = load_json(json_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object in {json_path}")
        return payload
    yaml_path = run_root / "resolved_config.yaml"
    if yaml_path.is_file():
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read resolved_config.yaml") from exc
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected YAML mapping in {yaml_path}")
        return payload
    raise FileNotFoundError(
        f"Neither resolved_config.json nor resolved_config.yaml exists below {run_root}."
    )


def factor_values(run_config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for factor, section, key in FACTOR_SPECS:
        result[factor] = run_config.get(section, {}).get(key)
    return result


def normalized_macro(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = summary["validation"]
    overall = validation.get("overall", {}).get("normalized_clamped_macro")
    if isinstance(overall, Mapping):
        return overall
    targets = list(summary.get("targets", []))
    if not targets:
        targets = list(validation.get("per_target", {}).keys())
    rows = [validation["per_target"][target]["normalized_clamped"] for target in targets]
    return {
        metric: float(np.nanmean([float(row[metric]) for row in rows]))
        for metric in ("mae", "mse", "rmse", "r2")
    }


def load_history(path: Path) -> dict[str, list[float]] | None:
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    result: dict[str, list[float]] = {}
    for key, values in payload.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        converted: list[float] = []
        for value in values:
            try:
                converted.append(float(value))
            except (TypeError, ValueError):
                converted.append(float("nan"))
        result[str(key)] = converted
    return result if result else None


def read_stage_runs(stage_root: Path, include_incomplete: bool) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]], dict[str, list[Path]]]:
    experiments = stage_root / "experiments"
    if not experiments.is_dir():
        raise FileNotFoundError(f"Experiments directory not found: {experiments}")
    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    config_dirs: dict[str, list[Path]] = defaultdict(list)
    for summary_path in sorted(experiments.glob("*/summary.json")):
        directory = summary_path.parent
        if not include_incomplete and not valid_completion(directory):
            continue
        config_path = directory / "configuration.json"
        if not config_path.is_file():
            continue
        try:
            summary = load_json(summary_path)
            run_config = load_json(config_path)
            macro = normalized_macro(summary)
            configuration_id = str(run_config.get("configuration_id", directory.name))
            row: dict[str, Any] = {
                "stage": str(run_config.get("stage", stage_root.name)),
                "experiment_id": str(run_config.get("experiment_id", directory.name)),
                "configuration_id": configuration_id,
                "run_dir": str(directory),
                "seed": run_config.get("seed"),
                "repetition": run_config.get("repetition"),
                "best_epoch_one_based": summary.get("best_epoch_one_based"),
                "best_validation_loss": summary.get("best_validation_loss"),
                "validation_loss": summary.get("validation", {}).get("loss"),
                "validation_mae": macro.get("mae"),
                "validation_mse": macro.get("mse"),
                "validation_rmse": macro.get("rmse"),
                "validation_r2": macro.get("r2"),
                "training_runtime_seconds": summary.get("training_runtime_seconds"),
                "total_runtime_seconds": summary.get("total_runtime_seconds"),
                **factor_values(run_config),
            }
            per_target = summary.get("validation", {}).get("per_target", {})
            for target, target_payload in per_target.items():
                normalized_metrics = target_payload.get("normalized_clamped", {})
                physical_metrics = target_payload.get("physical", {})
                for metric in TARGET_METRICS:
                    # Backward-compatible target_* keys are normalized values.
                    row[f"target_{target}_{metric}"] = normalized_metrics.get(metric)
                    row[f"physical_target_{target}_{metric}"] = physical_metrics.get(metric)
            rows.append(row)
            history = load_history(directory / "history" / "history.json")
            if history is not None:
                histories[str(directory)] = history
            config_dirs[configuration_id].append(directory)
        except Exception as exc:
            print(f"WARNING: skipping {directory}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not rows:
        raise RuntimeError(f"No usable completed runs found below {experiments}")
    return rows, histories, dict(config_dirs)


def finite_mean(values: Iterable[Any]) -> float | None:
    data = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            data.append(number)
    return float(np.mean(data)) if data else None


def minmax_scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan), float("nan"), float("nan")
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(values), low, high
    return (values - low) / (high - low), low, high


def compute_selection_scores(
    mse: np.ndarray,
    rmse: np.ndarray,
    r2: np.ndarray,
    selection: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    metric = str(selection.get("metric", "mse_r2"))
    if metric == "composite":
        metric = "mse_r2"
    if metric == "mse":
        return np.asarray(mse, dtype=float), {}
    if metric == "rmse":
        return np.asarray(rmse, dtype=float), {}
    if metric == "r2":
        return -np.asarray(r2, dtype=float), {}
    if metric != "mse_r2":
        raise ValueError(f"Unsupported selection metric: {metric}")
    mse_values = np.asarray(mse, dtype=float)
    deficit = 1.0 - np.asarray(r2, dtype=float)
    scaling = str(selection.get("scaling", "minmax"))
    metadata: dict[str, float] = {}
    if scaling == "minmax":
        mse_component, mse_min, mse_max = minmax_scale(mse_values)
        deficit_component, deficit_min, deficit_max = minmax_scale(deficit)
        metadata = {
            "mse_min": mse_min,
            "mse_max": mse_max,
            "r2_deficit_min": deficit_min,
            "r2_deficit_max": deficit_max,
        }
    elif scaling == "raw":
        mse_component = mse_values
        deficit_component = deficit
    else:
        raise ValueError(f"Unsupported selection scaling: {scaling}")
    score = (
        float(selection.get("mse_weight", 1.0)) * mse_component
        + float(selection.get("r2_weight", 1.0)) * deficit_component
    )
    return score, metadata


def aggregate_configurations(
    run_rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[str(row["configuration_id"])].append(row)
    config_rows: list[dict[str, Any]] = []
    for configuration_id, rows in groups.items():
        first = rows[0]
        aggregate: dict[str, Any] = {
            "configuration_id": configuration_id,
            "n_repetitions": len(rows),
            "representative_run_dir": first["run_dir"],
        }
        for factor, _, _ in FACTOR_SPECS:
            aggregate[factor] = first.get(factor)
        for metric in (
            "validation_loss",
            "validation_mae",
            "validation_mse",
            "validation_rmse",
            "validation_r2",
            "best_validation_loss",
            "training_runtime_seconds",
            "total_runtime_seconds",
        ):
            aggregate[metric] = finite_mean(row.get(metric) for row in rows)
        target_keys = sorted({
            key
            for row in rows
            for key in row
            if key.startswith("target_") or key.startswith("physical_target_")
        })
        for key in target_keys:
            aggregate[key] = finite_mean(row.get(key) for row in rows)
        config_rows.append(aggregate)

    mse = np.asarray([float(row["validation_mse"]) for row in config_rows], dtype=float)
    rmse = np.asarray([float(row["validation_rmse"]) for row in config_rows], dtype=float)
    r2 = np.asarray([float(row["validation_r2"]) for row in config_rows], dtype=float)
    scores, scaling = compute_selection_scores(mse, rmse, r2, selection)
    for row, score in zip(config_rows, scores):
        row["selection_score"] = float(score)
    config_rows.sort(key=lambda row: (float(row["selection_score"]), str(row["configuration_id"])))
    for rank, row in enumerate(config_rows, start=1):
        row["selection_rank"] = rank
    return config_rows, scaling


def load_stage(run_root: Path, name: str, selection: Mapping[str, Any], include_incomplete: bool) -> StageData:
    root = run_root / name
    run_rows, histories, representative_runs = read_stage_runs(root, include_incomplete)
    config_rows, scaling = aggregate_configurations(run_rows, selection)
    return StageData(
        name=name,
        root=root,
        selection=selection,
        run_rows=run_rows,
        config_rows=config_rows,
        histories=histories,
        representative_runs=representative_runs,
        scaling=scaling,
    )


def factor_level_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (1, int(value))
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (2, str(value))


def factor_label(value: Any, factor: str) -> str:
    if factor == "pretrained":
        return "pretrained" if bool(value) else "scratch"
    if isinstance(value, float):
        if abs(value) < 1e-3 and value != 0:
            return f"{value:.0e}"
        return f"{value:g}"
    return str(value)


def safe_name(value: str, max_length: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (cleaned or "item")[:max_length]


def metric_direction(metric: str) -> str:
    for key, _, direction in METRIC_SPECS:
        if key == metric:
            return direction
    return "lower"


def metric_title(metric: str) -> str:
    for key, title, _ in METRIC_SPECS:
        if key == metric:
            return title
    return metric


def finite_metric_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> list[Mapping[str, Any]]:
    result = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(row)
    return result


def effect_summary(rows: Sequence[Mapping[str, Any]], factor: str, metric: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in finite_metric_rows(rows, metric):
        level = row.get(factor)
        try:
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        grouped[level].append(value)
    levels = sorted(grouped, key=factor_level_sort_key)
    table: list[dict[str, Any]] = []
    for level in levels:
        values = np.asarray(grouped[level], dtype=float)
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        sem = std / math.sqrt(values.size) if values.size > 0 else float("nan")
        table.append(
            {
                "factor": factor,
                "level": level,
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": std,
                "sem": sem,
                "ci95_low": float(np.mean(values) - 1.96 * sem),
                "ci95_high": float(np.mean(values) + 1.96 * sem),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    all_values = np.asarray(
        [float(row[metric]) for row in finite_metric_rows(rows, metric)], dtype=float
    )
    total_mean = float(np.mean(all_values)) if all_values.size else float("nan")
    total_ss = float(np.sum((all_values - total_mean) ** 2)) if all_values.size else 0.0
    between_ss = 0.0
    for item in table:
        between_ss += int(item["n"]) * (float(item["mean"]) - total_mean) ** 2
    eta_squared = between_ss / total_ss if total_ss > 0 else 0.0
    if table:
        direction = metric_direction(metric)
        best = min(table, key=lambda item: float(item["mean"])) if direction == "lower" else max(table, key=lambda item: float(item["mean"]))
        worst = max(table, key=lambda item: float(item["mean"])) if direction == "lower" else min(table, key=lambda item: float(item["mean"]))
        beneficial_span = (
            float(worst["mean"]) - float(best["mean"])
            if direction == "lower"
            else float(best["mean"]) - float(worst["mean"])
        )
        overall_std = float(np.std(all_values, ddof=1)) if all_values.size > 1 else 0.0
        normalized_span = beneficial_span / overall_std if overall_std > 0 else 0.0
        best_level = best["level"]
        worst_level = worst["level"]
    else:
        beneficial_span = normalized_span = eta_squared = 0.0
        best_level = worst_level = None
    meta = {
        "factor": factor,
        "metric": metric,
        "n_levels": len(table),
        "eta_squared": float(eta_squared),
        "best_level": best_level,
        "worst_level": worst_level,
        "beneficial_mean_span": float(beneficial_span),
        "normalized_mean_span": float(normalized_span),
    }
    return table, meta


def plot_factor_effect(
    rows: Sequence[Mapping[str, Any]],
    factor: str,
    metric: str,
    output_path: Path,
    dpi: int,
) -> None:
    valid = finite_metric_rows(rows, metric)
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in valid:
        grouped[row.get(factor)].append(float(row[metric]))
    levels = sorted(grouped, key=factor_level_sort_key)
    if len(levels) < 2:
        return
    labels = [factor_label(level, factor) for level in levels]
    arrays = [np.asarray(grouped[level], dtype=float) for level in levels]
    means = np.asarray([np.mean(values) for values in arrays], dtype=float)
    sems = np.asarray(
        [np.std(values, ddof=1) / math.sqrt(values.size) if values.size > 1 else 0.0 for values in arrays],
        dtype=float,
    )
    x = np.arange(len(levels), dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(levels) + 3.0), 6.0))
    rng = np.random.default_rng(7301)
    for index, values in enumerate(arrays):
        jitter = rng.normal(0.0, 0.055, size=values.size)
        ax.scatter(np.full(values.size, x[index]) + jitter, values, s=18, alpha=0.28)
    ax.errorbar(x, means, yerr=1.96 * sems, marker="o", linestyle="-", capsize=4, linewidth=1.6, label="mean +/- 95% normal CI")
    ax.set_xticks(x, labels, rotation=25 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_xlabel(factor)
    ax.set_ylabel(metric_title(metric))
    direction = "lower is better" if metric_direction(metric) == "lower" else "higher is better"
    ax.set_title(f"{factor}: marginal effect on {metric_title(metric)} ({direction})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_importance(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    effect_meta: Sequence[Mapping[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    usable = [item for item in effect_meta if int(item.get("n_levels", 0)) > 1]
    if not usable:
        return
    usable = sorted(usable, key=lambda item: float(item.get("eta_squared", 0.0)))
    labels = [str(item["factor"]) for item in usable]
    values = np.asarray([float(item["eta_squared"]) for item in usable], dtype=float)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.55 * len(labels) + 2.0)))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.set_xlabel("One-factor eta-squared (descriptive variance fraction)")
    ax.set_title(f"Marginal hyperparameter importance: {metric_title(metric)}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def factor_matrix(rows: Sequence[Mapping[str, Any]], factors: Sequence[str]) -> np.ndarray:
    data: list[list[Any]] = []
    for row in rows:
        values: list[Any] = []
        for factor in factors:
            value = row.get(factor)
            if factor in NUMERIC_FACTORS:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = float("nan")
            else:
                value = str(value)
            values.append(value)
        data.append(values)
    return np.asarray(data, dtype=object)


def permutation_importance_cv(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    factors: Sequence[str],
    *,
    trees: int,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(f"scikit-learn import failed: {SKLEARN_IMPORT_ERROR}")
    valid_rows = finite_metric_rows(rows, metric)
    if len(valid_rows) < max(10, folds * 2):
        raise RuntimeError(f"Too few configurations ({len(valid_rows)}) for {folds}-fold importance.")
    factors = [factor for factor in factors if len({str(row.get(factor)) for row in valid_rows}) > 1]
    if not factors:
        raise RuntimeError("No varied factors are available for permutation importance.")
    X = factor_matrix(valid_rows, factors)
    y = np.asarray([float(row[metric]) for row in valid_rows], dtype=float)
    num_indices = [index for index, factor in enumerate(factors) if factor in NUMERIC_FACTORS]
    cat_indices = [index for index, factor in enumerate(factors) if factor not in NUMERIC_FACTORS]
    transformers = []
    if num_indices:
        transformers.append(("numeric", "passthrough", num_indices))
    if cat_indices:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), cat_indices))
    n_splits = max(2, min(int(folds), len(valid_rows) // 2))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_importances: list[np.ndarray] = []
    fold_scores: list[float] = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(X), start=1):
        model = Pipeline(
            steps=[
                ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(trees),
                        random_state=seed + fold_index,
                        min_samples_leaf=2,
                        max_features=1.0,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(X[train_index], y[train_index])
        fold_scores.append(float(model.score(X[test_index], y[test_index])))
        result = permutation_importance(
            model,
            X[test_index],
            y[test_index],
            scoring="r2",
            n_repeats=int(repeats),
            random_state=seed + 1000 + fold_index,
            n_jobs=1,
        )
        fold_importances.append(np.asarray(result.importances_mean, dtype=float))
    matrix = np.vstack(fold_importances)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
    table = [
        {
            "factor": factor,
            "metric": metric,
            "permutation_importance_mean": float(means[index]),
            "permutation_importance_fold_std": float(stds[index]),
            "n_folds": int(matrix.shape[0]),
        }
        for index, factor in enumerate(factors)
    ]
    table.sort(key=lambda row: float(row["permutation_importance_mean"]), reverse=True)
    meta = {
        "metric": metric,
        "fold_test_r2_mean": float(np.mean(fold_scores)),
        "fold_test_r2_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "n_configurations": len(valid_rows),
        "n_folds": matrix.shape[0],
        "trees": int(trees),
        "permutation_repeats": int(repeats),
        "note": "Permutation importance is predictive/descriptive, not a causal effect estimate.",
    }
    return table, meta


def plot_rf_importance(table: Sequence[Mapping[str, Any]], metric: str, output_path: Path, dpi: int) -> None:
    if not table:
        return
    ordered = sorted(table, key=lambda row: float(row["permutation_importance_mean"]))
    labels = [str(row["factor"]) for row in ordered]
    values = np.asarray([float(row["permutation_importance_mean"]) for row in ordered], dtype=float)
    errors = np.asarray([float(row["permutation_importance_fold_std"]) for row in ordered], dtype=float)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.55 * len(labels) + 2.0)))
    ax.barh(y, values, xerr=errors, capsize=3)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Cross-validated permutation importance (drop in held-out R2)")
    ax.set_title(f"Predictive hyperparameter importance: {metric_title(metric)}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def history_arrays(history: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(history.get("train_loss", []), dtype=float)
    val = np.asarray(history.get("validation_loss", []), dtype=float)
    count = min(train.size, val.size)
    if count == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    epoch_values = history.get("epoch", [])
    epochs = np.asarray(epoch_values[:count], dtype=float) if len(epoch_values) >= count else np.arange(1, count + 1, dtype=float)
    return epochs, train[:count], val[:count]


def short_config_label(row: Mapping[str, Any]) -> str:
    arch = str(row.get("architecture", "?"))
    channels = row.get("in_channels", "?")
    pretrained = "pre" if bool(row.get("pretrained")) else "scratch"
    dropout = row.get("dropout", "?")
    lr = row.get("learning_rate", "?")
    batch = row.get("batch_size", "?")
    return f"{arch}_c{channels}_{pretrained}_d{dropout}_lr{lr}_b{batch}"


def plot_single_history(history: Mapping[str, Sequence[float]], title: str, output_path: Path, dpi: int) -> None:
    epochs, train, val = history_arrays(history)
    if epochs.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="validation")
    best_index = int(np.nanargmin(val)) if np.any(np.isfinite(val)) else None
    if best_index is not None:
        ax.scatter([epochs[best_index]], [val[best_index]], s=38, label=f"best val epoch {int(epochs[best_index])}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def row_for_run(stage: StageData, run_dir: Path) -> Mapping[str, Any] | None:
    run_dir_text = str(run_dir)
    return next((row for row in stage.run_rows if str(row.get("run_dir")) == run_dir_text), None)


def plot_candidate_curves(stage: StageData, output_dir: Path, dpi: int) -> int:
    per_run_dir = output_dir / "curves" / "candidate_retraining" / "per_run"
    plotted = 0
    by_architecture: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Sequence[float]]]]] = defaultdict(list)
    all_items: list[tuple[Mapping[str, Any], Mapping[str, Sequence[float]]]] = []
    for row in stage.run_rows:
        history = stage.histories.get(str(row["run_dir"]))
        if not history:
            continue
        label = short_config_label(row)
        plot_single_history(
            history,
            f"Candidate retraining: {label}",
            per_run_dir / f"{safe_name(str(row['experiment_id']))}.png",
            dpi,
        )
        plotted += 1
        by_architecture[str(row.get("architecture"))].append((row, history))
        all_items.append((row, history))

    def overlay(items: Sequence[tuple[Mapping[str, Any], Mapping[str, Sequence[float]]]], title: str, path: Path) -> None:
        if not items:
            return
        fig, ax = plt.subplots(figsize=(11, 7))
        for row, history in items:
            epochs, train, val = history_arrays(history)
            if epochs.size == 0:
                continue
            label = short_config_label(row)
            ax.plot(epochs, train, alpha=0.35, linewidth=1.0, label=f"{label} train")
            ax.plot(epochs, val, alpha=0.75, linewidth=1.4, linestyle="--", label=f"{label} val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if len(items) <= 10:
            ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    overlay(all_items, "Candidate retraining: all train/validation curves", output_dir / "curves" / "candidate_retraining" / "all_candidates_overlay.png")
    for architecture, items in sorted(by_architecture.items()):
        overlay(
            items,
            f"Candidate retraining: {architecture}",
            output_dir / "curves" / "candidate_retraining" / f"architecture_{safe_name(architecture)}.png",
        )
    return plotted


def stack_histories(histories: Sequence[Mapping[str, Sequence[float]]], key: str) -> np.ndarray:
    max_epochs = max((len(history.get(key, [])) for history in histories), default=0)
    matrix = np.full((len(histories), max_epochs), np.nan, dtype=float)
    for row_index, history in enumerate(histories):
        values = np.asarray(history.get(key, []), dtype=float)
        matrix[row_index, : values.size] = values
    return matrix


def plot_grid_architecture_envelopes(stage: StageData, output_dir: Path, dpi: int) -> int:
    by_architecture: dict[str, list[Mapping[str, Sequence[float]]]] = defaultdict(list)
    for row in stage.run_rows:
        history = stage.histories.get(str(row["run_dir"]))
        if history:
            by_architecture[str(row.get("architecture"))].append(history)
    plotted = 0
    for architecture, histories in sorted(by_architecture.items()):
        if not histories:
            continue
        train = stack_histories(histories, "train_loss")
        val = stack_histories(histories, "validation_loss")
        epochs = np.arange(1, max(train.shape[1], val.shape[1]) + 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        for matrix, label, linestyle in ((train, "train", "-"), (val, "validation", "--")):
            median = np.nanmedian(matrix, axis=0)
            q25 = np.nanquantile(matrix, 0.25, axis=0)
            q75 = np.nanquantile(matrix, 0.75, axis=0)
            ax.plot(epochs[: median.size], median, label=f"{label} median", linestyle=linestyle)
            ax.fill_between(epochs[: median.size], q25, q75, alpha=0.18, label=f"{label} IQR")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Grid learning-curve envelope: {architecture} (n={len(histories)})")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = output_dir / "curves" / "grid_architecture_envelopes" / f"{safe_name(architecture)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        plotted += 1
    return plotted


def plot_matching_grid_curves(grid: StageData, candidate: StageData, output_dir: Path, dpi: int) -> tuple[int, list[dict[str, Any]]]:
    matched = sorted(set(grid.representative_runs) & set(candidate.representative_runs))
    comparison_rows: list[dict[str, Any]] = []
    plotted = 0
    grid_by_id = {str(row["configuration_id"]): row for row in grid.config_rows}
    candidate_by_id = {str(row["configuration_id"]): row for row in candidate.config_rows}
    for config_id in matched:
        grid_row = grid_by_id.get(config_id)
        candidate_row = candidate_by_id.get(config_id)
        if grid_row and candidate_row:
            comparison_rows.append(
                {
                    "configuration_id": config_id,
                    "label": short_config_label(candidate_row),
                    "architecture": candidate_row.get("architecture"),
                    "grid_selection_rank": grid_row.get("selection_rank"),
                    "candidate_selection_rank": candidate_row.get("selection_rank"),
                    "grid_selection_score": grid_row.get("selection_score"),
                    "candidate_selection_score": candidate_row.get("selection_score"),
                    "grid_validation_mse": grid_row.get("validation_mse"),
                    "candidate_validation_mse": candidate_row.get("validation_mse"),
                    "grid_validation_rmse": grid_row.get("validation_rmse"),
                    "candidate_validation_rmse": candidate_row.get("validation_rmse"),
                    "grid_validation_mae": grid_row.get("validation_mae"),
                    "candidate_validation_mae": candidate_row.get("validation_mae"),
                    "grid_validation_r2": grid_row.get("validation_r2"),
                    "candidate_validation_r2": candidate_row.get("validation_r2"),
                }
            )
        grid_dirs = grid.representative_runs[config_id]
        candidate_dirs = candidate.representative_runs[config_id]
        for cand_index, cand_dir in enumerate(candidate_dirs, start=1):
            cand_history = candidate.histories.get(str(cand_dir))
            if not cand_history:
                continue
            grid_dir = grid_dirs[min(cand_index - 1, len(grid_dirs) - 1)]
            grid_history = grid.histories.get(str(grid_dir))
            if not grid_history:
                continue
            grid_epochs, grid_train, grid_val = history_arrays(grid_history)
            cand_epochs, cand_train, cand_val = history_arrays(cand_history)
            if grid_epochs.size == 0 or cand_epochs.size == 0:
                continue
            title_label = short_config_label(candidate_row or grid_row or {"architecture": "config"})
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(grid_epochs, grid_train, label="grid/small train", alpha=0.75)
            ax.plot(grid_epochs, grid_val, label="grid/small validation", linestyle="--", alpha=0.9)
            ax.plot(cand_epochs, cand_train, label="candidate/large train", alpha=0.75)
            ax.plot(cand_epochs, cand_val, label="candidate/large validation", linestyle="--", alpha=0.9)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Same configuration, small vs large data\n{title_label}")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            path = output_dir / "curves" / "matching_grid_vs_candidate" / f"{safe_name(config_id)}_rep{cand_index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            plotted += 1
    return plotted, comparison_rows


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: Sequence[float], y: Sequence[float], spearman: bool = False) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return float("nan")
    if spearman:
        x_arr = rankdata(x_arr)
        y_arr = rankdata(y_arr)
    x_arr = x_arr - np.mean(x_arr)
    y_arr = y_arr - np.mean(y_arr)
    denominator = math.sqrt(float(np.dot(x_arr, x_arr) * np.dot(y_arr, y_arr)))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(x_arr, y_arr) / denominator)


def plot_grid_candidate_comparison(rows: Sequence[Mapping[str, Any]], output_dir: Path, dpi: int) -> dict[str, Any]:
    if len(rows) < 2:
        return {"matched_configurations": len(rows)}
    results: dict[str, Any] = {"matched_configurations": len(rows)}
    for metric in ("validation_mse", "validation_rmse", "validation_mae", "validation_r2"):
        x = np.asarray([float(row[f"grid_{metric}"]) for row in rows], dtype=float)
        y = np.asarray([float(row[f"candidate_{metric}"]) for row in rows], dtype=float)
        pearson = correlation(x, y)
        spearman = correlation(x, y, spearman=True)
        results[metric] = {"pearson": pearson, "spearman": spearman}
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(x, y, s=45, alpha=0.8)
        for row, x_value, y_value in zip(rows, x, y):
            ax.annotate(str(row.get("architecture", "")), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel(f"Grid/small {metric_title(metric)}")
        ax.set_ylabel(f"Candidate/large {metric_title(metric)}")
        ax.set_title(f"Selected configurations: small vs large data\nSpearman rho={spearman:.3f}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / "cross_stage" / f"grid_vs_candidate_{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    rank_x = np.asarray([float(row["grid_selection_rank"]) for row in rows], dtype=float)
    rank_y = np.asarray([float(row["candidate_selection_rank"]) for row in rows], dtype=float)
    results["selection_rank_spearman"] = correlation(rank_x, rank_y, spearman=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rank_x, rank_y, s=45, alpha=0.8)
    max_rank = int(max(np.max(rank_x), np.max(rank_y)))
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Grid/small validation rank")
    ax.set_ylabel("Candidate/large validation rank")
    ax.set_title(f"Rank stability after large-data retraining\nSpearman rho={results['selection_rank_spearman']:.3f}")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "cross_stage" / "grid_vs_candidate_selection_rank.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return results


def annotate_pipeline_selection(stage: StageData) -> dict[str, Any]:
    """Annotate configuration rows with the selector outputs already on disk."""
    selection_path = stage.root / "selection" / "selected_configurations.json"
    ranked_path = stage.root / "selection" / "ranked_configurations.json"
    annotations: dict[str, dict[str, Any]] = {}
    selected_count = 0
    if selection_path.is_file():
        try:
            payload = load_json(selection_path)
            for item in payload.get("selected_configurations", []):
                config_id = str(item.get("configuration_id"))
                annotations[config_id] = {
                    "selected_by_pipeline": True,
                    "selection_group": item.get("selection_group"),
                    "group_rank": item.get("group_rank"),
                    "stored_validation_rank": item.get("validation_rank"),
                    "stored_selection_score": item.get("validation_selection_score"),
                }
                selected_count += 1
        except Exception as exc:
            print(f"WARNING: could not read {selection_path}: {exc}", file=sys.stderr)
    for row in stage.config_rows:
        item = annotations.get(str(row["configuration_id"]), {})
        row["selected_by_pipeline"] = bool(item.get("selected_by_pipeline", False))
        row["selection_group"] = item.get("selection_group")
        row["group_rank"] = item.get("group_rank")
        row["stored_validation_rank"] = item.get("stored_validation_rank")
        row["stored_selection_score"] = item.get("stored_selection_score")

    verification: dict[str, Any] = {
        "selection_file": str(selection_path) if selection_path.is_file() else None,
        "ranked_file": str(ranked_path) if ranked_path.is_file() else None,
        "selected_count": selected_count,
        "reconstruction_checked": False,
    }
    if ranked_path.is_file():
        try:
            stored_rows = load_json(ranked_path)
            stored = {str(row["configuration_id"]): row for row in stored_rows}
            differences = []
            rank_mismatches = 0
            for row in stage.config_rows:
                stored_row = stored.get(str(row["configuration_id"]))
                if not stored_row:
                    continue
                stored_score = float(stored_row["validation_selection_score"] if "validation_selection_score" in stored_row else stored_row.get("selection_score"))
                differences.append(abs(float(row["selection_score"]) - stored_score))
                stored_rank = stored_row.get("validation_rank")
                if stored_rank is not None and int(stored_rank) != int(row["selection_rank"]):
                    rank_mismatches += 1
            verification.update({
                "reconstruction_checked": True,
                "matched_configurations": len(differences),
                "max_abs_score_difference": max(differences) if differences else None,
                "rank_mismatches": rank_mismatches,
                "matches_pipeline_selector": bool(differences) and max(differences) <= 1e-10 and rank_mismatches == 0,
            })
        except Exception as exc:
            verification["verification_error"] = f"{type(exc).__name__}: {exc}"
    return verification


def selection_formula(selection: Mapping[str, Any]) -> str:
    metric = str(selection.get("metric", "mse_r2"))
    if metric in {"mse_r2", "composite"}:
        scaling = str(selection.get("scaling", "minmax"))
        return (
            f"score = {float(selection.get('mse_weight', 1.0)):g} * {scaling}(MSE) + "
            f"{float(selection.get('r2_weight', 1.0)):g} * {scaling}(1 - R2); lower is better"
        )
    if metric == "r2":
        return "score = -R2; lower score is better (equivalent to maximizing R2)"
    return f"score = {metric.upper()}; lower is better"


def numeric_boundary_message(factor: str, table: Sequence[Mapping[str, Any]], best_level: Any) -> str | None:
    if factor not in BOUNDARY_REFINABLE_FACTORS or len(table) < 2:
        return None
    try:
        numeric_levels = sorted(float(item["level"]) for item in table)
        best = float(best_level)
    except (TypeError, ValueError):
        return None
    if math.isclose(best, numeric_levels[0], rel_tol=1e-12, abs_tol=1e-15):
        return f"best observed level is the lower boundary ({best_level}); consider extending the next grid below it"
    if math.isclose(best, numeric_levels[-1], rel_tol=1e-12, abs_tol=1e-15):
        return f"best observed level is the upper boundary ({best_level}); consider extending the next grid above it if computationally feasible"
    return f"best observed level ({best_level}) is interior; consider narrowing around this region"


def format_value(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    if number != 0 and (abs(number) < 1e-3 or abs(number) >= 1e4):
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def analyze_factor_effects(stage: StageData, output_dir: Path, dpi: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    level_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for metric, _, _ in METRIC_SPECS:
        metric_meta: list[dict[str, Any]] = []
        for factor, _, _ in FACTOR_SPECS:
            table, meta = effect_summary(stage.config_rows, factor, metric)
            for row in table:
                level_rows.append({"stage": stage.name, **row})
            meta_rows.append({"stage": stage.name, **meta})
            metric_meta.append(meta)
            if len(table) > 1:
                plot_factor_effect(
                    stage.config_rows,
                    factor,
                    metric,
                    output_dir / stage.name / "parameter_effects" / metric / f"{factor}.png",
                    dpi,
                )
        plot_parameter_importance(
            stage.config_rows,
            metric,
            metric_meta,
            output_dir / stage.name / "importance" / f"marginal_eta_squared_{metric}.png",
            dpi,
        )
    write_csv(level_rows, output_dir / stage.name / "parameter_level_summary.csv")
    write_csv(meta_rows, output_dir / stage.name / "marginal_importance_summary.csv")
    return level_rows, meta_rows


def analyze_rf_importance(stage: StageData, output_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    factors = [
        factor
        for factor, _, _ in FACTOR_SPECS
        if len({str(row.get(factor)) for row in stage.config_rows}) > 1
    ]
    for metric, _, _ in METRIC_SPECS:
        try:
            table, meta = permutation_importance_cv(
                stage.config_rows,
                metric,
                factors,
                trees=args.rf_trees,
                folds=args.rf_folds,
                repeats=args.rf_permutation_repeats,
                seed=args.random_seed,
            )
        except Exception as exc:
            metadata.append({"metric": metric, "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"WARNING: RF importance skipped for {metric}: {exc}", file=sys.stderr)
            continue
        for row in table:
            rows.append({"stage": stage.name, **row})
        metadata.append({"stage": stage.name, "status": "completed", **meta})
        plot_rf_importance(
            table,
            metric,
            output_dir / stage.name / "importance" / f"rf_permutation_{metric}.png",
            args.dpi,
        )
    write_csv(rows, output_dir / stage.name / "rf_permutation_importance.csv")
    write_json(metadata, output_dir / stage.name / "rf_permutation_metadata.json")
    return rows, metadata


def per_target_summary(stage: StageData, output_dir: Path, dpi: int) -> list[dict[str, Any]]:
    targets = sorted(
        {
            match.group(1)
            for row in stage.config_rows
            for key in row
            if (match := re.match(r"target_(.+)_(?:mae|rmse|r2)$", key))
        }
    )
    rows: list[dict[str, Any]] = []
    for space in ("normalized", "physical"):
        prefix = "" if space == "normalized" else "physical_"
        for target in targets:
            for metric in TARGET_METRICS:
                key = f"{prefix}target_{target}_{metric}"
                values = np.asarray(
                    [float(row[key]) for row in stage.config_rows if row.get(key) is not None], dtype=float
                )
                if values.size:
                    rows.append(
                        {
                            "stage": stage.name,
                            "space": space,
                            "target": target,
                            "metric": metric,
                            "n": int(values.size),
                            "mean": float(np.nanmean(values)),
                            "std": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
                            "median": float(np.nanmedian(values)),
                            "best": float(np.nanmax(values) if metric == "r2" else np.nanmin(values)),
                        }
                    )
        for metric in TARGET_METRICS:
            metric_targets = [
                target
                for target in targets
                if any(row.get(f"{prefix}target_{target}_{metric}") is not None for row in stage.config_rows)
            ]
            if not metric_targets:
                continue
            data = [
                np.asarray(
                    [
                        float(row[f"{prefix}target_{target}_{metric}"])
                        for row in stage.config_rows
                        if row.get(f"{prefix}target_{target}_{metric}") is not None
                    ],
                    dtype=float,
                )
                for target in metric_targets
            ]
            fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(metric_targets) + 3.0), 6))
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(np.arange(1, len(metric_targets) + 1), metric_targets)
            ax.set_xlabel("Target")
            ax.set_ylabel(f"Validation {space} {metric.upper()}")
            direction = "higher is better" if metric == "r2" else "lower is better"
            ax.set_title(f"{stage.name}: per-target {space} {metric.upper()} across configurations ({direction})")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = output_dir / stage.name / "per_target" / f"all_configurations_{space}_{metric}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    write_csv(rows, output_dir / stage.name / "per_target_metric_summary.csv")
    return rows



def _same_level(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a).strip().lower() == str(b).strip().lower()
    try:
        af = float(a)
        bf = float(b)
        if math.isfinite(af) and math.isfinite(bf):
            return math.isclose(af, bf, rel_tol=1e-12, abs_tol=1e-15)
    except (TypeError, ValueError):
        pass
    return str(a) == str(b)


def pairwise_interaction_summary(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    """Descriptive two-factor interaction eta-squared.

    For each varied factor pair, this computes the weighted sum of squares of
    cell-mean departures from an additive two-main-effect model:

        cell_mean - factor_a_mean - factor_b_mean + grand_mean

    divided by the total sum of squares of the metric. On a balanced factorial
    grid this is the classical two-way interaction contribution. On incomplete
    grids it remains descriptive and should not be interpreted causally.
    """
    valid = [row for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
    if len(valid) < 4:
        return []
    factors = [
        factor
        for factor, _, _ in FACTOR_SPECS
        if len({str(row.get(factor)) for row in valid}) > 1
    ]
    y = np.asarray([float(row[metric]) for row in valid], dtype=float)
    grand = float(np.mean(y))
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total <= 0.0:
        return []

    output: list[dict[str, Any]] = []
    for index_a, factor_a in enumerate(factors):
        for factor_b in factors[index_a + 1 :]:
            mean_a: dict[str, float] = {}
            mean_b: dict[str, float] = {}
            groups_a: dict[str, list[float]] = defaultdict(list)
            groups_b: dict[str, list[float]] = defaultdict(list)
            cells: dict[tuple[str, str], list[float]] = defaultdict(list)
            for row in valid:
                a = str(row.get(factor_a))
                b = str(row.get(factor_b))
                value = float(row[metric])
                groups_a[a].append(value)
                groups_b[b].append(value)
                cells[(a, b)].append(value)
            mean_a = {key: float(np.mean(values)) for key, values in groups_a.items()}
            mean_b = {key: float(np.mean(values)) for key, values in groups_b.items()}

            ss_interaction = 0.0
            for (a, b), values in cells.items():
                cell_mean = float(np.mean(values))
                departure = cell_mean - mean_a[a] - mean_b[b] + grand
                ss_interaction += len(values) * departure * departure

            output.append(
                {
                    "metric": metric,
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "n_levels_a": len(groups_a),
                    "n_levels_b": len(groups_b),
                    "n_cells": len(cells),
                    "interaction_eta_squared": float(ss_interaction / ss_total),
                }
            )
    output.sort(key=lambda row: float(row["interaction_eta_squared"]), reverse=True)
    return output


def analyze_pairwise_interactions(
    stage: StageData,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, _, _ in METRIC_SPECS:
        for item in pairwise_interaction_summary(stage.config_rows, metric):
            rows.append({"stage": stage.name, **item})
    write_csv(rows, output_dir / stage.name / "interaction_importance_summary.csv")
    return rows


def analyze_conditional_best_regime(
    stage: StageData,
    main_effect_meta: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
    *,
    n_conditioning_factors: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Recompute marginal effects after fixing the strongest main-effect factors.

    This is specifically intended to expose interaction-driven reversals such
    as a batch-size preference that changes once the dominant learning-rate and
    pretraining regime is fixed.
    """
    candidates = [
        row
        for row in main_effect_meta
        if row.get("metric") == "selection_score" and int(row.get("n_levels", 0)) > 1
    ]
    candidates.sort(key=lambda row: float(row.get("eta_squared", 0.0)), reverse=True)
    conditioning = candidates[: max(0, int(n_conditioning_factors))]
    filters = {str(row["factor"]): row.get("best_level") for row in conditioning}

    subset = [
        row
        for row in stage.config_rows
        if all(_same_level(row.get(factor), level) for factor, level in filters.items())
    ]
    context = {
        "stage": stage.name,
        "metric": "selection_score",
        "conditioning_factors": filters,
        "original_configurations": len(stage.config_rows),
        "conditional_configurations": len(subset),
        "note": (
            "Conditional descriptive analysis only. The conditioning levels were chosen "
            "from the same validation grid, so this is for interaction diagnosis and "
            "search refinement, not an unbiased estimate of generalization."
        ),
    }
    write_json(context, output_dir / stage.name / "conditional_best_regime" / "context.json")

    level_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    if not subset:
        write_csv(level_rows, output_dir / stage.name / "conditional_best_regime" / "parameter_level_summary.csv")
        write_csv(meta_rows, output_dir / stage.name / "conditional_best_regime" / "marginal_importance_summary.csv")
        return level_rows, meta_rows, context

    conditioned_factors = set(filters)
    for factor, _, _ in FACTOR_SPECS:
        if factor in conditioned_factors:
            continue
        table, meta = effect_summary(subset, factor, "selection_score")
        for row in table:
            level_rows.append({"stage": stage.name, "conditional": True, **row})
        meta_rows.append({"stage": stage.name, "conditional": True, **meta})
        if len(table) > 1:
            plot_factor_effect(
                subset,
                factor,
                "selection_score",
                output_dir / stage.name / "conditional_best_regime" / "parameter_effects" / f"{factor}.png",
                dpi,
            )

    write_csv(level_rows, output_dir / stage.name / "conditional_best_regime" / "parameter_level_summary.csv")
    write_csv(meta_rows, output_dir / stage.name / "conditional_best_regime" / "marginal_importance_summary.csv")
    return level_rows, meta_rows, context


def write_report(
    run_root: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    grid: StageData,
    candidate: StageData,
    grid_effect_levels: Sequence[Mapping[str, Any]],
    grid_effect_meta: Sequence[Mapping[str, Any]],
    grid_rf_rows: Sequence[Mapping[str, Any]],
    grid_interactions: Sequence[Mapping[str, Any]],
    conditional_levels: Sequence[Mapping[str, Any]],
    conditional_meta: Sequence[Mapping[str, Any]],
    conditional_context: Mapping[str, Any],
    cross_stage: Mapping[str, Any],
    curve_counts: Mapping[str, int],
    selection_verification: Mapping[str, Mapping[str, Any]],
) -> None:
    selection_metric = "selection_score"
    selection_effects = [row for row in grid_effect_meta if row.get("metric") == selection_metric]
    selection_effects = sorted(selection_effects, key=lambda row: float(row.get("eta_squared", 0.0)), reverse=True)
    level_lookup: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in grid_effect_levels:
        if row.get("metric") == selection_metric:
            level_lookup[str(row["factor"])].append(row)
    rf_selection = [row for row in grid_rf_rows if row.get("metric") == selection_metric]
    rf_selection = sorted(rf_selection, key=lambda row: float(row.get("permutation_importance_mean", 0.0)), reverse=True)
    interaction_selection = [
        row for row in grid_interactions if row.get("metric") == selection_metric
    ]
    interaction_selection = sorted(
        interaction_selection,
        key=lambda row: float(row.get("interaction_eta_squared", 0.0)),
        reverse=True,
    )
    conditional_selection = [
        row for row in conditional_meta if row.get("metric") == selection_metric
    ]
    conditional_selection = sorted(
        conditional_selection,
        key=lambda row: float(row.get("eta_squared", 0.0)),
        reverse=True,
    )
    conditional_level_lookup: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in conditional_levels:
        if row.get("metric") == selection_metric:
            conditional_level_lookup[str(row["factor"])].append(row)

    lines: list[str] = []
    lines.append("# Hyperparameter analysis report")
    lines.append("")
    lines.append("## Scope and scientific guardrails")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Grid configurations analyzed: **{len(grid.config_rows)}** from **{len(grid.run_rows)}** completed runs.")
    lines.append(f"- Candidate configurations analyzed: **{len(candidate.config_rows)}** from **{len(candidate.run_rows)}** completed runs.")
    lines.append("- Hyperparameter ranking uses **validation metrics only**. No test result is used here.")
    lines.append(f"- Grid selection rule: `{selection_formula(grid.selection)}`.")
    lines.append(f"- Candidate selection rule: `{selection_formula(candidate.selection)}`.")
    for stage_name in ("grid_search", "candidate_retraining"):
        check = selection_verification.get(stage_name, {})
        if check.get("reconstruction_checked"):
            status = "MATCH" if check.get("matches_pipeline_selector") else "MISMATCH"
            lines.append(
                f"- Selection-score reconstruction check for **{stage_name}**: **{status}** "
                f"(max abs difference={format_value(check.get('max_abs_score_difference'), 4)}, "
                f"rank mismatches={check.get('rank_mismatches')})."
            )
    lines.append("- The selection score is stage-relative when min-max scaling is used, so raw MSE/RMSE/R2 are preferred for cross-stage comparisons.")
    lines.append("")
    lines.append("## Best validation-ranked configurations")
    lines.append("")
    for stage in (grid, candidate):
        best = stage.config_rows[0]
        lines.append(
            f"- **{stage.name}**: rank 1 `{short_config_label(best)}`, "
            f"score={format_value(best.get('selection_score'))}, "
            f"MSE={format_value(best.get('validation_mse'))}, "
            f"RMSE={format_value(best.get('validation_rmse'))}, "
            f"R2={format_value(best.get('validation_r2'))}."
        )
    lines.append("")
    lines.append("## Which hyperparameters mattered most in the grid?")
    lines.append("")
    lines.append("The table below ranks **descriptive main effects** by one-factor eta-squared for the exact validation selection score. Because the grid is factorial, this is useful for screening, but interactions may still matter.")
    lines.append("")
    lines.append("| Rank | Hyperparameter | Levels | eta^2 | Best marginal level | Mean-score span / overall SD |")
    lines.append("|---:|---|---:|---:|---|---:|")
    for index, item in enumerate(selection_effects, start=1):
        lines.append(
            f"| {index} | {item['factor']} | {item['n_levels']} | {format_value(item['eta_squared'], 4)} | "
            f"{item.get('best_level')} | {format_value(item.get('normalized_mean_span'), 4)} |"
        )
    lines.append("")
    if rf_selection:
        lines.append("### Cross-validated predictive importance")
        lines.append("")
        lines.append("A random-forest model was also used as a nonlinear diagnostic. Permutation importance is the average drop in held-out R2 when one original hyperparameter column is shuffled. It is **predictive importance, not causal importance**.")
        lines.append("")
        lines.append("| Rank | Hyperparameter | Permutation importance | Fold SD |")
        lines.append("|---:|---|---:|---:|")
        for index, item in enumerate(rf_selection, start=1):
            lines.append(
                f"| {index} | {item['factor']} | {format_value(item['permutation_importance_mean'], 4)} | "
                f"{format_value(item['permutation_importance_fold_std'], 4)} |"
            )
        lines.append("")
    if interaction_selection:
        lines.append("### Pairwise interaction diagnostics")
        lines.append("")
        lines.append(
            "Main effects can be misleading when hyperparameters interact. The table below shows "
            "descriptive two-factor interaction eta-squared values for the selection score. "
            "These are most interpretable for a balanced factorial grid."
        )
        lines.append("")
        lines.append("| Rank | Interaction | eta^2 |")
        lines.append("|---:|---|---:|")
        for index, item in enumerate(interaction_selection[:10], start=1):
            lines.append(
                f"| {index} | {item['factor_a']} x {item['factor_b']} | "
                f"{format_value(item.get('interaction_eta_squared'), 4)} |"
            )
        lines.append("")

    conditioning = conditional_context.get("conditioning_factors", {})
    if conditioning:
        lines.append("### Conditional analysis inside the dominant regime")
        lines.append("")
        condition_text = ", ".join(f"`{key}={value}`" for key, value in conditioning.items())
        lines.append(
            f"To diagnose interaction-driven reversals, marginal effects were recomputed only among "
            f"configurations satisfying {condition_text}. This retained "
            f"**{conditional_context.get('conditional_configurations', 0)}** of "
            f"**{conditional_context.get('original_configurations', 0)}** grid configurations."
        )
        lines.append("")
        lines.append(
            "This is a search-refinement diagnostic selected using the same validation grid, not an "
            "independent estimate of generalization."
        )
        lines.append("")
        lines.append("| Hyperparameter | Levels | eta^2 in conditional regime | Best level |")
        lines.append("|---|---:|---:|---|")
        for item in conditional_selection:
            if int(item.get("n_levels", 0)) <= 1:
                continue
            lines.append(
                f"| {item['factor']} | {item['n_levels']} | "
                f"{format_value(item.get('eta_squared'), 4)} | {item.get('best_level')} |"
            )
        lines.append("")

    lines.append("## Grid-refinement flags")
    lines.append("")
    for item in selection_effects:
        factor = str(item["factor"])
        table = sorted(level_lookup.get(factor, []), key=lambda row: factor_level_sort_key(row.get("level")))
        if int(item.get("n_levels", 0)) <= 1:
            only = table[0]["level"] if table else "unknown"
            lines.append(f"- **{factor}**: only one level was tested (`{only}`), so its effect cannot be estimated from this grid.")
            continue
        message = numeric_boundary_message(factor, table, item.get("best_level"))
        if message:
            lines.append(
                f"- **{factor}**: {message}. eta^2={format_value(item.get('eta_squared'), 4)}; "
                f"best marginal selection-score mean at `{item.get('best_level')}`."
            )
        else:
            ordered = sorted(table, key=lambda row: float(row["mean"]))
            best = ordered[0]
            second = ordered[1] if len(ordered) > 1 else None
            second_text = f", next `{second['level']}`={format_value(second['mean'])}" if second else ""
            lines.append(
                f"- **{factor}**: best marginal level `{best['level']}` with mean score {format_value(best['mean'])}{second_text}; "
                f"eta^2={format_value(item.get('eta_squared'), 4)}."
            )
    lines.append("")
    lines.append("These flags are intended to guide a **new** grid, not to alter the completed run. For any scientifically changed rerun, increment `experiment.version` (for example from `local_v2` to `local_v3`) so the current results remain immutable and comparable.")
    lines.append("")
    lines.append("## Does the small-data grid ranking survive large-data candidate retraining?")
    lines.append("")
    matched = int(cross_stage.get("matched_configurations", 0))
    lines.append(f"- Exact configurations matched across stages: **{matched}**.")
    rank_rho = cross_stage.get("selection_rank_spearman")
    if rank_rho is not None:
        lines.append(f"- Grid-vs-candidate validation rank Spearman correlation: **{format_value(rank_rho, 4)}**.")
    for metric in ("validation_mse", "validation_rmse", "validation_mae", "validation_r2"):
        result = cross_stage.get(metric)
        if isinstance(result, Mapping):
            lines.append(
                f"- {metric}: Pearson={format_value(result.get('pearson'), 4)}, "
                f"Spearman={format_value(result.get('spearman'), 4)}."
            )
    lines.append("")
    lines.append("A high positive rank correlation means the small-data screen transfers reasonably well to the large dataset. Weak or negative correlation means the next search should put more emphasis on the large-data candidate stage and avoid over-pruning based only on the small grid.")
    lines.append("")
    lines.append("## Learning curves generated")
    lines.append("")
    lines.append(f"- Candidate per-run learning curves: **{curve_counts.get('candidate_per_run', 0)}**.")
    lines.append(f"- Exact grid-vs-candidate paired curves: **{curve_counts.get('matching_pairs', 0)}**.")
    lines.append(f"- Grid architecture envelopes: **{curve_counts.get('grid_architecture_envelopes', 0)}**.")
    lines.append("")
    lines.append("The paired plots compare the same hyperparameter configuration on the small grid dataset and the large candidate dataset. Absolute loss magnitudes need not be directly comparable across datasets; focus on convergence, instability, overfitting gap, early-stopping behavior, and whether conclusions are consistent.")
    lines.append("")
    lines.append("## Output map")
    lines.append("")
    lines.append("- `grid_search/configuration_metrics.csv`: one row per grid configuration.")
    lines.append("- `candidate_retraining/configuration_metrics.csv`: one row per candidate configuration.")
    lines.append("- `*/parameter_level_summary.csv`: marginal means, SD, SEM, 95% normal CI, quantiles.")
    lines.append("- `*/marginal_importance_summary.csv`: eta-squared and mean-span effect summaries.")
    lines.append("- `grid_search/rf_permutation_importance.csv`: nonlinear cross-validated importance diagnostic.")
    lines.append("- `grid_search/interaction_importance_summary.csv`: descriptive pairwise interaction strengths.")
    lines.append("- `grid_search/conditional_best_regime/`: marginal effects after fixing the dominant grid factors at their best observed levels.")
    lines.append("- `cross_stage/grid_candidate_comparison.csv`: exact selected configurations across small and large data.")
    lines.append("- `curves/candidate_retraining/per_run/`: every candidate train/validation curve.")
    lines.append("- `curves/matching_grid_vs_candidate/`: same configuration on grid vs candidate data.")
    lines.append("- `curves/grid_architecture_envelopes/`: median and IQR learning curves across all grid runs of each architecture.")
    lines.append("")
    (output_dir / "ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_root / "hyperparameter_analysis").resolve()
    )
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    if output_dir == run_root or run_root in output_dir.parents and output_dir.name in {"grid_search", "candidate_retraining", "ensemble"}:
        raise ValueError("Refusing to write analysis over a pipeline stage directory.")
    if output_dir.exists():
        if not args.overwrite:
            if any(output_dir.iterdir()):
                raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite.")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_resolved_config(run_root)
    search_selection = config.get("search", {}).get("selection")
    candidate_selection = config.get("candidate_training", {}).get("selection")
    if not isinstance(search_selection, Mapping) or not isinstance(candidate_selection, Mapping):
        raise KeyError("Resolved config is missing search.selection or candidate_training.selection.")

    print("=" * 88)
    print("Hyperparameter analysis (validation only; no training)")
    print(f"Run root        : {run_root}")
    print(f"Output directory: {output_dir}")
    print(f"Grid criterion  : {selection_formula(search_selection)}")
    print(f"Candidate rule  : {selection_formula(candidate_selection)}")
    print("=" * 88)

    grid = load_stage(run_root, "grid_search", search_selection, args.include_incomplete)
    candidate = load_stage(run_root, "candidate_retraining", candidate_selection, args.include_incomplete)
    selection_verification = {
        "grid_search": annotate_pipeline_selection(grid),
        "candidate_retraining": annotate_pipeline_selection(candidate),
    }
    write_json(selection_verification, output_dir / "selection_reconstruction_verification.json")
    print(f"Loaded grid: {len(grid.run_rows)} runs, {len(grid.config_rows)} configurations")
    print(f"Loaded candidate retraining: {len(candidate.run_rows)} runs, {len(candidate.config_rows)} configurations")

    write_csv(grid.run_rows, output_dir / "grid_search" / "run_metrics.csv")
    write_csv(grid.config_rows, output_dir / "grid_search" / "configuration_metrics.csv")
    write_json({"selection": grid.selection, "scaling": grid.scaling}, output_dir / "grid_search" / "selection_reconstruction.json")
    write_csv(candidate.run_rows, output_dir / "candidate_retraining" / "run_metrics.csv")
    write_csv(candidate.config_rows, output_dir / "candidate_retraining" / "configuration_metrics.csv")
    write_json({"selection": candidate.selection, "scaling": candidate.scaling}, output_dir / "candidate_retraining" / "selection_reconstruction.json")

    print("Analyzing marginal hyperparameter effects...")
    grid_level_rows, grid_meta_rows = analyze_factor_effects(grid, output_dir, args.dpi)
    analyze_factor_effects(candidate, output_dir, args.dpi)
    per_target_summary(grid, output_dir, args.dpi)
    per_target_summary(candidate, output_dir, args.dpi)

    print("Analyzing pairwise interactions and the dominant conditional regime...")
    grid_interactions = analyze_pairwise_interactions(grid, output_dir)
    conditional_levels, conditional_meta, conditional_context = analyze_conditional_best_regime(
        grid, grid_meta_rows, output_dir, args.dpi, n_conditioning_factors=2
    )

    grid_rf_rows: list[dict[str, Any]] = []
    grid_rf_meta: list[dict[str, Any]] = []
    if not args.skip_rf:
        print("Computing cross-validated random-forest permutation importance for the grid...")
        grid_rf_rows, grid_rf_meta = analyze_rf_importance(grid, output_dir, args)
    else:
        write_json([{"status": "skipped", "reason": "--skip-rf"}], output_dir / "grid_search" / "rf_permutation_metadata.json")

    print("Generating candidate and matched learning curves...")
    candidate_curve_count = plot_candidate_curves(candidate, output_dir, args.dpi)
    envelope_count = plot_grid_architecture_envelopes(grid, output_dir, args.dpi)
    pair_count, comparison_rows = plot_matching_grid_curves(grid, candidate, output_dir, args.dpi)
    write_csv(comparison_rows, output_dir / "cross_stage" / "grid_candidate_comparison.csv")
    cross_stage = plot_grid_candidate_comparison(comparison_rows, output_dir, args.dpi)
    write_json(cross_stage, output_dir / "cross_stage" / "grid_candidate_correlations.json")

    curve_counts = {
        "candidate_per_run": candidate_curve_count,
        "matching_pairs": pair_count,
        "grid_architecture_envelopes": envelope_count,
    }
    summary = {
        "status": "completed",
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "analysis_policy": "validation metrics only; no neural-network training or test-set model selection",
        "grid": {
            "runs": len(grid.run_rows),
            "configurations": len(grid.config_rows),
            "selection": grid.selection,
            "selection_scaling": grid.scaling,
            "best_configuration": grid.config_rows[0],
        },
        "candidate_retraining": {
            "runs": len(candidate.run_rows),
            "configurations": len(candidate.config_rows),
            "selection": candidate.selection,
            "selection_scaling": candidate.scaling,
            "best_configuration": candidate.config_rows[0],
        },
        "cross_stage": cross_stage,
        "curve_counts": curve_counts,
        "random_forest_importance": grid_rf_meta,
        "pairwise_interactions": grid_interactions,
        "conditional_best_regime": {
            "context": conditional_context,
            "marginal_importance": conditional_meta,
        },
        "selection_reconstruction_verification": selection_verification,
    }
    write_json(summary, output_dir / "analysis_summary.json")
    write_report(
        run_root,
        output_dir,
        config,
        grid,
        candidate,
        grid_level_rows,
        grid_meta_rows,
        grid_rf_rows,
        grid_interactions,
        conditional_levels,
        conditional_meta,
        conditional_context,
        cross_stage,
        curve_counts,
        selection_verification,
    )

    print("Analysis completed successfully.")
    print(f"Report: {output_dir / 'ANALYSIS_REPORT.md'}")
    print(f"Summary: {output_dir / 'analysis_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
