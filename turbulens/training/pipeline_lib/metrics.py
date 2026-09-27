from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_correlation(_rankdata(x), _rankdata(y))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError(f"Metric arrays must have equal non-empty shape, got {y_true.shape} and {y_pred.shape}.")
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    residual_std = float(np.std(residual, ddof=1)) if residual.size > 1 else 0.0
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan")
    pearson = pearson_correlation(y_true, y_pred)
    spearman = spearman_correlation(y_true, y_pred)
    if denominator > 0:
        slope = float(np.dot(y_true - y_true.mean(), y_pred - y_pred.mean()) / denominator)
        intercept = float(y_pred.mean() - slope * y_true.mean())
    else:
        slope = float("nan")
        intercept = float("nan")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "bias": bias,
        "residual_std": residual_std,
        "slope": slope,
        "intercept": intercept,
    }


def compute_full_metrics(
    target_normalized: np.ndarray,
    prediction_normalized_raw: np.ndarray,
    prediction_normalized_clamped: np.ndarray,
    target_physical: np.ndarray,
    prediction_physical: np.ndarray,
    targets: Sequence[str],
    loss: float,
) -> dict[str, Any]:
    arrays = [
        np.asarray(target_normalized), np.asarray(prediction_normalized_raw),
        np.asarray(prediction_normalized_clamped), np.asarray(target_physical),
        np.asarray(prediction_physical),
    ]
    expected = (arrays[0].shape[0], len(targets))
    if any(array.shape != expected for array in arrays):
        raise ValueError(f"All metric arrays must have shape {expected}; got {[a.shape for a in arrays]}.")
    per_target: dict[str, Any] = {}
    normalized_metrics: list[dict[str, float]] = []
    physical_metrics: list[dict[str, float]] = []
    for column, target in enumerate(targets):
        raw = regression_metrics(target_normalized[:, column], prediction_normalized_raw[:, column])
        clamped = regression_metrics(target_normalized[:, column], prediction_normalized_clamped[:, column])
        physical = regression_metrics(target_physical[:, column], prediction_physical[:, column])
        out_of_range = float(np.mean(
            (prediction_normalized_raw[:, column] < 0.0) |
            (prediction_normalized_raw[:, column] > 1.0)
        ))
        per_target[target] = {
            "normalized_raw": raw,
            "normalized_clamped": clamped,
            "physical": physical,
            "out_of_range_fraction_raw": out_of_range,
        }
        normalized_metrics.append(clamped)
        physical_metrics.append(physical)

    def macro(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
        keys = ("mae", "mse", "rmse", "r2", "pearson", "spearman", "bias", "residual_std")
        return {
            key: float(np.nanmean([float(row[key]) for row in rows]))
            for key in keys
        }

    return {
        "loss": float(loss),
        "sample_count": int(expected[0]),
        "targets": list(targets),
        "overall": {
            "normalized_clamped_macro": macro(normalized_metrics),
            "physical_macro": macro(physical_metrics),
        },
        "per_target": per_target,
    }


def metric_value(
    metrics: Mapping[str, Any],
    *,
    metric: str,
    target: str,
    space: str,
    targets: Sequence[str],
) -> tuple[float, float, float]:
    """Return (mse, rmse, r2) for selection, regardless of requested metric."""
    if target == "overall":
        if space != "normalized":
            raise ValueError("Overall selection must use normalized space.")
        rows = [metrics["per_target"][name]["normalized_clamped"] for name in targets]
        mse = float(np.nanmean([row["mse"] for row in rows]))
        rmse = float(np.nanmean([row["rmse"] for row in rows]))
        r2 = float(np.nanmean([row["r2"] for row in rows]))
        return mse, rmse, r2
    key = "normalized_clamped" if space == "normalized" else "physical"
    row = metrics["per_target"][target][key]
    return float(row["mse"]), float(row["rmse"]), float(row["r2"])


def minmax_scale(values: np.ndarray, minimum: float | None = None, maximum: float | None = None) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan), float("nan"), float("nan")
    low = float(np.min(finite)) if minimum is None else float(minimum)
    high = float(np.max(finite)) if maximum is None else float(maximum)
    if high <= low:
        scaled = np.zeros_like(values, dtype=np.float64)
    else:
        scaled = (values - low) / (high - low)
    return scaled, low, high


def selection_scores(
    mse: np.ndarray,
    rmse: np.ndarray,
    r2: np.ndarray,
    *,
    metric: str,
    mse_weight: float,
    r2_weight: float,
    scaling: str,
) -> tuple[np.ndarray, dict[str, float]]:
    metric = "mse_r2" if metric == "composite" else metric
    if metric == "mse":
        return np.asarray(mse, dtype=float), {}
    if metric == "rmse":
        return np.asarray(rmse, dtype=float), {}
    if metric == "r2":
        return -np.asarray(r2, dtype=float), {}
    if metric != "mse_r2":
        raise ValueError(metric)
    mse_arr = np.asarray(mse, dtype=float)
    deficit = 1.0 - np.asarray(r2, dtype=float)
    metadata: dict[str, float] = {}
    if scaling == "minmax":
        mse_component, mse_min, mse_max = minmax_scale(mse_arr)
        deficit_component, deficit_min, deficit_max = minmax_scale(deficit)
        metadata = {
            "mse_min": mse_min, "mse_max": mse_max,
            "r2_deficit_min": deficit_min, "r2_deficit_max": deficit_max,
        }
    elif scaling == "raw":
        mse_component = mse_arr
        deficit_component = deficit
    else:
        raise ValueError(scaling)
    return mse_weight * mse_component + r2_weight * deficit_component, metadata


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    *,
    repetitions: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = y_true.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = regression_metrics(y_true, y_pred)[metric_name]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, n, size=n)
        estimates[index] = regression_metrics(y_true[sample], y_pred[sample])[metric_name]
    alpha = 1.0 - level
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(point), float(low), float(high)
