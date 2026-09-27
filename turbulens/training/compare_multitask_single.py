#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pipeline_lib.common import atomic_csv_write, atomic_json_dump, load_json, utc_now
from pipeline_lib.metrics import bootstrap_metric_ci, regression_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multitask and single-target final ensembles on identical test samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Comparison YAML root must be a mapping.")
    return payload


def resolve_config_path(value: Any, config_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()




def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite values in [0,1].")
    n = len(values)
    if n == 0:
        return []
    order = np.argsort(values)
    adjusted_sorted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (n - rank) * float(values[index]))
        running = max(running, adjusted)
        adjusted_sorted[rank] = running
    result = np.empty(n, dtype=np.float64)
    for rank, index in enumerate(order):
        result[index] = adjusted_sorted[rank]
    return result.tolist()


def validate_comparison_config(config: Mapping[str, Any]) -> None:
    required = {
        "multitask_root", "single_target_roots", "output_dir",
        "bootstrap_repetitions", "permutation_repetitions", "seed",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"Comparison config is missing required fields: {missing}")
    roots = config["single_target_roots"]
    if not isinstance(roots, Mapping) or set(roots) != {"k_min", "k_max", "sigma", "beta"}:
        raise ValueError("single_target_roots must define exactly k_min, k_max, sigma, and beta.")
    for key in ("bootstrap_repetitions", "permutation_repetitions"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TypeError(f"{key} must be a positive integer.")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TypeError("seed must be a non-negative integer.")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def prediction_path(run_root: Path) -> Path:
    return run_root / "ensemble" / "evaluation" / "predictions" / "test_ensemble_predictions.npz"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_bootstrap_difference(
    truth: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    truth = np.asarray(truth).reshape(-1)
    a = np.asarray(prediction_a).reshape(-1)
    b = np.asarray(prediction_b).reshape(-1)
    point = regression_metrics(truth, b)[metric] - regression_metrics(truth, a)[metric]
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, truth.size, size=truth.size)
        values[index] = regression_metrics(truth[sample], b[sample])[metric] - regression_metrics(truth[sample], a[sample])[metric]
    low, high = np.quantile(values, [0.025, 0.975])
    return float(point), float(low), float(high)


def sign_flip_pvalue(differences: np.ndarray, repetitions: int, seed: int) -> float:
    differences = np.asarray(differences, dtype=np.float64).reshape(-1)
    observed = abs(float(np.mean(differences)))
    if not differences.size:
        return float("nan")
    rng = np.random.default_rng(seed)
    extreme = 0
    # Chunking avoids allocating repetitions x N for large test sets.
    for _ in range(repetitions):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=differences.size)
        if abs(float(np.mean(differences * signs))) >= observed:
            extreme += 1
    return float((extreme + 1) / (repetitions + 1))


def resolved_config(run_root: Path) -> dict[str, Any]:
    path = run_root / "resolved_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing resolved pipeline configuration: {path}")
    return load_json(path)


def first_member_summary(run_root: Path) -> dict[str, Any]:
    members_root = run_root / "ensemble" / "members"
    summaries = sorted(members_root.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No ensemble member summaries found below {members_root}")
    return load_json(summaries[0])


def member_metric_summary(run_root: Path, target: str, approach: str) -> dict[str, Any]:
    path = run_root / "ensemble" / "evaluation" / "metrics" / "individual_member_metrics_summary.csv"
    for row in read_csv(path):
        if row.get("split") == "test" and row.get("target") == target:
            return {"approach": approach, **row}
    raise KeyError(f"No test member-metric summary for {target} in {path}")


def summarize_cost(run_root: Path) -> dict[str, float]:
    path = run_root / "ensemble" / "evaluation" / "ensemble_members.csv"
    rows = read_csv(path)
    def values(key: str) -> np.ndarray:
        result = []
        for row in rows:
            value = row.get(key)
            if value not in (None, "", "None"):
                result.append(float(value))
        return np.asarray(result, dtype=float)
    training = values("training_runtime_seconds")
    inference = values("test_inference_seconds")
    parameters = values("parameter_count_trainable")
    return {
        "member_count": float(len(rows)),
        "training_seconds_mean": float(np.mean(training)) if training.size else float("nan"),
        "training_seconds_sum": float(np.sum(training)) if training.size else float("nan"),
        "test_inference_seconds_mean": float(np.mean(inference)) if inference.size else float("nan"),
        "parameter_count_trainable_mean": float(np.mean(parameters)) if parameters.size else float("nan"),
    }


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    validate_comparison_config(config)
    config_dir = config_path.parent
    multitask_root = resolve_config_path(config["multitask_root"], config_dir)
    single_roots = {
        str(target): resolve_config_path(path, config_dir)
        for target, path in config["single_target_roots"].items()
    }
    output = resolve_config_path(config["output_dir"], config_dir)
    repetitions = int(config["bootstrap_repetitions"])
    permutation_repetitions = int(config["permutation_repetitions"])
    seed = int(config["seed"])
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    multitask_config = resolved_config(multitask_root)
    multitask_member_summary = first_member_summary(multitask_root)
    multitask_seeds = list(multitask_config["ensemble"]["seeds"][: int(multitask_config["ensemble"]["members"])])
    multitask = load_npz(prediction_path(multitask_root))
    multitask_targets = [str(x) for x in multitask["target_names"].tolist()]
    missing = sorted(set(multitask_targets).difference(single_roots))
    if missing:
        raise KeyError(f"Single-target roots are missing targets: {missing}")

    metric_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    member_summary_rows: list[dict[str, Any]] = []
    preprocessing_checks: dict[str, Any] = {}
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    multitask_cost = summarize_cost(multitask_root)
    cost_rows.append({"approach": "multitask", "target": "all", **multitask_cost})
    total_single_training = 0.0
    total_single_parameters = 0.0
    total_single_inference = 0.0

    for target_index, target in enumerate(multitask_targets):
        single_root = single_roots[target]
        single_config = resolved_config(single_root)
        single_member_summary = first_member_summary(single_root)
        single_seeds = list(single_config["ensemble"]["seeds"][: int(single_config["ensemble"]["members"])])
        for key, multi_value, single_value in (
            ("large_root", multitask_config["data"]["large_root"], single_config["data"]["large_root"]),
            ("split_directories", multitask_config["data"]["split_directories"], single_config["data"]["split_directories"]),
            ("normalization_configuration", multitask_config["data"]["normalization"], single_config["data"]["normalization"]),
            ("augmentation", multitask_config["training"]["augmentation"], single_config["training"]["augmentation"]),
            ("ensemble_seeds", multitask_seeds, single_seeds),
            ("realized_normalization_stats", multitask_member_summary["normalization_stats"], single_member_summary["normalization_stats"]),
        ):
            if multi_value != single_value:
                raise ValueError(f"Unfair comparison: configured {key} differs for target {target}.")
        preprocessing_checks[target] = {
            "same_large_root": True,
            "same_split_directories": True,
            "same_normalization_configuration": True,
            "same_realized_normalization_statistics": True,
            "same_augmentation_policy": True,
            "same_ensemble_seeds": True,
        }
        single = load_npz(prediction_path(single_root))
        single_targets = [str(x) for x in single["target_names"].tolist()]
        if single_targets != [target]:
            raise ValueError(f"Expected single-target run for {target}, found targets {single_targets}.")
        for key in ("sample_index", "sample_key"):
            if not np.array_equal(multitask[key], single[key]):
                raise ValueError(f"Unfair comparison: {key} differs for target {target}.")
        multitask_truth = multitask["target_physical"][:, target_index]
        single_truth = single["target_physical"][:, 0]
        if not np.allclose(multitask_truth, single_truth, rtol=0, atol=1e-7):
            raise ValueError(f"Unfair comparison: ground truth differs for target {target}.")
        multitask_prediction = multitask["ensemble_prediction_physical"][:, target_index]
        single_prediction = single["ensemble_prediction_physical"][:, 0]

        member_summary_rows.append(member_metric_summary(multitask_root, target, "multitask"))
        member_summary_rows.append(member_metric_summary(single_root, target, "single_target"))

        metrics_multi = regression_metrics(multitask_truth, multitask_prediction)
        metrics_single = regression_metrics(single_truth, single_prediction)
        for approach, prediction, metrics in (
            ("multitask", multitask_prediction, metrics_multi),
            ("single_target", single_prediction, metrics_single),
        ):
            row: dict[str, Any] = {"target": target, "approach": approach, "n_test_samples": len(multitask_truth)}
            for metric_name in ("mae", "rmse", "r2", "pearson", "spearman", "bias"):
                point, low, high = bootstrap_metric_ci(
                    multitask_truth, prediction, metric_name,
                    repetitions=repetitions, seed=seed + target_index * 100 + (0 if approach == "multitask" else 1),
                )
                row[metric_name] = point
                row[f"{metric_name}_ci95_low"] = low
                row[f"{metric_name}_ci95_high"] = high
            metric_rows.append(row)

        for metric_name in ("mae", "rmse", "r2", "pearson", "spearman", "bias"):
            difference, low, high = paired_bootstrap_difference(
                multitask_truth, multitask_prediction, single_prediction, metric_name,
                repetitions, seed + target_index * 1000 + len(metric_name),
            )
            if metric_name in {"mae", "rmse"}:
                interpretation = "negative favors single-target"
            elif metric_name in {"r2", "pearson", "spearman"}:
                interpretation = "positive favors single-target"
            else:
                interpretation = "signed bias difference; compare each method's absolute bias to zero"
            difference_rows.append({
                "target": target, "metric": metric_name,
                "difference_single_minus_multitask": difference,
                "ci95_low": low, "ci95_high": high,
                "interpretation": interpretation,
            })
        abs_error_difference = np.abs(single_prediction - single_truth) - np.abs(multitask_prediction - multitask_truth)
        pvalue = sign_flip_pvalue(abs_error_difference, permutation_repetitions, seed + target_index * 77)
        difference_rows.append({
            "target": target, "metric": "paired_absolute_error_mean",
            "difference_single_minus_multitask": float(np.mean(abs_error_difference)),
            "ci95_low": None, "ci95_high": None,
            "permutation_p_value": pvalue,
            "interpretation": "negative favors single-target",
        })

        cost = summarize_cost(single_root)
        cost_rows.append({"approach": "single_target", "target": target, **cost})
        total_single_training += cost["training_seconds_sum"]
        total_single_parameters += cost["parameter_count_trainable_mean"]
        total_single_inference += cost["test_inference_seconds_mean"]

        low_axis = float(min(multitask_truth.min(), multitask_prediction.min(), single_prediction.min()))
        high_axis = float(max(multitask_truth.max(), multitask_prediction.max(), single_prediction.max()))
        fig, ax = plt.subplots(figsize=(6.4, 5.5))
        ax.scatter(multitask_truth, multitask_prediction, s=7, alpha=0.25, label="multitask", rasterized=True)
        ax.scatter(single_truth, single_prediction, s=7, alpha=0.25, label="single-target", rasterized=True)
        ax.plot([low_axis, high_axis], [low_axis, high_axis], "--", linewidth=1)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"Test prediction comparison: {target}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{target}_prediction_vs_truth.png", dpi=180)
        plt.close(fig)

        residual_multi = multitask_prediction - multitask_truth
        residual_single = single_prediction - single_truth
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.hist(residual_multi, bins=80, alpha=0.55, label="multitask")
        ax.hist(residual_single, bins=80, alpha=0.55, label="single-target")
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Prediction - truth")
        ax.set_ylabel("Count")
        ax.set_title(f"Test residual distribution: {target}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{target}_residual_distribution.png", dpi=180)
        plt.close(fig)

    cost_rows.append({
        "approach": "single_target_system", "target": "all_four_models",
        "member_count": float(sum(row["member_count"] for row in cost_rows if row["approach"] == "single_target")),
        "training_seconds_sum": total_single_training,
        "test_inference_seconds_mean": total_single_inference,
        "parameter_count_trainable_mean": total_single_parameters,
    })
    absolute_error_rows = [
        row for row in difference_rows if row.get("metric") == "paired_absolute_error_mean"
    ]
    adjusted = holm_adjust([float(row["permutation_p_value"]) for row in absolute_error_rows])
    for row, adjusted_p in zip(absolute_error_rows, adjusted):
        row["holm_adjusted_p_value_across_targets"] = adjusted_p

    atomic_csv_write(metric_rows, output / "metrics_by_target.csv")
    atomic_csv_write(difference_rows, output / "paired_comparisons.csv")
    atomic_csv_write(cost_rows, output / "computational_cost.csv")
    atomic_csv_write(member_summary_rows, output / "member_training_variability.csv")

    report_lines = [
        "# Multitask versus single-target comparison", "",
        f"Generated: {utc_now()}", "",
        "All comparisons use the same held-out test sample keys and verified identical ground truth arrays.",
        "Model selection metrics are not reused as final test statistics.", "",
        "## Results", "",
    ]
    for target in multitask_targets:
        report_lines.append(f"### {target}")
        for row in metric_rows:
            if row["target"] == target:
                report_lines.append(
                    f"- {row['approach']}: MAE={row['mae']:.6g}, RMSE={row['rmse']:.6g}, "
                    f"R2={row['r2']:.6g}, Pearson={row['pearson']:.6g}, Spearman={row['spearman']:.6g}, bias={row['bias']:.6g}."
                )
        report_lines.append("")
    report_lines += [
        "## Statistical interpretation", "",
        "`paired_comparisons.csv` reports paired bootstrap confidence intervals for differences calculated on identical test samples.",
        "For MAE and RMSE, a negative `single_minus_multitask` value favors the single-target model. For R2 and correlations, a positive value favors the single-target model.",
        "The paired absolute-error randomization test uses sign flips and does not assume normally distributed errors. Holm-Bonferroni adjusted p-values across the four target-wise tests are also reported.",
        "`member_training_variability.csv` reports mean and standard deviation across independently trained ensemble members for each approach and target.", "",
        "## Computational cost", "",
        "The multitask approach uses one network for all targets. The complete single-target system requires four independently trained networks, so total parameters, training time, and inference time are reported separately from per-target cost.",
    ]
    (output / "comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "status": "completed", "completed_at_utc": utc_now(),
        "multitask_root": str(multitask_root),
        "single_target_roots": {target: str(path) for target, path in single_roots.items()},
        "targets": multitask_targets,
        "bootstrap_repetitions": repetitions,
        "permutation_repetitions": permutation_repetitions,
        "fairness_checks": {
            "identical_sample_keys": True,
            "identical_ground_truth": True,
            "held_out_test_only": True,
            "same_large_data_root": True,
            "same_split_directories": True,
            "same_normalization_configuration": True,
            "same_realized_normalization_statistics": True,
            "same_augmentation_policy": True,
            "same_ensemble_seeds": True,
            "per_target_preprocessing_checks": preprocessing_checks,
        },
    }
    atomic_json_dump(summary, output / "FINAL_COMPARISON.json")
    atomic_json_dump({"status": "completed", "completed_at_utc": utc_now()}, output / "COMPLETED.json")
    print(f"Comparison complete: {output}")


if __name__ == "__main__":
    main()
