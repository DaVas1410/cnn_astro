#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import shutil
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline_lib.common import atomic_csv_write, atomic_json_dump, atomic_npz_save, load_json, utc_now, valid_status_marker
from pipeline_lib.metrics import bootstrap_metric_ci, compute_full_metrics, pearson_correlation, regression_metrics, spearman_correlation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a completed multitask or single-target deep ensemble.")
    parser.add_argument("--ensemble-root", type=Path, required=True,
                        help="Directory containing member experiment directories, normally <run>/ensemble/members.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-members", type=int, required=True)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation", "test"])
    parser.add_argument("--interval-levels", nargs="+", type=float, default=[0.68, 0.90, 0.95])
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--write-per-image-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-plot-points", type=int, default=30000)
    parser.add_argument("--plot-seed", type=int, default=2026)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def discover_members(root: Path) -> list[Path]:
    members = []
    for directory in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        if valid_status_marker(directory / "COMPLETED.json") and (directory / "summary.json").is_file():
            members.append(directory)
    return members


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def validate_prediction_compatibility(reference: Mapping[str, np.ndarray], current: Mapping[str, np.ndarray], member: str, split: str) -> None:
    for key in ("sample_index", "sample_key", "target_names", "target_physical", "target_normalized"):
        if key not in current:
            raise KeyError(f"{member} {split}: missing prediction array {key}")
        if not np.array_equal(reference[key], current[key]):
            raise ValueError(f"{member} {split}: {key} differs from the first ensemble member.")
    if str(reference["dataset_fingerprint"].item()) != str(current["dataset_fingerprint"].item()):
        raise ValueError(f"{member} {split}: dataset fingerprint differs.")


def aggregate_member_predictions(
    member_normalized_raw: np.ndarray,
    member_physical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate ensemble members consistently across normalized/physical spaces.

    Individual reported predictions are bounded to the physical target range. The
    bounded ensemble prediction is therefore the arithmetic mean of each member's
    bounded prediction, i.e. mean(clamp(member)), not clamp(mean(member)). The raw
    normalized mean is retained separately for diagnostics.
    """
    normalized_raw = np.asarray(member_normalized_raw, dtype=np.float64)
    physical = np.asarray(member_physical, dtype=np.float64)
    if normalized_raw.ndim != 3 or physical.shape != normalized_raw.shape:
        raise ValueError(
            "Member prediction arrays must have matching (members,samples,targets) shapes."
        )
    if normalized_raw.shape[0] < 2:
        raise ValueError("At least two ensemble members are required.")
    normalized_clamped = np.clip(normalized_raw, 0.0, 1.0)
    return (
        normalized_raw.mean(axis=0),
        normalized_clamped.mean(axis=0),
        physical.mean(axis=0),
        physical.std(axis=0, ddof=1),
    )


def sample_for_plot(n: int, maximum: int, seed: int) -> np.ndarray:
    if n <= maximum:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=maximum, replace=False))


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_members < 2:
        raise ValueError("--expected-members must be at least two.")
    for level in args.interval_levels:
        if not 0 < level < 1:
            raise ValueError("Interval levels must lie in (0,1).")
    root = args.ensemble_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    members = discover_members(root)
    if len(members) != args.expected_members and not args.allow_partial:
        raise RuntimeError(f"Found {len(members)} completed members; expected {args.expected_members}.")
    if len(members) < 2:
        raise RuntimeError("At least two completed members are required.")

    member_rows: list[dict[str, Any]] = []
    member_summaries: list[dict[str, Any]] = []
    for member in members:
        summary = load_json(member / "summary.json")
        config = load_json(member / "configuration.json")
        checkpoint = member / "checkpoints" / "best_checkpoint.pt"
        member_rows.append({
            "member": member.name,
            "member_dir": str(member),
            "seed": config["seed"],
            "architecture": config["model"]["architecture"],
            "in_channels": config["model"]["in_channels"],
            "pretrained": config["model"]["pretrained"],
            "dropout": config["model"]["dropout"],
            "batch_size": config["optimization"]["batch_size"],
            "learning_rate": config["optimization"]["learning_rate"],
            "training_runtime_seconds": summary.get("training_runtime_seconds"),
            "validation_inference_seconds": summary.get("validation", {}).get("inference", {}).get("seconds"),
            "test_inference_seconds": summary.get("test", {}).get("inference", {}).get("seconds"),
            "test_samples_per_second": summary.get("test", {}).get("inference", {}).get("samples_per_second"),
            "parameter_count_total": summary.get("parameter_count", {}).get("total"),
            "parameter_count_trainable": summary.get("parameter_count", {}).get("trainable"),
            "best_checkpoint": str(checkpoint) if checkpoint.is_file() else None,
        })
        member_summaries.append(summary)
    atomic_csv_write(member_rows, output / "ensemble_members.csv")

    overall_result: dict[str, Any] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "ensemble_root": str(root),
        "member_count": len(members),
        "members": [row["member"] for row in member_rows],
        "splits": {},
        "interval_note": "Intervals based on member mean +/- z*SD are descriptive ensemble-spread intervals, not automatically calibrated confidence intervals.",
    }
    all_individual_rows: list[dict[str, Any]] = []

    for split in list(dict.fromkeys(args.splits)):
        prediction_sets: list[dict[str, np.ndarray]] = []
        reference: dict[str, np.ndarray] | None = None
        for member in members:
            path = member / "predictions" / f"{split}_predictions.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {split} predictions for {member.name}: {path}")
            prediction = load_prediction(path)
            if reference is None:
                reference = prediction
            else:
                validate_prediction_compatibility(reference, prediction, member.name, split)
            prediction_sets.append(prediction)
        if reference is None:
            raise RuntimeError(f"No prediction sets were loaded for split {split}.")
        target_names = [str(x) for x in reference["target_names"].tolist()]
        truth_physical = reference["target_physical"].astype(np.float64)
        truth_normalized = reference["target_normalized"].astype(np.float64)
        member_physical = np.stack([item["prediction_physical"] for item in prediction_sets], axis=0).astype(np.float64)
        member_normalized_raw = np.stack([item["prediction_normalized_raw"] for item in prediction_sets], axis=0).astype(np.float64)
        member_normalized_clamped = np.clip(member_normalized_raw, 0.0, 1.0)
        (
            ensemble_normalized_raw_mean,
            ensemble_normalized_clamped_mean,
            ensemble_physical_mean,
            ensemble_physical_sd,
        ) = aggregate_member_predictions(member_normalized_raw, member_physical)

        # Physical truth is already available. For the normalized ensemble metrics, use member mean.
        ensemble_metrics = compute_full_metrics(
            truth_normalized,
            ensemble_normalized_raw_mean,
            ensemble_normalized_clamped_mean,
            truth_physical,
            ensemble_physical_mean,
            target_names,
            loss=float(np.mean((ensemble_normalized_raw_mean - truth_normalized) ** 2)),
        )

        for member_index, member in enumerate(members):
            for column, target in enumerate(target_names):
                physical = regression_metrics(truth_physical[:, column], member_physical[member_index, :, column])
                normalized = regression_metrics(truth_normalized[:, column], member_normalized_clamped[member_index, :, column])
                all_individual_rows.append({
                    "split": split, "member": member.name,
                    "seed": member_rows[member_index]["seed"], "target": target,
                    **{f"physical_{key}": value for key, value in physical.items()},
                    **{f"normalized_{key}": value for key, value in normalized.items()},
                })

        uncertainty_rows: list[dict[str, Any]] = []
        bootstrap_rows: list[dict[str, Any]] = []
        per_image_columns: dict[str, np.ndarray] = {
            "sample_index": reference["sample_index"],
            "sample_key": reference["sample_key"],
        }
        for column, target in enumerate(target_names):
            absolute_error = np.abs(ensemble_physical_mean[:, column] - truth_physical[:, column])
            sd = ensemble_physical_sd[:, column]
            row: dict[str, Any] = {
                "split": split,
                "target": target,
                "mean_ensemble_sd": float(np.mean(sd)),
                "median_ensemble_sd": float(np.median(sd)),
                "ensemble_sd_p75": float(np.quantile(sd, 0.75)),
                "ensemble_sd_p90": float(np.quantile(sd, 0.90)),
                "ensemble_sd_p95": float(np.quantile(sd, 0.95)),
                "maximum_ensemble_sd": float(np.max(sd)),
                "pearson_sd_vs_absolute_error": pearson_correlation(sd, absolute_error),
                "spearman_sd_vs_absolute_error": spearman_correlation(sd, absolute_error),
            }
            for level in sorted(set(args.interval_levels)):
                z = NormalDist().inv_cdf(0.5 + level / 2.0)
                low = ensemble_physical_mean[:, column] - z * sd
                high = ensemble_physical_mean[:, column] + z * sd
                key = f"level_{level:.2f}".replace(".", "p")
                row[f"{key}_coverage"] = float(np.mean((truth_physical[:, column] >= low) & (truth_physical[:, column] <= high)))
                row[f"{key}_mean_width"] = float(np.mean(high - low))
            uncertainty_rows.append(row)

            for metric_name in ("mae", "rmse", "r2", "pearson", "spearman", "bias"):
                point, low, high = bootstrap_metric_ci(
                    truth_physical[:, column], ensemble_physical_mean[:, column], metric_name,
                    repetitions=args.bootstrap_repetitions,
                    seed=args.plot_seed + column * 101 + len(split),
                )
                bootstrap_rows.append({
                    "split": split, "target": target, "metric": metric_name,
                    "estimate": point, "ci95_low": low, "ci95_high": high,
                    "bootstrap_repetitions": args.bootstrap_repetitions,
                })

            per_image_columns[f"truth_{target}"] = truth_physical[:, column]
            per_image_columns[f"ensemble_mean_{target}"] = ensemble_physical_mean[:, column]
            per_image_columns[f"ensemble_sd_{target}"] = sd
            per_image_columns[f"absolute_error_{target}"] = absolute_error
            for member_index, member in enumerate(members):
                per_image_columns[f"prediction_{member.name}_{target}"] = member_physical[member_index, :, column]

            indices = sample_for_plot(len(truth_physical), args.max_plot_points, args.plot_seed + column)
            fig, ax = plt.subplots(figsize=(6.2, 5.4))
            scatter = ax.scatter(
                truth_physical[indices, column], ensemble_physical_mean[indices, column],
                c=sd[indices], s=8, alpha=0.35, rasterized=True,
            )
            low_axis = float(min(truth_physical[:, column].min(), ensemble_physical_mean[:, column].min()))
            high_axis = float(max(truth_physical[:, column].max(), ensemble_physical_mean[:, column].max()))
            ax.plot([low_axis, high_axis], [low_axis, high_axis], "--", linewidth=1)
            ax.set_xlabel(f"True {target}")
            ax.set_ylabel(f"Ensemble mean {target}")
            ax.set_title(f"{split}: ensemble prediction colored by ensemble SD")
            fig.colorbar(scatter, ax=ax, label="Ensemble SD")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            plot_dir = output / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_dir / f"{split}_{target}_ensemble_prediction.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.2, 5.0))
            ax.scatter(sd[indices], absolute_error[indices], s=8, alpha=0.3, rasterized=True)
            ax.set_xlabel("Ensemble SD")
            ax.set_ylabel("Absolute error")
            ax.set_title(f"{split}: uncertainty versus error for {target}")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{split}_{target}_uncertainty_vs_error.png", dpi=180)
            plt.close(fig)

        metrics_dir = output / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(ensemble_metrics, metrics_dir / f"{split}_ensemble_metrics.json")
        atomic_csv_write(uncertainty_rows, metrics_dir / f"{split}_uncertainty_metrics.csv")
        atomic_csv_write(bootstrap_rows, metrics_dir / f"{split}_bootstrap_confidence_intervals.csv")
        atomic_npz_save(
            output / "predictions" / f"{split}_ensemble_predictions.npz",
            sample_index=reference["sample_index"], sample_key=reference["sample_key"],
            target_names=np.asarray(target_names, dtype=str),
            dataset_fingerprint=reference["dataset_fingerprint"],
            target_normalized=truth_normalized,
            target_physical=truth_physical,
            member_names=np.asarray([member.name for member in members], dtype=str),
            member_predictions_normalized_raw=member_normalized_raw,
            member_predictions_physical=member_physical,
            ensemble_prediction_normalized_raw=ensemble_normalized_raw_mean,
            ensemble_prediction_normalized_clamped=ensemble_normalized_clamped_mean,
            ensemble_prediction_physical=ensemble_physical_mean,
            ensemble_prediction_sd_physical=ensemble_physical_sd,
        )
        if args.write_per_image_csv:
            rows = []
            for index in range(len(reference["sample_index"])):
                rows.append({key: values[index] for key, values in per_image_columns.items()})
            atomic_csv_write(rows, output / "predictions" / f"{split}_per_image_predictions.csv")
        overall_result["splits"][split] = {
            "ensemble_metrics": ensemble_metrics,
            "uncertainty_metrics": uncertainty_rows,
            "bootstrap_confidence_intervals": bootstrap_rows,
        }

    atomic_csv_write(all_individual_rows, output / "metrics" / "individual_member_metrics.csv")
    # Summary across member repetitions.
    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_individual_rows:
        summary_groups.setdefault((str(row["split"]), str(row["target"])), []).append(row)
    member_summary_rows: list[dict[str, Any]] = []
    for (split, target), rows in sorted(summary_groups.items()):
        summary_row: dict[str, Any] = {"split": split, "target": target, "n_members": len(rows)}
        for metric in ("mae", "rmse", "r2", "pearson", "spearman", "bias"):
            values = np.asarray([float(row[f"physical_{metric}"]) for row in rows])
            summary_row[f"physical_{metric}_mean"] = float(np.mean(values))
            summary_row[f"physical_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        member_summary_rows.append(summary_row)
    atomic_csv_write(member_summary_rows, output / "metrics" / "individual_member_metrics_summary.csv")

    atomic_json_dump(overall_result, output / "FINAL_ENSEMBLE_EVALUATION.json")
    atomic_json_dump({"status": "completed", "completed_at_utc": utc_now()}, output / "COMPLETED.json")
    print(f"Evaluated {len(members)} ensemble members. Outputs: {output}")


if __name__ == "__main__":
    main()
