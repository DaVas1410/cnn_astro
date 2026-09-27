#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline_lib.common import atomic_csv_write, atomic_json_dump, load_json, utc_now, valid_status_marker
from pipeline_lib.metrics import metric_value, selection_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank completed regression configurations using validation metrics only."
    )
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True,
                        help="JSON file containing metric/target/space/weights/scaling/grouping settings.")
    parser.add_argument("--expected-runs", type=int, default=0)
    parser.add_argument("--require-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def discover_runs(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/summary.json" if recursive else "*/summary.json"
    return sorted({path.parent.resolve() for path in root.glob(pattern) if path.is_file()}, key=str)


def flatten_config_fields(run_config: Mapping[str, Any]) -> dict[str, Any]:
    model = run_config.get("model", {})
    optimization = run_config.get("optimization", {})
    task = run_config.get("task", {})
    return {
        "task_mode": task.get("mode"),
        "targets": ",".join(task.get("targets", [])),
        "architecture": model.get("architecture"),
        "in_channels": model.get("in_channels"),
        "pretrained": model.get("pretrained"),
        "dropout": model.get("dropout"),
        "preserve_resolution": model.get("preserve_resolution"),
        "input_standardization": model.get("input_standardization"),
        "batch_size": optimization.get("batch_size"),
        "learning_rate": optimization.get("learning_rate"),
        "weight_decay": optimization.get("weight_decay"),
        "loss": optimization.get("loss"),
        "smooth_l1_beta": optimization.get("smooth_l1_beta"),
    }


def main() -> None:
    args = build_parser().parse_args()
    root = args.experiments_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    selection = load_json(args.selection_json.expanduser().resolve())
    if output.exists() and args.overwrite:
        import shutil
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(root, args.recursive)
    completed_dirs = [directory for directory in run_dirs if valid_status_marker(directory / "COMPLETED.json")]
    if args.expected_runs > 0 and len(completed_dirs) != args.expected_runs:
        message = f"Found {len(completed_dirs)} completed runs; expected {args.expected_runs}."
        if args.require_all:
            raise RuntimeError(message)
        print("WARNING:", message)
    if not completed_dirs:
        raise RuntimeError(f"No completed experiment summaries found below {root}")

    run_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for directory in completed_dirs:
        try:
            summary = load_json(directory / "summary.json")
            run_config = load_json(directory / "configuration.json")
            targets = list(run_config["task"]["targets"])
            mse, rmse, r2 = metric_value(
                summary["validation"], metric=str(selection["metric"]),
                target=str(selection["target"]), space=str(selection["space"]), targets=targets,
            )
            test_metrics = summary.get("test")
            if test_metrics is not None:
                test_mse, test_rmse, test_r2 = metric_value(
                    test_metrics, metric=str(selection["metric"]),
                    target=str(selection["target"]), space=str(selection["space"]), targets=targets,
                )
            else:
                test_mse = test_rmse = test_r2 = None
            row = {
                "run_dir": str(directory),
                "experiment_id": run_config.get("experiment_id", directory.name),
                "configuration_id": run_config.get("configuration_id", directory.name),
                "repetition": int(run_config.get("repetition", 0)),
                "seed": int(run_config["seed"]),
                "validation_mse": mse,
                "validation_rmse": rmse,
                "validation_r2": r2,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "best_epoch_one_based": summary.get("best_epoch_one_based"),
                "training_runtime_seconds": summary.get("training_runtime_seconds"),
                "total_runtime_seconds": summary.get("total_runtime_seconds"),
                "parameter_count": summary.get("parameter_count", {}).get("trainable"),
                **flatten_config_fields(run_config),
            }
            run_rows.append(row)
        except Exception as exc:
            skipped.append({"run_dir": str(directory), "reason": f"{type(exc).__name__}: {exc}"})
    if not run_rows:
        raise RuntimeError("No usable completed runs remained after validation.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["configuration_id"])].append(row)

    def optional_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else None

    configuration_rows: list[dict[str, Any]] = []
    config_payloads: dict[str, dict[str, Any]] = {}
    for configuration_id, rows in grouped.items():
        first_config = load_json(Path(rows[0]["run_dir"]) / "configuration.json")
        config_payloads[configuration_id] = first_config
        aggregate = {
            "configuration_id": configuration_id,
            "n_repetitions": len(rows),
            "validation_mse_mean": float(np.mean([r["validation_mse"] for r in rows])),
            "validation_mse_std": float(np.std([r["validation_mse"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "validation_rmse_mean": float(np.mean([r["validation_rmse"] for r in rows])),
            "validation_rmse_std": float(np.std([r["validation_rmse"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "validation_r2_mean": float(np.mean([r["validation_r2"] for r in rows])),
            "validation_r2_std": float(np.std([r["validation_r2"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "test_mse_mean": optional_mean(rows, "test_mse"),
            "test_rmse_mean": optional_mean(rows, "test_rmse"),
            "test_r2_mean": optional_mean(rows, "test_r2"),
            "training_runtime_seconds_mean": float(np.mean([float(r["training_runtime_seconds"] or 0) for r in rows])),
            "representative_run_dir": rows[0]["run_dir"],
            **flatten_config_fields(first_config),
        }
        configuration_rows.append(aggregate)

    mse = np.asarray([row["validation_mse_mean"] for row in configuration_rows], dtype=float)
    rmse = np.asarray([row["validation_rmse_mean"] for row in configuration_rows], dtype=float)
    r2 = np.asarray([row["validation_r2_mean"] for row in configuration_rows], dtype=float)
    scores, scaling_metadata = selection_scores(
        mse, rmse, r2,
        metric=str(selection["metric"]),
        mse_weight=float(selection.get("mse_weight", 1.0)),
        r2_weight=float(selection.get("r2_weight", 1.0)),
        scaling=str(selection.get("scaling", "minmax")),
    )
    for row, score in zip(configuration_rows, scores):
        row["validation_selection_score"] = float(score)
    configuration_rows.sort(key=lambda row: (float(row["validation_selection_score"]), str(row["configuration_id"])))
    for rank, row in enumerate(configuration_rows, start=1):
        row["validation_rank"] = rank

    group_by = selection.get("group_by")
    top_per_group = int(selection.get("top_per_group", 1))
    selected_rows: list[dict[str, Any]] = []
    if group_by:
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in configuration_rows:
            by_group[str(row.get(group_by))].append(row)
        for group in sorted(by_group):
            for group_rank, row in enumerate(by_group[group][:top_per_group], start=1):
                copy_row = dict(row)
                copy_row["selection_group"] = group
                copy_row["group_rank"] = group_rank
                selected_rows.append(copy_row)
    else:
        selected_rows = [dict(row) for row in configuration_rows[:top_per_group]]
        for index, row in enumerate(selected_rows, start=1):
            row["selection_group"] = "overall"
            row["group_rank"] = index

    selected_payload = []
    for row in selected_rows:
        config_id = str(row["configuration_id"])
        source = config_payloads[config_id]
        selected_payload.append({
            "configuration_id": config_id,
            "validation_rank": row["validation_rank"],
            "selection_group": row["selection_group"],
            "group_rank": row["group_rank"],
            "validation_selection_score": row["validation_selection_score"],
            "validation_metrics": {
                "mse_mean": row["validation_mse_mean"],
                "rmse_mean": row["validation_rmse_mean"],
                "r2_mean": row["validation_r2_mean"],
            },
            "test_metrics_for_reporting_only": {
                "mse_mean": row["test_mse_mean"],
                "rmse_mean": row["test_rmse_mean"],
                "r2_mean": row["test_r2_mean"],
            },
            "model": source["model"],
            "optimization": source["optimization"],
            "task": source["task"],
            "source_run_config": source,
        })

    result = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "experiments_root": str(root),
        "completed_run_count": len(completed_dirs),
        "usable_run_count": len(run_rows),
        "configuration_count": len(configuration_rows),
        "selection": selection,
        "selection_scaling": scaling_metadata,
        "selected_count": len(selected_payload),
        "selected_configurations": selected_payload,
        "important_note": "Ranking and selection use validation metrics only. Test metrics are optional and never used for selection.",
    }
    atomic_json_dump(result, output / "selected_configurations.json")
    atomic_json_dump(configuration_rows, output / "ranked_configurations.json")
    atomic_csv_write(configuration_rows, output / "ranked_configurations.csv")
    top_k = int(selection.get("top_k", len(configuration_rows)))
    top_rows = configuration_rows[:top_k]
    atomic_json_dump(top_rows, output / "top_configurations.json")
    atomic_csv_write(top_rows, output / "top_configurations.csv")
    atomic_csv_write(run_rows, output / "ranked_runs.csv")
    atomic_json_dump(skipped, output / "skipped_runs.json")
    atomic_json_dump({"status": "completed", "completed_at_utc": utc_now()}, output / "COMPLETED.json")

    print(f"Selected {len(selected_payload)} configurations from {len(configuration_rows)} validation-ranked configurations.")
    if selected_payload:
        print("Best configuration:", selected_payload[0]["configuration_id"])


if __name__ == "__main__":
    main()
