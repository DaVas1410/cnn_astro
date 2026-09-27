#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeline_lib.common import (
    ExclusiveFileLock,
    atomic_csv_write,
    atomic_json_dump,
    atomic_text_write,
    atomic_yaml_dump,
    capture_environment,
    command_string,
    directory_size_bytes,
    elapsed_text,
    format_bytes,
    file_sha256,
    load_json,
    object_sha256,
    run_checked,
    utc_now,
    valid_status_marker,
)
from pipeline_lib.config import (
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    configuration_count,
    load_and_resolve_config,
    selected_stage_window,
    selection_group_count,
)
from pipeline_lib.data import inspect_dataset
from pipeline_lib.gpu_guard import GPUAvailabilityState, ready_gpu_tokens, wait_for_gpu_tokens

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_regression.py"
SELECT_SCRIPT = SCRIPT_DIR / "select_experiments.py"
EVALUATE_SCRIPT = SCRIPT_DIR / "evaluate_ensemble.py"
INTERPRET_SCRIPT = SCRIPT_DIR / "interpret_embeddings.py"


@dataclass
class TrainingJob:
    experiment_id: str
    output_dir: Path
    config_path: Path
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified YAML-driven, resumable joint-regression pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stage names. Missing dependencies are added unless already valid.")
    parser.add_argument("--from-stage", choices=STAGE_ORDER, default=None)
    parser.add_argument("--to-stage", choices=STAGE_ORDER, default=None)
    parser.add_argument("--force-stage", action="append", default=[], choices=STAGE_ORDER,
                        help="Re-enter a stage even if its stage marker is valid. Completed individual runs remain protected.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--no-auto-dependencies", action="store_true")
    return parser


def stage_marker(config: Mapping[str, Any], stage: str) -> Path:
    return Path(config["paths"]["stage_state"]) / f"{stage}.json"


def stage_output_valid(config: Mapping[str, Any], stage: str) -> bool:
    marker = stage_marker(config, stage)
    if not valid_status_marker(marker):
        return False
    run_root = Path(config["paths"]["run_root"])
    checks: dict[str, Sequence[Path]] = {
        "validate_data": (
            run_root / "data_validation" / "small_manifest.json",
            run_root / "data_validation" / "large_manifest.json",
        ),
        "grid_search": (Path(config["paths"]["grid_root"]) / "COMPLETED.json",),
        "select_candidates": (Path(config["paths"]["grid_root"]) / "selection" / "selected_configurations.json",),
        "retrain_candidates": (Path(config["paths"]["candidate_root"]) / "COMPLETED.json",),
        "select_final": (Path(config["paths"]["candidate_root"]) / "selection" / "selected_configurations.json",),
        "train_ensemble": (Path(config["paths"]["ensemble_root"]) / "members" / "COMPLETED.json",),
        "evaluate_ensemble": (Path(config["paths"]["ensemble_root"]) / "evaluation" / "FINAL_ENSEMBLE_EVALUATION.json",),
        "interpretability": (Path(config["paths"]["ensemble_root"]) / "interpretability" / "COMPLETED.json",),
        "report": (Path(config["paths"]["report_root"]) / "pipeline_report.md",),
    }
    return all(path.is_file() for path in checks.get(stage, ()))


def add_missing_dependencies(config: Mapping[str, Any], stages: Sequence[str], disabled: bool) -> list[str]:
    if disabled:
        for stage in stages:
            for dependency in STAGE_DEPENDENCIES[stage]:
                if dependency not in stages and not stage_output_valid(config, dependency):
                    raise RuntimeError(
                        f"Stage {stage} requires {dependency}, but it was not requested and no valid output exists."
                    )
        return list(stages)
    required = set(stages)
    changed = True
    while changed:
        changed = False
        for stage in list(required):
            for dependency in STAGE_DEPENDENCIES[stage]:
                if dependency not in required and not stage_output_valid(config, dependency):
                    required.add(dependency)
                    changed = True
    return [stage for stage in STAGE_ORDER if stage in required]


def write_stage_marker(config: Mapping[str, Any], stage: str, status: str, **details: Any) -> None:
    payload = {
        "status": status,
        "stage": stage,
        "timestamp_utc": utc_now(),
        "config_hash": config["_meta"]["resolved_config_hash"],
        **details,
    }
    atomic_json_dump(payload, stage_marker(config, stage))


def safe_float_tag(value: Any) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def configuration_identity(task: Mapping[str, Any], model: Mapping[str, Any], optimization: Mapping[str, Any]) -> str:
    return object_sha256({"task": task, "model": model, "optimization": optimization})[:16]


def experiment_name(model: Mapping[str, Any], optimization: Mapping[str, Any], config_id: str, repetition: int, seed: int) -> str:
    initialization = "pretrained" if model["pretrained"] else "scratch"
    return (
        f"{model['architecture']}_c{model['in_channels']}_{initialization}"
        f"_do{safe_float_tag(model['dropout'])}_lr{safe_float_tag(optimization['learning_rate'])}"
        f"_bs{optimization['batch_size']}_{config_id[:8]}_rep{repetition + 1:02d}_seed{seed}"
    )


def build_run_config(
    config: Mapping[str, Any], *, stage: str, data_root: str,
    model: Mapping[str, Any], optimization: Mapping[str, Any],
    seed: int, repetition: int, storage_profile: str,
    configuration_id: str | None = None,
) -> dict[str, Any]:
    task = config["task"]
    config_id = configuration_id or configuration_identity(task, model, optimization)
    run_config = {
        "schema_version": 1,
        "pipeline_config_hash": config["_meta"]["resolved_config_hash"],
        "pipeline_experiment": config["experiment"],
        "stage": stage,
        "configuration_id": config_id,
        "repetition": int(repetition),
        "seed": int(seed),
        "task": task,
        "data": {
            "root": data_root,
            "split_directories": config["data"]["split_directories"],
            "normalization": dict(config["data"]["normalization"]),
            "require_square_for_augmentation": config["data"]["require_square_for_augmentation"],
            "enforce_disjoint_files": config["data"]["enforce_disjoint_files"],
        },
        "model": dict(model),
        "optimization": dict(optimization),
        "training": config["training"],
        "runtime": {
            "device": config["runtime"]["device"],
            "num_workers": config["training"]["num_workers"],
        },
        "checkpointing": config["checkpointing"],
        "evaluation": config["evaluation"][storage_profile],
        "storage": config["storage"][storage_profile],
        "logging": config["logging"],
    }
    run_config["experiment_id"] = experiment_name(model, optimization, config_id, repetition, seed)
    return json.loads(json.dumps(run_config))


def enumerate_grid_jobs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_base = config["model"]
    space = config["search"]["space"]
    repetitions = int(config["search"]["repetitions_per_configuration"])
    seeds = list(config["search"]["seeds"])
    keys = list(space.keys())
    jobs: list[dict[str, Any]] = []
    for architecture, in_channels, pretrained, values in itertools.product(
        model_base["architectures"], model_base["in_channels"], model_base["pretrained"],
        itertools.product(*(space[key] for key in keys)),
    ):
        model = {
            "architecture": architecture,
            "in_channels": int(in_channels),
            "pretrained": bool(pretrained),
            "pretrained_weight": (
                str(model_base["pretrained_weights"][architecture]) if pretrained else None
            ),
            "dropout": float(dict(zip(keys, values))["dropout"]),
            "preserve_resolution": bool(model_base["preserve_resolution"]),
            "input_standardization": model_base["input_standardization"],
            "shared_dimensions": model_base["shared_dimensions"],
            "head_hidden_dimension": model_base["head_hidden_dimension"],
            "dropout_policy": model_base["dropout_policy"],
        }
        value_map = dict(zip(keys, values))
        optimization = {
            "batch_size": int(value_map["batch_size"]),
            "learning_rate": float(value_map["learning_rate"]),
            "weight_decay": float(value_map["weight_decay"]),
            "loss": str(value_map["loss"]),
            "smooth_l1_beta": float(value_map["smooth_l1_beta"]),
        }
        config_id = configuration_identity(config["task"], model, optimization)
        for repetition in range(repetitions):
            jobs.append(build_run_config(
                config, stage="grid_search", data_root=config["data"]["small_root"],
                model=model, optimization=optimization, seed=int(seeds[repetition]),
                repetition=repetition, storage_profile="grid", configuration_id=config_id,
            ))
    return jobs


def jobs_from_selection(
    config: Mapping[str, Any], selection_path: Path, *, stage: str,
    data_root: str, repetitions: int, seeds: Sequence[int], storage_profile: str,
) -> list[dict[str, Any]]:
    payload = load_json(selection_path)
    selected = payload["selected_configurations"]
    jobs: list[dict[str, Any]] = []
    for item in selected:
        source = item["source_run_config"]
        model = source["model"]
        optimization = source["optimization"]
        config_id = str(item["configuration_id"])
        for repetition in range(repetitions):
            jobs.append(build_run_config(
                config, stage=stage, data_root=data_root,
                model=model, optimization=optimization,
                seed=int(seeds[repetition]), repetition=repetition,
                storage_profile=storage_profile, configuration_id=config_id,
            ))
    return jobs


def effective_gpu_tokens(config: Mapping[str, Any]) -> list[str]:
    """Return physical GPU tokens used by local child processes.

    Priority:
    1. PIPELINE_GPU_TOKENS / PIPELINE_GPUS environment override.
    2. An existing CUDA_VISIBLE_DEVICES mask, interpreted as allowed physical
       tokens; runtime.gpus selects logical positions within that mask.
    3. runtime.gpus from YAML.
    """
    explicit = os.environ.get("PIPELINE_GPU_TOKENS") or os.environ.get("PIPELINE_GPUS")
    if explicit:
        tokens = [token.strip() for token in explicit.replace(",", " ").split() if token.strip()]
        if not tokens:
            raise ValueError("GPU override was set but contained no GPU tokens.")
        return tokens

    requested_slots = [int(x) for x in config["runtime"]["gpus"]]
    inherited_mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if inherited_mask:
        mask_tokens = [token.strip() for token in inherited_mask.split(",") if token.strip()]
        if any(slot >= len(mask_tokens) for slot in requested_slots):
            raise RuntimeError(
                "runtime.gpus refers to more logical GPUs than are exposed by CUDA_VISIBLE_DEVICES. "
                f"runtime.gpus={requested_slots}, CUDA_VISIBLE_DEVICES={inherited_mask!r}."
            )
        return [mask_tokens[slot] for slot in requested_slots]

    return [str(index) for index in requested_slots]


def prepare_jobs(job_configs: Sequence[Mapping[str, Any]], experiments_root: Path) -> list[TrainingJob]:
    jobs: list[TrainingJob] = []
    for run_config in job_configs:
        output = experiments_root / str(run_config["experiment_id"])
        config_path = output / "run_config.requested.json"
        output.mkdir(parents=True, exist_ok=True)
        if config_path.is_file():
            saved = load_json(config_path)
            if object_sha256(saved) != object_sha256(run_config):
                raise ValueError(f"Requested run configuration changed for existing output: {output}")
        else:
            atomic_json_dump(run_config, config_path)
        jobs.append(TrainingJob(str(run_config["experiment_id"]), output, config_path, int(run_config["seed"])))
    return jobs


def run_training_jobs(
    config: Mapping[str, Any], jobs: Sequence[TrainingJob], stage: str,
    status_path: Path, fail_fast: bool, dry_run: bool,
) -> dict[str, int]:
    gpu_tokens = effective_gpu_tokens(config)
    device = str(config["runtime"]["device"])
    max_parallel = int(config["runtime"]["max_parallel_jobs"])
    if device != "cpu":
        max_parallel = min(max_parallel, len(gpu_tokens))
    else:
        gpu_tokens = [f"cpu-slot-{index}" for index in range(max_parallel)]
    max_parallel = max(1, max_parallel)
    pending = list(jobs)
    counts = {"completed": 0, "skipped": 0, "failed": 0, "resumed_or_started": 0}
    status_rows: list[dict[str, Any]] = []
    filtered: list[TrainingJob] = []
    for job in pending:
        if valid_status_marker(job.output_dir / "COMPLETED.json"):
            counts["skipped"] += 1
            status_rows.append({
                "timestamp_utc": utc_now(), "stage": stage, "experiment": job.experiment_id,
                "gpu": None, "pid": None, "exit_code": 0, "result": "skipped_completed",
            })
        else:
            filtered.append(job)
    pending = filtered
    if dry_run:
        print(f"DRY RUN: {stage} would execute {len(pending)} jobs and skip {counts['skipped']} completed jobs.")
        return counts

    active: dict[int, dict[str, Any]] = {}
    # Keep every configured GPU token available to the local launcher. The GPU
    # guard decides which tokens are free; max_parallel controls concurrency.
    available = list(gpu_tokens) if device != "cpu" else list(gpu_tokens[:max_parallel])
    gpu_guard_state = GPUAvailabilityState()
    python_setting = str(config["runtime"]["python_executable"])
    python_executable = sys.executable if python_setting == "current" else python_setting
    base_env = os.environ.copy()
    for key, value in config["runtime"].get("environment", {}).items():
        base_env[str(key)] = str(value)
    poll_seconds = int(config["runtime"]["poll_seconds"])

    def terminate_active() -> None:
        for info in active.values():
            process: subprocess.Popen[Any] = info["process"]
            with contextlib.suppress(Exception):
                process.terminate()
        deadline = time.time() + 30
        for info in active.values():
            process = info["process"]
            remaining = max(0.0, deadline - time.time())
            with contextlib.suppress(Exception):
                process.wait(timeout=remaining)
            if process.poll() is None:
                with contextlib.suppress(Exception):
                    process.kill()

    try:
        while pending or active:
            while pending and len(active) < max_parallel and available:
                if device == "cpu":
                    ready_tokens = list(available)
                else:
                    ready_tokens = ready_gpu_tokens(config, available, gpu_guard_state)
                if not ready_tokens:
                    break
                gpu_token = ready_tokens[0]
                available.remove(gpu_token)
                job = pending.pop(0)
                env = base_env.copy()
                if device != "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
                command = [
                    python_executable, "-u", str(TRAIN_SCRIPT),
                    "--run-config", str(job.config_path),
                    "--output-dir", str(job.output_dir),
                    "--resume", "auto" if config["checkpointing"]["resume"] else "never",
                ]
                log_path = job.output_dir / "launcher.log"
                log_handle = log_path.open("a", encoding="utf-8")
                log_handle.write(f"\n[{utc_now()}] COMMAND: {command_string(command)}\n")
                log_handle.flush()
                process = subprocess.Popen(
                    command, cwd=str(SCRIPT_DIR), env=env,
                    stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                )
                active[process.pid] = {
                    "process": process, "job": job, "gpu": gpu_token, "log": log_handle,
                    "started": time.time(),
                }
                counts["resumed_or_started"] += 1
                print(
                    f"[{utc_now()}] START {stage}: {job.experiment_id} "
                    f"pid={process.pid} gpu_token={gpu_token}",
                    flush=True,
                )

            if not active:
                # No training child is active. This commonly means every requested
                # GPU is occupied by an external process. Avoid a busy loop while
                # the guard waits for the next availability check.
                guard_poll = int(config.get("runtime", {}).get("gpu_wait", {}).get("poll_seconds", poll_seconds))
                time.sleep(max(1, guard_poll if device != "cpu" else poll_seconds))
                continue
            time.sleep(poll_seconds)
            finished: list[int] = []
            for pid, info in active.items():
                process = info["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                finished.append(pid)
                job: TrainingJob = info["job"]
                gpu = info["gpu"]
                info["log"].write(f"[{utc_now()}] EXIT_CODE: {return_code}\n")
                info["log"].close()
                available.append(gpu)
                result = "completed" if return_code == 0 and valid_status_marker(job.output_dir / "COMPLETED.json") else "failed"
                if result == "completed":
                    counts["completed"] += 1
                    if not bool(config["logging"]["retain_individual_run_logs"]):
                        for removable in (job.output_dir / "launcher.log", job.output_dir / "training.log"):
                            with contextlib.suppress(FileNotFoundError):
                                removable.unlink()
                else:
                    counts["failed"] += 1
                status_rows.append({
                    "timestamp_utc": utc_now(), "stage": stage, "experiment": job.experiment_id,
                    "gpu": gpu, "pid": pid, "exit_code": return_code, "result": result,
                    "elapsed_seconds": time.time() - info["started"],
                })
                print(f"[{utc_now()}] END {stage}: {job.experiment_id} result={result} exit={return_code}", flush=True)
                if result == "failed" and fail_fast:
                    terminate_active()
                    atomic_csv_write(status_rows, status_path)
                    raise RuntimeError(f"Training job failed in {stage}: {job.experiment_id}")
            for pid in finished:
                active.pop(pid, None)
    except BaseException:
        terminate_active()
        raise

    atomic_csv_write(status_rows, status_path)
    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} training jobs failed in stage {stage}.")
    return counts


def stage_validate_data(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    output = Path(config["paths"]["run_root"]) / "data_validation"
    if dry_run:
        return {"would_validate": [config["data"]["small_root"], config["data"]["large_root"]]}
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, root_key in (("small", "small_root"), ("large", "large_root")):
        _, _, manifest = inspect_dataset(
            Path(config["data"][root_key]), config["data"]["split_directories"],
            config["task"]["targets"], config["task"]["target_ranges"],
            augmentation=bool(config["training"]["augmentation"]),
            require_square_for_augmentation=bool(config["data"]["require_square_for_augmentation"]),
            enforce_disjoint_files=bool(config["data"]["enforce_disjoint_files"]),
        )
        atomic_json_dump(manifest, output / f"{label}_manifest.json")
        results[label] = {
            "root": root_key,
            "fingerprint": manifest["dataset_fingerprint"],
            "sample_counts": {split: manifest["splits"][split]["total_samples"] for split in manifest["splits"]},
        }
    return results


def stage_grid_search(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    root = Path(config["paths"]["grid_root"])
    experiments = root / "experiments"
    job_configs = enumerate_grid_jobs(config)
    if len(job_configs) != configuration_count(config):
        raise RuntimeError("Internal grid count mismatch.")
    if dry_run:
        print(f"DRY RUN: grid_search would execute up to {len(job_configs)} runs.")
        return {"expected_runs": len(job_configs), "dry_run": True}
    jobs = prepare_jobs(job_configs, experiments)
    counts = run_training_jobs(
        config, jobs, "grid_search", root / "experiment_status.csv",
        bool(config["runtime"]["fail_fast_grid"]), dry_run,
    )
    if not dry_run:
        atomic_json_dump({
            "status": "completed", "completed_at_utc": utc_now(),
            "expected_runs": len(jobs), **counts,
        }, root / "COMPLETED.json")
    return {"expected_runs": len(jobs), **counts}


def invoke_selector(
    config: Mapping[str, Any], *, experiments_root: Path, output_dir: Path,
    selection: Mapping[str, Any], expected_runs: int, dry_run: bool,
) -> dict[str, Any]:
    selection_path = output_dir.parent / "selection_settings.json"
    if dry_run:
        return {"would_select_from": str(experiments_root), "expected_runs": expected_runs}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(selection, selection_path)
    command = [
        sys.executable, "-u", str(SELECT_SCRIPT),
        "--experiments-root", str(experiments_root),
        "--output-dir", str(output_dir),
        "--selection-json", str(selection_path),
        "--expected-runs", str(expected_runs),
        "--require-all" if bool(selection.get("require_all", True)) else "--no-require-all",
        "--overwrite",
    ]
    run_checked(command, Path(config["paths"]["logs"]) / f"select_{output_dir.parent.name}.log", cwd=SCRIPT_DIR)
    payload = load_json(output_dir / "selected_configurations.json")
    return {"selected_count": payload["selected_count"], "configuration_count": payload["configuration_count"]}


def stage_select_candidates(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    grid = Path(config["paths"]["grid_root"])
    return invoke_selector(
        config, experiments_root=grid / "experiments", output_dir=grid / "selection",
        selection=config["search"]["selection"], expected_runs=configuration_count(config), dry_run=dry_run,
    )


def stage_retrain_candidates(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    selection_path = Path(config["paths"]["grid_root"]) / "selection" / "selected_configurations.json"
    root = Path(config["paths"]["candidate_root"])
    repetitions = int(config["candidate_training"]["repetitions_per_configuration"])
    if dry_run and not selection_path.is_file():
        selection = config["search"]["selection"]
        groups = selection_group_count(config, selection)
        estimated = groups * int(selection.get("top_per_group", 1)) * repetitions
        print(f"DRY RUN: candidate retraining would execute approximately {estimated} runs.")
        return {"estimated_runs": estimated, "dry_run": True}
    jobs_config = jobs_from_selection(
        config, selection_path, stage="candidate_retraining",
        data_root=config["data"]["large_root"], repetitions=repetitions,
        seeds=config["candidate_training"]["seeds"], storage_profile="candidate",
    )
    jobs = prepare_jobs(jobs_config, root / "experiments")
    counts = run_training_jobs(
        config, jobs, "retrain_candidates", root / "experiment_status.csv",
        bool(config["runtime"]["fail_fast_other_stages"]), dry_run,
    )
    if not dry_run:
        atomic_json_dump({
            "status": "completed", "completed_at_utc": utc_now(),
            "expected_runs": len(jobs), **counts,
        }, root / "COMPLETED.json")
    return {"expected_runs": len(jobs), **counts}


def stage_select_final(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    candidate = Path(config["paths"]["candidate_root"])
    candidate_selection_path = Path(config["paths"]["grid_root"]) / "selection" / "selected_configurations.json"
    if dry_run and not candidate_selection_path.is_file():
        search_selection = config["search"]["selection"]
        groups = selection_group_count(config, search_selection)
        expected = groups * int(search_selection.get("top_per_group", 1)) * int(config["candidate_training"]["repetitions_per_configuration"])
    else:
        selected_candidates = load_json(candidate_selection_path)
        expected = int(selected_candidates["selected_count"]) * int(config["candidate_training"]["repetitions_per_configuration"])
    return invoke_selector(
        config, experiments_root=candidate / "experiments", output_dir=candidate / "selection",
        selection=config["candidate_training"]["selection"], expected_runs=expected, dry_run=dry_run,
    )


def stage_train_ensemble(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    final_selection = Path(config["paths"]["candidate_root"]) / "selection" / "selected_configurations.json"
    if dry_run and not final_selection.is_file():
        members = int(config["ensemble"]["members"])
        print(f"DRY RUN: final ensemble would train {members} members.")
        return {"expected_members": members, "dry_run": True}
    payload = load_json(final_selection)
    if not payload["selected_configurations"]:
        raise RuntimeError("Final selection contains no configuration.")
    # Always use exactly the best validation-ranked configuration for the final ensemble.
    best = payload["selected_configurations"][0]
    source = best["source_run_config"]
    members = int(config["ensemble"]["members"])
    jobs_config = []
    for repetition in range(members):
        jobs_config.append(build_run_config(
            config, stage="final_ensemble", data_root=config["data"]["large_root"],
            model=source["model"], optimization=source["optimization"],
            seed=int(config["ensemble"]["seeds"][repetition]), repetition=repetition,
            storage_profile="final", configuration_id=str(best["configuration_id"]),
        ))
    if not dry_run:
        Path(config["paths"]["ensemble_root"]).mkdir(parents=True, exist_ok=True)
        atomic_json_dump({**payload, "selected_configurations": [best]}, Path(config["paths"]["ensemble_root"]) / "final_configuration.json")
    # Keep normalization identical across ensemble members.
    for job in jobs_config:
        job["data"]["normalization"]["seed"] = int(config["ensemble"]["normalization_seed"])
        job["experiment_id"] = f"member_{job['repetition'] + 1:02d}_seed{job['seed']}"
    members_root = Path(config["paths"]["ensemble_root"]) / "members"
    jobs = prepare_jobs(jobs_config, members_root)
    counts = run_training_jobs(
        config, jobs, "train_ensemble", Path(config["paths"]["ensemble_root"]) / "member_status.csv",
        bool(config["runtime"]["fail_fast_other_stages"]), dry_run,
    )
    if not dry_run:
        atomic_json_dump({
            "status": "completed", "completed_at_utc": utc_now(),
            "expected_members": members, **counts,
        }, members_root / "COMPLETED.json")
    return {"expected_members": members, **counts}


def stage_evaluate_ensemble(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    root = Path(config["paths"]["ensemble_root"])
    output = root / "evaluation"
    command = [
        sys.executable, "-u", str(EVALUATE_SCRIPT),
        "--ensemble-root", str(root / "members"),
        "--output-dir", str(output),
        "--expected-members", str(config["ensemble"]["members"]),
        "--splits", "validation", "test",
        "--interval-levels", *[str(x) for x in config["ensemble"]["interval_levels"]],
        "--bootstrap-repetitions", str(config["ensemble"]["bootstrap_repetitions"]),
        "--write-per-image-csv" if config["storage"]["write_per_image_csv"] else "--no-write-per-image-csv",
        "--overwrite",
    ]
    if dry_run:
        print("DRY RUN:", command_string(command))
        return {"command": command}
    run_checked(command, Path(config["paths"]["logs"]) / "evaluate_ensemble.log", cwd=SCRIPT_DIR)
    return {"output": str(output)}


def stage_interpretability(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    root = Path(config["paths"]["ensemble_root"])
    top_output = root / "interpretability"
    layers = list(config["interpretability"]["embedding_layers"])
    commands = []
    for layer in layers:
        output = top_output / str(layer)
        command = [
            sys.executable, "-u", str(INTERPRET_SCRIPT),
            "--members-root", str(root / "members"),
            "--output-dir", str(output),
            "--expected-members", str(config["ensemble"]["members"]),
            "--embedding-layer", str(layer),
            "--fit-split", str(config["interpretability"]["fit_split"]),
            "--plot-split", str(config["interpretability"]["plot_split"]),
            "--batch-size", str(config["interpretability"]["batch_size"]),
            "--num-workers", str(config["training"]["num_workers"]),
            "--prefetch-factor", str(config["training"]["prefetch_factor"]),
            "--device", str(config["runtime"]["device"]),
            "--amp" if config["training"]["amp"] else "--no-amp",
            "--pca-components", str(config["interpretability"]["pca_components"]),
            "--pca-fit-max-samples", str(config["interpretability"]["pca_fit_max_samples"]),
            "--pca-backend", str(config["interpretability"]["pca_backend"]),
            "--pca-device", str(config["interpretability"]["pca_device"]),
            "--pca-oversamples", str(config["interpretability"]["pca_oversamples"]),
            "--pca-power-iterations", str(config["interpretability"]["pca_power_iterations"]),
            "--pca-cpu-batch-size", str(config["interpretability"]["pca_cpu_batch_size"]),
            "--pca-transform-batch-size", str(config["interpretability"]["pca_transform_batch_size"]),
            "--linear-algebra-threads", str(config["interpretability"]["linear_algebra_threads"]),
            "--pc-correlation-components", str(config["interpretability"]["pc_correlation_components"]),
            "--cca-components", str(config["interpretability"]["cca_components"]),
            "--cca-fit-max-samples", str(config["interpretability"]["cca_fit_max_samples"]),
            "--cca-max-iter", str(config["interpretability"]["cca_max_iter"]),
            "--cca-tol", str(config["interpretability"]["cca_tol"]),
            "--cca-pca-components", str(config["interpretability"]["cca_pca_components"]),
            "--cca-pca-variance", str(config["interpretability"]["cca_pca_variance"]),
            "--umap-policy", str(config["interpretability"]["umap"]["policy"]),
            "--umap-neighbors", str(config["interpretability"]["umap"]["neighbors"]),
            "--umap-min-dist", str(config["interpretability"]["umap"]["min_dist"]),
            "--umap-fit-max-samples", str(config["interpretability"]["umap"]["fit_max_samples"]),
            "--umap-transform-batch-size", str(config["interpretability"]["umap"]["transform_batch_size"]),
            "--umap-representations", str(config["interpretability"]["umap"]["representations"]),
            "--max-analysis-samples", str(config["interpretability"]["max_analysis_samples"]),
            "--max-plot-points", str(config["interpretability"]["max_plot_points"]),
            "--sample-seed", str(config["ensemble"]["normalization_seed"]),
            "--save-embeddings" if config["interpretability"]["save_embeddings"] else "--no-save-embeddings",
            "--consensus", "--overwrite",
        ]
        commands.append(command)
        if dry_run:
            print("DRY RUN:", command_string(command))
        else:
            env = os.environ.copy()
            for key, value in config["runtime"].get("environment", {}).items():
                env[str(key)] = str(value)
            if str(config["runtime"]["device"]) != "cpu":
                selected_gpu = wait_for_gpu_tokens(config, effective_gpu_tokens(config), required_count=1)[0]
                env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
                print(
                    f"[{utc_now()}] [GPU GUARD] interpretability layer={layer} "
                    f"using GPU token {selected_gpu}",
                    flush=True,
                )
            run_checked(
                command,
                Path(config["paths"]["logs"]) / f"interpretability_{layer}.log",
                env=env,
                cwd=SCRIPT_DIR,
            )
    if not dry_run:
        top_output.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({
            "status": "completed", "completed_at_utc": utc_now(), "layers": layers,
        }, top_output / "COMPLETED.json")
    return {"layers": layers, "command_count": len(commands)}


def stage_report(config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    output = Path(config["paths"]["report_root"])
    if dry_run:
        return {"would_write": str(output / "pipeline_report.md")}
    output.mkdir(parents=True, exist_ok=True)
    run_root = Path(config["paths"]["run_root"])
    stage_rows = []
    enabled_stages = set(config["stages"]["enabled"])
    for stage in STAGE_ORDER:
        marker = stage_marker(config, stage)
        if stage == "report":
            payload = {"status": "completed", "timestamp_utc": utc_now()}
        elif stage not in enabled_stages:
            payload = {"status": "disabled", "timestamp_utc": None}
        else:
            payload = load_json(marker) if marker.is_file() else {"status": "not_run"}
        stage_rows.append({"stage": stage, "status": payload.get("status"), "timestamp_utc": payload.get("timestamp_utc")})
    atomic_csv_write(stage_rows, output / "stage_summary.csv")
    size_rows = []
    for name, path in (
        ("grid_search", Path(config["paths"]["grid_root"])),
        ("candidate_retraining", Path(config["paths"]["candidate_root"])),
        ("ensemble", Path(config["paths"]["ensemble_root"])),
    ):
        size = directory_size_bytes(path)
        size_rows.append({"component": name, "path": str(path), "size_bytes": size, "size_human": format_bytes(size)})
    atomic_csv_write(size_rows, output / "storage_summary.csv")

    lines = [
        f"# Pipeline report: {config['experiment']['name']} / {config['experiment']['version']}", "",
        f"Generated: {utc_now()}", "",
        f"Task mode: `{config['task']['mode']}`", "",
        f"Targets: `{', '.join(config['task']['targets'])}`", "",
        f"Resolved configuration hash: `{config['_meta']['resolved_config_hash']}`", "",
        "## Stage status", "",
        "| Stage | Status | Timestamp |", "|---|---|---|",
    ]
    for row in stage_rows:
        lines.append(f"| {row['stage']} | {row['status']} | {row['timestamp_utc'] or ''} |")
    lines += ["", "## Storage", "", "| Component | Size | Path |", "|---|---:|---|"]
    for row in size_rows:
        lines.append(f"| {row['component']} | {row['size_human']} | `{row['path']}` |")
    grid_count = configuration_count(config)
    search_selection = config["search"]["selection"]
    groups = selection_group_count(config, search_selection)
    candidate_config_count = groups * int(search_selection.get("top_per_group", 1))
    candidate_run_count = candidate_config_count * int(config["candidate_training"]["repetitions_per_configuration"])
    lines += [
        "", "## Experimental design", "",
        f"- Grid runs: **{grid_count}**",
        f"- Candidate configurations retained: **{candidate_config_count}**",
        f"- Large-data candidate runs: **{candidate_run_count}**",
        f"- Final ensemble members: **{config['ensemble']['members']}**",
        f"- Grid learning rates: `{config['search']['space']['learning_rate']}`",
        f"- Grid batch sizes: `{config['search']['space']['batch_size']}`",
        f"- Grid pretrained choices: `{config['model']['pretrained']}`",
        "", "## Selection policy", "",
        "Hyperparameter and final-configuration selection use validation metrics only. Test metrics are produced only for final held-out reporting and are not used for ranking.",
        f"Selection metric: `{search_selection['metric']}`; target: `{search_selection['target']}`; space: `{search_selection['space']}`; scaling: `{search_selection['scaling']}`.",
    ]
    final_selection_path = Path(config["paths"]["candidate_root"]) / "selection" / "selected_configurations.json"
    if final_selection_path.is_file():
        with contextlib.suppress(Exception):
            final_payload = load_json(final_selection_path)
            if final_payload.get("selected_configurations"):
                best = final_payload["selected_configurations"][0]
                lines += [
                    "", "## Final selected configuration", "",
                    f"Configuration ID: `{best.get('configuration_id')}`",
                    f"Validation selection score: `{best.get('validation_selection_score')}`",
                    "", "```json",
                    json.dumps({"model": best.get("model"), "optimization": best.get("optimization")}, indent=2, sort_keys=True),
                    "```",
                ]
    lines += [
        "", "## Main final outputs", "",
        f"- Final selection: `{final_selection_path}`",
        f"- Ensemble evaluation: `{Path(config['paths']['ensemble_root']) / 'evaluation' / 'FINAL_ENSEMBLE_EVALUATION.json'}`",
        f"- Ensemble test predictions: `{Path(config['paths']['ensemble_root']) / 'evaluation' / 'predictions' / 'test_ensemble_predictions.npz'}`",
        f"- Resolved configuration: `{run_root / 'resolved_config.yaml'}`",
        f"- Environment capture: `{run_root / 'environment.json'}`",
        f"- Source provenance: `{run_root / 'source_provenance.json'}`",
    ]
    atomic_text_write("\n".join(lines) + "\n", output / "pipeline_report.md")
    report_json: dict[str, Any] = {
        "status": "completed",
        "generated_utc": utc_now(),
        "configuration_hash": config["_meta"]["resolved_config_hash"],
        "task": config["task"],
        "experimental_design": {
            "grid_runs": grid_count,
            "candidate_configurations": candidate_config_count,
            "candidate_runs": candidate_run_count,
            "ensemble_members": int(config["ensemble"]["members"]),
        },
        "selection": search_selection,
        "stages": stage_rows,
        "storage": size_rows,
    }
    if final_selection_path.is_file():
        with contextlib.suppress(Exception):
            final_payload = load_json(final_selection_path)
            if final_payload.get("selected_configurations"):
                report_json["final_selected_configuration"] = final_payload["selected_configurations"][0]
    atomic_json_dump(report_json, output / "pipeline_report.json")
    return {"report": str(output / "pipeline_report.md")}


STAGE_FUNCTIONS: dict[str, Callable[[Mapping[str, Any], bool], dict[str, Any]]] = {
    "validate_data": stage_validate_data,
    "grid_search": stage_grid_search,
    "select_candidates": stage_select_candidates,
    "retrain_candidates": stage_retrain_candidates,
    "select_final": stage_select_final,
    "train_ensemble": stage_train_ensemble,
    "evaluate_ensemble": stage_evaluate_ensemble,
    "interpretability": stage_interpretability,
    "report": stage_report,
}


def source_provenance() -> dict[str, Any]:
    """Fingerprint executable source/configuration files for run provenance."""
    included_suffixes = {".py", ".sh", ".yaml", ".toml"}
    included_names = {
        "VERSION.txt", "requirements.txt", "requirements-dev.txt", ".gitignore"
    }
    checksums: dict[str, str] = {}
    for path in sorted(SCRIPT_DIR.rglob("*"), key=lambda item: str(item.relative_to(SCRIPT_DIR))):
        if not path.is_file():
            continue
        relative = path.relative_to(SCRIPT_DIR)
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts):
            continue
        if path.suffix not in included_suffixes and path.name not in included_names:
            continue
        checksums[str(relative)] = file_sha256(path)
    version_path = SCRIPT_DIR / "VERSION.txt"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else "unknown"
    return {
        "captured_utc": utc_now(),
        "release_version": version,
        "source_tree_sha256": object_sha256(checksums),
        "file_sha256": checksums,
    }


def initialize_run(config: Mapping[str, Any]) -> None:
    run_root = Path(config["paths"]["run_root"])
    state = Path(config["paths"]["stage_state"])
    logs = Path(config["paths"]["logs"])
    run_root.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    hash_path = run_root / "resolved_config_hash.txt"
    config_hash = str(config["_meta"]["resolved_config_hash"])
    if hash_path.is_file():
        previous = hash_path.read_text(encoding="utf-8").strip()
        if previous != config_hash and not bool(config["experiment"]["allow_config_change"]):
            raise ValueError(
                f"Run root already contains a different resolved configuration hash.\n"
                f"run_root={run_root}\nprevious={previous}\ncurrent={config_hash}\n"
                "Use a new experiment.version for changed scientific settings."
            )
    atomic_text_write(config_hash + "\n", hash_path)
    atomic_json_dump(config, run_root / "resolved_config.json")
    atomic_yaml_dump(config, run_root / "resolved_config.yaml")
    environment_path = run_root / "environment.json"
    if not environment_path.is_file():
        atomic_json_dump(capture_environment(include_pip_freeze=True), environment_path)
    provenance_path = run_root / "source_provenance.json"
    if not provenance_path.is_file():
        atomic_json_dump(source_provenance(), provenance_path)


def main() -> None:
    def handle_termination(signum: int, frame: Any) -> None:
        del frame
        print(f"[{utc_now()}] Received signal {signum}; stopping safely.", flush=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)
    args = build_parser().parse_args()
    config = load_and_resolve_config(args.config)
    initialize_run(config)
    requested = args.stages.split(",") if args.stages else None
    stages = selected_stage_window(config, requested, args.from_stage, args.to_stage)
    stages = add_missing_dependencies(config, stages, args.no_auto_dependencies)
    if args.print_plan or args.dry_run:
        print("Resolved run root:", config["paths"]["run_root"])
        print("Configuration hash:", config["_meta"]["resolved_config_hash"])
        print("Stage plan:", " -> ".join(stages))
        print("Grid runs:", configuration_count(config))
    lock_path = Path(config["paths"]["run_root"]) / ".pipeline.lock"
    pipeline_start = time.time()
    pipeline_marker = Path(config["paths"]["stage_state"]) / "pipeline.json"
    with ExclusiveFileLock(lock_path):
        atomic_json_dump({
            "status": "running", "started_at_utc": utc_now(), "pid": os.getpid(),
            "stages": stages, "config_hash": config["_meta"]["resolved_config_hash"],
        }, pipeline_marker)
        try:
            for stage in stages:
                if stage_output_valid(config, stage) and stage not in args.force_stage:
                    print(f"[{utc_now()}] SKIP stage {stage}: valid completed output exists.", flush=True)
                    continue
                # Check dependencies that are outside the selected plan or were skipped.
                for dependency in STAGE_DEPENDENCIES[stage]:
                    if dependency not in stages and not stage_output_valid(config, dependency):
                        raise RuntimeError(f"Stage {stage} requires valid output from {dependency}.")
                print(f"[{utc_now()}] START stage {stage}", flush=True)
                if not args.dry_run:
                    write_stage_marker(config, stage, "running", started_at_utc=utc_now(), pid=os.getpid())
                stage_start = time.time()
                try:
                    details = STAGE_FUNCTIONS[stage](config, args.dry_run)
                except BaseException as exc:
                    if not args.dry_run:
                        write_stage_marker(
                            config, stage, "failed", failed_at_utc=utc_now(),
                            elapsed_seconds=time.time() - stage_start,
                            exception_type=type(exc).__name__, message=str(exc),
                        )
                    raise
                if not args.dry_run:
                    write_stage_marker(
                        config, stage, "completed", completed_at_utc=utc_now(),
                        elapsed_seconds=time.time() - stage_start, details=details,
                    )
                print(f"[{utc_now()}] END stage {stage} elapsed={elapsed_text(time.time() - stage_start)}", flush=True)
            atomic_json_dump({
                "status": "completed", "completed_at_utc": utc_now(),
                "elapsed_seconds": time.time() - pipeline_start, "stages": stages,
                "config_hash": config["_meta"]["resolved_config_hash"],
            }, pipeline_marker)
            print(f"Pipeline completed in {elapsed_text(time.time() - pipeline_start)}", flush=True)
        except BaseException as exc:
            atomic_json_dump({
                "status": "failed", "failed_at_utc": utc_now(),
                "elapsed_seconds": time.time() - pipeline_start,
                "exception_type": type(exc).__name__, "message": str(exc),
                "stages": stages, "config_hash": config["_meta"]["resolved_config_hash"],
            }, pipeline_marker)
            if isinstance(exc, KeyboardInterrupt):
                raise SystemExit(130)
            raise


if __name__ == "__main__":
    main()
