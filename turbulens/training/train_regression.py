#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pipeline_lib.common import (
    atomic_csv_write,
    atomic_json_dump,
    atomic_npz_save,
    capture_environment,
    elapsed_text,
    file_sha256,
    load_json,
    object_sha256,
    to_jsonable,
    utc_now,
    valid_status_marker,
)
from pipeline_lib.data import (
    EpochRandomSampler,
    MultiFileHDF5Dataset,
    NormalizationStats,
    denormalize_targets,
    estimate_normalization,
    inspect_dataset,
    load_target_matrix,
    normalize_targets,
    worker_init_fn,
)
from pipeline_lib.metrics import compute_full_metrics
from pipeline_lib.model import build_model_from_config, parameter_counts

COMPLETION_MARKER = "COMPLETED.json"
FAILURE_MARKER = "FAILED.json"
CHECKPOINT_FORMAT_VERSION = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one multitask or single-target ResNet regression experiment from a resolved run configuration."
    )
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def setup_logger(output_dir: Path, level: str = "INFO") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"train.{output_dir.name}.{os.getpid()}")
    logger.setLevel(getattr(logging, str(level).upper()))
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "training.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def validate_run_config(config: Mapping[str, Any]) -> None:
    required = {"task", "data", "model", "optimization", "training", "runtime", "checkpointing", "evaluation", "storage", "logging", "seed"}
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"Run configuration is missing keys: {missing}")
    targets = list(config["task"]["targets"])
    if not targets:
        raise ValueError("At least one target is required.")
    if config["task"]["mode"] == "single_target" and len(targets) != 1:
        raise ValueError("single_target mode requires exactly one target.")
    if config["task"]["mode"] == "multitask" and len(targets) < 2:
        raise ValueError("multitask mode requires at least two targets.")
    optimization = config["optimization"]
    if int(optimization["batch_size"]) < 1 or int(config["training"]["epochs"]) < 1:
        raise ValueError("batch_size and epochs must be positive.")
    if float(optimization["learning_rate"]) <= 0 or float(optimization["weight_decay"]) < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    evaluation = config["evaluation"]
    for key in ("evaluate_validation", "evaluate_test"):
        if not isinstance(evaluation.get(key), bool):
            raise TypeError(f"evaluation.{key} must be boolean.")
    if not evaluation["evaluate_validation"]:
        raise ValueError("Validation evaluation is required for checkpoint selection.")
    normalization = config["data"]["normalization"]
    lower, upper = float(normalization["lower_percentile"]), float(normalization["upper_percentile"])
    if not 0 <= lower < upper <= 100:
        raise ValueError("Invalid normalization percentiles.")


def set_reproducibility(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark and not deterministic)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda:0")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temp)
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def verify_checkpoint(path: Path, expected_hash: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint {path} is not a mapping.")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}.")
    if payload.get("run_config_hash") != expected_hash:
        raise ValueError(f"Checkpoint configuration hash mismatch in {path}.")
    required = {"model_state", "epoch", "normalization_stats", "data_fingerprint"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"Checkpoint {path} is missing keys: {missing}")
    return dict(payload)


def make_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)  # type: ignore[attr-defined]


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled)


def create_loader(
    dataset: MultiFileHDF5Dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    sampler: EpochRandomSampler | None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**kwargs)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    gradient_clip_norm: float,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total = 0.0
    count = 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} train", leave=False)
    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            prediction = model(images)
            loss = criterion(prediction, targets)
        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch = int(images.shape[0])
        total += float(loss.detach().cpu()) * batch
        count += batch
        progress.set_postfix(loss=f"{total / max(count, 1):.5f}")
    if count == 0:
        raise RuntimeError("Training loader produced no samples.")
    return total / count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    targets: Sequence[str],
    target_ranges: Mapping[str, Sequence[float]],
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    loss_total = 0.0
    count = 0
    target_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels, indices in tqdm(loader, desc=f"Evaluate {split_name}", leave=False):
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            with autocast_context(device, amp):
                prediction = model(images)
                loss = criterion(prediction, labels_device)
            batch = int(images.shape[0])
            loss_total += float(loss.detach().cpu()) * batch
            count += batch
            target_parts.append(labels.numpy().astype(np.float32, copy=False))
            prediction_parts.append(prediction.float().cpu().numpy().astype(np.float32, copy=False))
            index_parts.append(np.asarray(indices, dtype=np.int64))
    seconds = time.perf_counter() - start
    if count == 0:
        raise RuntimeError(f"{split_name} loader produced no samples.")
    target_normalized = np.concatenate(target_parts)
    prediction_raw = np.concatenate(prediction_parts)
    sample_index = np.concatenate(index_parts)
    order = np.argsort(sample_index)
    sample_index = sample_index[order]
    target_normalized = target_normalized[order]
    prediction_raw = prediction_raw[order]
    prediction_clamped = np.clip(prediction_raw, 0.0, 1.0)
    target_physical = denormalize_targets(target_normalized, targets, target_ranges, clamp=False)
    prediction_physical = denormalize_targets(prediction_clamped, targets, target_ranges, clamp=False)
    metrics = compute_full_metrics(
        target_normalized, prediction_raw, prediction_clamped,
        target_physical, prediction_physical, targets, loss_total / count,
    )
    metrics["inference"] = {
        "seconds": float(seconds),
        "samples_per_second": float(count / max(seconds, 1e-12)),
        "milliseconds_per_sample": float(1000.0 * seconds / count),
    }
    return {
        "loss": loss_total / count,
        "sample_index": sample_index,
        "target_normalized": target_normalized,
        "prediction_normalized_raw": prediction_raw,
        "prediction_normalized_clamped": prediction_clamped,
        "target_physical": target_physical,
        "prediction_physical": prediction_physical,
        "metrics": metrics,
    }


def save_history(history: Mapping[str, Sequence[float]], output_dir: Path) -> None:
    atomic_json_dump(history, output_dir / "history" / "history.json")
    rows: list[dict[str, Any]] = []
    count = len(history.get("train_loss", []))
    for index in range(count):
        rows.append({key: values[index] for key, values in history.items()})
    atomic_csv_write(rows, output_dir / "history" / "history.csv")


def save_predictions(
    result: Mapping[str, Any], split: str, output_dir: Path, targets: Sequence[str],
    sample_keys: np.ndarray, dataset_fingerprint: str,
) -> None:
    indices = np.asarray(result["sample_index"], dtype=np.int64)
    atomic_npz_save(
        output_dir / "predictions" / f"{split}_predictions.npz",
        sample_index=indices,
        sample_key=sample_keys[indices],
        target_names=np.asarray(targets, dtype=str),
        dataset_fingerprint=np.asarray(dataset_fingerprint),
        target_normalized=result["target_normalized"],
        prediction_normalized_raw=result["prediction_normalized_raw"],
        prediction_normalized_clamped=result["prediction_normalized_clamped"],
        target_physical=result["target_physical"],
        prediction_physical=result["prediction_physical"],
    )
    atomic_json_dump(result["metrics"], output_dir / "metrics" / f"{split}_metrics.json")


def plot_results(result: Mapping[str, Any], split: str, output_dir: Path, targets: Sequence[str]) -> None:
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    truth = np.asarray(result["target_physical"])
    prediction = np.asarray(result["prediction_physical"])
    for column, target in enumerate(targets):
        metric = result["metrics"]["per_target"][target]["physical"]
        low = float(min(truth[:, column].min(), prediction[:, column].min()))
        high = float(max(truth[:, column].max(), prediction[:, column].max()))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(truth[:, column], prediction[:, column], s=6, alpha=0.18, rasterized=True)
        ax.plot([low, high], [low, high], "--", linewidth=1.2)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(
            f"{split}: {target}\nR2={metric['r2']:.4f}, MAE={metric['mae']:.4g}, RMSE={metric['rmse']:.4g}"
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"{split}_{target}_prediction_vs_truth.png", dpi=180)
        plt.close(fig)

        residual = prediction[:, column] - truth[:, column]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(truth[:, column], residual, s=6, alpha=0.18, rasterized=True)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel("Prediction - truth")
        ax.set_title(f"{split}: {target} residuals")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"{split}_{target}_residuals.png", dpi=180)
        plt.close(fig)


def plot_history(history: Mapping[str, Sequence[float]], output_dir: Path, targets: Sequence[str]) -> None:
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["validation_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "learning_curves.png", dpi=180)
    plt.close(fig)


def build_checkpoint(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    history: Mapping[str, Sequence[float]],
    run_config: Mapping[str, Any],
    run_config_hash: str,
    normalization: NormalizationStats,
    data_fingerprint: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_validation_loss": float(best_validation_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": to_jsonable(history),
        "run_config": to_jsonable(run_config),
        "run_config_hash": run_config_hash,
        "normalization_stats": to_jsonable(normalization),
        "data_fingerprint": data_fingerprint,
        "timestamp_utc": utc_now(),
        "torch_version": torch.__version__,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    run_config = load_json(args.run_config.expanduser().resolve())
    validate_run_config(run_config)
    run_config_hash = object_sha256(run_config)

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, str(run_config["logging"]["level"]))
    start_time = time.time()

    try:
        if valid_status_marker(output_dir / COMPLETION_MARKER):
            saved_config = output_dir / "configuration.json"
            if saved_config.is_file() and object_sha256(load_json(saved_config)) == run_config_hash:
                logger.info("Valid completed experiment already exists; skipping: %s", output_dir)
                return
            raise FileExistsError(f"Completed output exists with a different configuration: {output_dir}")

        config_path = output_dir / "configuration.json"
        if config_path.is_file():
            saved_hash = object_sha256(load_json(config_path))
            if saved_hash != run_config_hash:
                raise ValueError(
                    f"Existing experiment configuration differs from requested configuration: {output_dir}"
                )
        else:
            existing = [p for p in output_dir.iterdir() if p.name not in {"training.log"}]
            if existing and args.resume == "never":
                raise FileExistsError(f"Non-empty output directory cannot be overwritten: {output_dir}")
            atomic_json_dump(run_config, config_path)

        targets = list(run_config["task"]["targets"])
        target_ranges = run_config["task"]["target_ranges"]
        data_config = run_config["data"]
        training_config = run_config["training"]
        optimization = run_config["optimization"]
        runtime = run_config["runtime"]
        checkpointing = run_config["checkpointing"]
        evaluation_config = run_config["evaluation"]
        storage = run_config["storage"]
        seed = int(run_config["seed"])
        deterministic = bool(training_config["deterministic"])
        cudnn_benchmark = bool(training_config["cudnn_benchmark"])
        set_reproducibility(seed, deterministic, cudnn_benchmark)
        device = resolve_device(str(runtime["device"]))
        amp_enabled = bool(training_config["amp"] and device.type == "cuda")
        logger.info("Device=%s AMP=%s seed=%d targets=%s", device, amp_enabled, seed, targets)

        data_root = Path(data_config["root"]).expanduser().resolve()
        files, infos, manifest = inspect_dataset(
            data_root,
            data_config["split_directories"],
            targets,
            target_ranges,
            augmentation=bool(training_config["augmentation"]),
            require_square_for_augmentation=bool(data_config["require_square_for_augmentation"]),
            enforce_disjoint_files=bool(data_config.get("enforce_disjoint_files", True)),
        )
        data_fingerprint = str(manifest["dataset_fingerprint"])
        manifest_path = output_dir / "data_manifest.json"
        if manifest_path.is_file():
            saved = load_json(manifest_path)
            saved_fingerprint = str(saved.get("dataset_fingerprint", ""))
            if not saved_fingerprint or saved_fingerprint != data_fingerprint:
                raise ValueError("Dataset manifest changed; refusing to resume or reuse the experiment.")
        else:
            atomic_json_dump(manifest, manifest_path)

        split_indices = {
            split: np.arange(sum(item.n_samples for item in infos[split]), dtype=np.int64)
            for split in ("train", "validation", "test")
        }
        split_file = output_dir / "split_indices.npz"
        if split_file.is_file():
            with np.load(split_file) as loaded:
                for split, expected in split_indices.items():
                    if split not in loaded or not np.array_equal(loaded[split], expected):
                        raise ValueError("Saved split indices do not match the current dataset.")
        else:
            atomic_npz_save(split_file, **split_indices)

        labels_physical = {split: load_target_matrix(files[split], targets) for split in files}
        labels_normalized = {
            split: normalize_targets(values, targets, target_ranges)
            for split, values in labels_physical.items()
        }

        checkpoint_dir = output_dir / "checkpoints"
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        best_path = checkpoint_dir / "best_checkpoint.pt"
        resume_path: Path | None = None
        if args.resume != "never" and latest_path.is_file():
            resume_path = latest_path
        elif args.resume == "required":
            raise FileNotFoundError(f"Resume required but latest checkpoint is missing: {latest_path}")

        if resume_path:
            checkpoint = verify_checkpoint(resume_path, run_config_hash)
            if checkpoint["data_fingerprint"] != data_fingerprint:
                raise ValueError("Resume checkpoint was created from a different dataset manifest.")
            normalization = NormalizationStats(**checkpoint["normalization_stats"])
        else:
            norm_config = data_config["normalization"]
            normalization = estimate_normalization(
                files["train"], infos["train"],
                float(norm_config["lower_percentile"]), float(norm_config["upper_percentile"]),
                int(norm_config["max_images"]), int(norm_config["max_pixels"]),
                int(norm_config["seed"]), float(norm_config["epsilon"]), logger,
            )
        atomic_json_dump(normalization, output_dir / "normalization.json")

        shared_epoch = mp.Value("i", 0)
        model_config = run_config["model"]
        datasets = {
            split: MultiFileHDF5Dataset(
                files[split], infos[split], labels_normalized[split], normalization,
                in_channels=int(model_config["in_channels"]),
                input_standardization=str(model_config["input_standardization"]),
                augment=bool(training_config["augmentation"] and split == "train"),
                augmentation_seed=seed,
                shared_epoch=shared_epoch,
            )
            for split in ("train", "validation", "test")
        }
        train_sampler = EpochRandomSampler(datasets["train"], seed)
        loaders = {
            split: create_loader(
                datasets[split],
                batch_size=int(optimization["batch_size"]),
                num_workers=int(training_config["num_workers"]),
                prefetch_factor=int(training_config["prefetch_factor"]),
                pin_memory=bool(training_config["pin_memory"] and device.type == "cuda"),
                persistent_workers=bool(training_config["persistent_workers"]),
                sampler=train_sampler if split == "train" else None,
            )
            for split in ("train", "validation", "test")
        }

        model = build_model_from_config(run_config).to(device)
        counts = parameter_counts(model)
        loss_name = str(optimization["loss"])
        if loss_name != "smooth_l1":
            raise ValueError(f"Unsupported loss: {loss_name}")
        criterion = nn.SmoothL1Loss(
            beta=float(optimization["smooth_l1_beta"]), reduction="mean"
        ).to(device)
        optimizer_config = training_config["optimizer"]
        if str(optimizer_config["name"]) != "adamw":
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']!r}")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            betas=tuple(float(x) for x in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
            amsgrad=bool(optimizer_config["amsgrad"]),
        )
        scheduler_config = training_config["scheduler"]
        if str(scheduler_config["name"]) != "reduce_on_plateau":
            raise ValueError(f"Unsupported scheduler: {scheduler_config['name']!r}")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config["min_lr"]),
            threshold=float(scheduler_config["threshold"]),
            threshold_mode=str(scheduler_config["threshold_mode"]),
            cooldown=int(scheduler_config["cooldown"]),
            eps=float(scheduler_config["eps"]),
        )
        scaler = make_scaler(amp_enabled)

        history: dict[str, list[float]] = {
            "epoch": [], "train_loss": [], "validation_loss": [], "learning_rate": [],
            "epoch_seconds": [],
        }
        for target in targets:
            history[f"validation_normalized_mae_{target}"] = []
            history[f"validation_physical_mae_{target}"] = []
        start_epoch = 0
        best_validation_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        if resume_path:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            history = {key: [float(x) for x in values] for key, values in checkpoint["history"].items()}
            start_epoch = int(checkpoint["epoch"]) + 1
            best_validation_loss = float(checkpoint["best_validation_loss"])
            epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
            if history.get("validation_loss"):
                best_epoch = int(np.argmin(history["validation_loss"]))
            logger.info("Resuming from epoch %d using %s", start_epoch + 1, resume_path)

        interrupted = False
        previous_handlers: dict[int, Any] = {}

        def handle_signal(signum: int, frame: Any) -> None:
            nonlocal interrupted
            interrupted = True
            logger.warning("Received signal %s; stopping after the current batch/epoch checkpoint.", signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)

        epochs = int(training_config["epochs"])
        early = training_config["early_stopping"]
        training_start = time.time()
        for epoch in range(start_epoch, epochs):
            if interrupted:
                raise KeyboardInterrupt("Interrupted before epoch start.")
            shared_epoch.value = epoch
            train_sampler.set_epoch(epoch)
            epoch_start = time.time()
            train_loss = train_epoch(
                model, loaders["train"], criterion, optimizer, scaler, device, amp_enabled,
                float(training_config["gradient_clip_norm"]), epoch, epochs,
            )
            validation = evaluate(
                model, loaders["validation"], criterion, device, amp_enabled,
                targets, target_ranges, "validation",
            )
            scheduler.step(float(validation["loss"]))
            current_lr = float(optimizer.param_groups[0]["lr"])
            history["epoch"].append(float(epoch + 1))
            history["train_loss"].append(float(train_loss))
            history["validation_loss"].append(float(validation["loss"]))
            history["learning_rate"].append(current_lr)
            history["epoch_seconds"].append(float(time.time() - epoch_start))
            for target in targets:
                row = validation["metrics"]["per_target"][target]
                history[f"validation_normalized_mae_{target}"].append(float(row["normalized_clamped"]["mae"]))
                history[f"validation_physical_mae_{target}"].append(float(row["physical"]["mae"]))

            improved = float(validation["loss"]) < best_validation_loss - float(early["min_delta"])
            if improved:
                best_validation_loss = float(validation["loss"])
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            checkpoint_payload = build_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                history=history, run_config=run_config, run_config_hash=run_config_hash,
                normalization=normalization, data_fingerprint=data_fingerprint,
            )
            if (epoch + 1) % int(checkpointing["every_n_epochs"]) == 0 or interrupted:
                atomic_torch_save(checkpoint_payload, latest_path)
            if improved:
                atomic_torch_save(checkpoint_payload, best_path)
                if bool(checkpointing["verify_after_write"]):
                    verify_checkpoint(best_path, run_config_hash)
            save_history(history, output_dir)
            logger.info(
                "Epoch %d/%d train=%.6f val=%.6f lr=%.3e best_epoch=%d elapsed=%s",
                epoch + 1, epochs, train_loss, validation["loss"], current_lr,
                best_epoch + 1, elapsed_text(time.time() - epoch_start),
            )
            if interrupted:
                raise KeyboardInterrupt("Interrupted after checkpoint save.")
            if int(early["patience"]) > 0 and epochs_without_improvement >= int(early["patience"]):
                logger.info("Early stopping after %d epochs without improvement.", epochs_without_improvement)
                break

        training_seconds = time.time() - training_start
        if not best_path.is_file():
            raise RuntimeError("Training finished without a best checkpoint.")
        best_checkpoint = verify_checkpoint(best_path, run_config_hash)
        model.load_state_dict(best_checkpoint["model_state"], strict=True)
        best_epoch = int(best_checkpoint["epoch"])
        best_validation_loss = float(best_checkpoint["best_validation_loss"])

        validation_final = evaluate(
            model, loaders["validation"], criterion, device, amp_enabled,
            targets, target_ranges, "validation-final",
        )
        test_final = None
        if bool(evaluation_config["evaluate_test"]):
            test_final = evaluate(
                model, loaders["test"], criterion, device, amp_enabled,
                targets, target_ranges, "test",
            )

        if bool(storage["save_predictions"]):
            save_predictions(
                validation_final, "validation", output_dir, targets,
                datasets["validation"].all_sample_keys(), data_fingerprint,
            )
            if test_final is not None:
                save_predictions(
                    test_final, "test", output_dir, targets,
                    datasets["test"].all_sample_keys(), data_fingerprint,
                )
        else:
            atomic_json_dump(validation_final["metrics"], output_dir / "metrics" / "validation_metrics.json")
            if test_final is not None:
                atomic_json_dump(test_final["metrics"], output_dir / "metrics" / "test_metrics.json")
        if bool(storage["save_plots"]):
            plot_history(history, output_dir, targets)
            plot_results(validation_final, "validation", output_dir, targets)
            if test_final is not None:
                plot_results(test_final, "test", output_dir, targets)
        if bool(storage["save_environment"]):
            atomic_json_dump(capture_environment(include_pip_freeze=True), output_dir / "environment.json")

        total_seconds = time.time() - start_time

        # Completed final models only need an inference checkpoint. Optimizer,
        # scheduler, scaler, and duplicated history states are retained in the
        # resumable latest checkpoint while a run is incomplete, but are removed
        # from the successful best checkpoint to control long-term storage.
        if bool(storage["keep_best_checkpoint"]):
            compact_best_checkpoint = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "checkpoint_kind": "inference",
                "epoch": best_epoch,
                "model_state": best_checkpoint["model_state"],
                "best_validation_loss": best_validation_loss,
                "run_config": to_jsonable(run_config),
                "run_config_hash": run_config_hash,
                "normalization_stats": to_jsonable(normalization),
                "data_fingerprint": data_fingerprint,
                "target_ranges": to_jsonable(target_ranges),
                "model_config": model.model_config(),
                "timestamp_utc": utc_now(),
                "torch_version": torch.__version__,
            }
            atomic_torch_save(compact_best_checkpoint, best_path)
            if bool(checkpointing["verify_after_write"]):
                verify_checkpoint(best_path, run_config_hash)

        summary = {
            "status": "completed",
            "experiment_id": run_config.get("experiment_id"),
            "configuration_id": run_config.get("configuration_id"),
            "stage": run_config.get("stage"),
            "task_mode": run_config["task"]["mode"],
            "targets": targets,
            "seed": seed,
            "repetition": int(run_config.get("repetition", 0)),
            "best_epoch_one_based": best_epoch + 1,
            "best_validation_loss": best_validation_loss,
            "training_runtime_seconds": training_seconds,
            "total_runtime_seconds": total_seconds,
            "validation": validation_final["metrics"],
            "test": test_final["metrics"] if test_final is not None else None,
            "parameter_count": counts,
            "model_config": model.model_config(),
            "optimization": optimization,
            "normalization_stats": to_jsonable(normalization),
            "data_fingerprint": data_fingerprint,
            "run_config_hash": run_config_hash,
            "checkpoint_sha256": file_sha256(best_path),
        }
        atomic_json_dump(summary, output_dir / "summary.json")
        summary_rows: list[dict[str, Any]] = []
        evaluated_splits = [("validation", validation_final)]
        if test_final is not None:
            evaluated_splits.append(("test", test_final))
        for split_name, result in evaluated_splits:
            for target in targets:
                target_metrics = result["metrics"]["per_target"][target]
                summary_rows.append({
                    "split": split_name,
                    "target": target,
                    "loss": result["loss"],
                    "normalized_mae": target_metrics["normalized_clamped"]["mae"],
                    "normalized_mse": target_metrics["normalized_clamped"]["mse"],
                    "normalized_rmse": target_metrics["normalized_clamped"]["rmse"],
                    "normalized_r2": target_metrics["normalized_clamped"]["r2"],
                    "physical_mae": target_metrics["physical"]["mae"],
                    "physical_mse": target_metrics["physical"]["mse"],
                    "physical_rmse": target_metrics["physical"]["rmse"],
                    "physical_r2": target_metrics["physical"]["r2"],
                    "physical_pearson": target_metrics["physical"]["pearson"],
                    "physical_spearman": target_metrics["physical"]["spearman"],
                    "physical_bias": target_metrics["physical"]["bias"],
                })
        atomic_csv_write(summary_rows, output_dir / "summary.csv")

        retained: list[str] = []
        if not bool(storage["keep_latest_checkpoint"]):
            with contextlib.suppress(FileNotFoundError):
                latest_path.unlink()
        elif latest_path.is_file():
            retained.append(str(latest_path.relative_to(output_dir)))
        if not bool(storage["keep_best_checkpoint"]):
            with contextlib.suppress(FileNotFoundError):
                best_path.unlink()
        elif best_path.is_file():
            retained.append(str(best_path.relative_to(output_dir)))
        if checkpoint_dir.is_dir() and not any(checkpoint_dir.iterdir()):
            checkpoint_dir.rmdir()

        completion = {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "best_epoch_one_based": best_epoch + 1,
            "best_validation_loss": best_validation_loss,
            "training_runtime_seconds": training_seconds,
            "total_runtime_seconds": total_seconds,
            "retained_checkpoints": retained,
            "summary_file": "summary.json",
            "run_config_hash": run_config_hash,
            "data_fingerprint": data_fingerprint,
        }
        atomic_json_dump(completion, output_dir / COMPLETION_MARKER)
        with contextlib.suppress(FileNotFoundError):
            (output_dir / FAILURE_MARKER).unlink()
        logger.info("Completed successfully in %s", elapsed_text(total_seconds))

    except BaseException as exc:
        failure = {
            "status": "failed",
            "failed_at_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "runtime_seconds": time.time() - start_time,
            "run_config_hash": run_config_hash,
        }
        with contextlib.suppress(Exception):
            atomic_json_dump(failure, output_dir / FAILURE_MARKER)
        logger.exception("Experiment failed: %s", exc)
        if isinstance(exc, KeyboardInterrupt):
            raise SystemExit(130)
        raise


if __name__ == "__main__":
    main()
