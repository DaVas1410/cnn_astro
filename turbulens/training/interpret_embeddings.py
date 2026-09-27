#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

# Limit BLAS/OpenMP threads before importing numerical libraries.
_PREIMPORT_LINALG_THREADS = str(max(1, int(os.environ.get("INTERPRET_LINALG_THREADS", "4"))))
for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = _PREIMPORT_LINALG_THREADS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from pipeline_lib.common import atomic_csv_write, atomic_json_dump, atomic_npz_save, load_json, utc_now, valid_status_marker
from pipeline_lib.data import MultiFileHDF5Dataset, NormalizationStats, inspect_dataset, load_target_matrix, normalize_targets, worker_init_fn
from pipeline_lib.metrics import pearson_correlation, spearman_correlation
from pipeline_lib.model import build_model_from_config
from pipeline_lib.pca_backend import fit_robust_pca, transform_in_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ensemble embeddings and analyze PCA, CCA, and optional UMAP.")
    parser.add_argument("--members-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-members", type=int, required=True)
    parser.add_argument("--embedding-layer", choices=("shared", "backbone"), default="shared")
    parser.add_argument("--fit-split", choices=("validation",), default="validation")
    parser.add_argument("--plot-split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--pca-fit-max-samples", type=int, default=10000)
    parser.add_argument("--pca-backend", choices=("auto", "torch", "incremental"), default="incremental")
    parser.add_argument("--pca-device", default="cpu", help="Device for PCA backend: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--pca-oversamples", type=int, default=10)
    parser.add_argument("--pca-power-iterations", type=int, default=3)
    parser.add_argument("--pca-cpu-batch-size", type=int, default=4096)
    parser.add_argument("--pca-transform-batch-size", type=int, default=10000)
    parser.add_argument("--linear-algebra-threads", type=int, default=4)
    parser.add_argument("--pc-correlation-components", type=int, default=10)
    parser.add_argument("--cca-components", type=int, default=4)
    parser.add_argument("--cca-fit-max-samples", type=int, default=5000)
    parser.add_argument("--cca-max-iter", type=int, default=1000)
    parser.add_argument("--cca-tol", type=float, default=1e-5)
    parser.add_argument("--cca-pca-components", type=int, default=50)
    parser.add_argument("--cca-pca-variance", type=float, default=0.95)
    parser.add_argument("--umap-policy", choices=("off", "optional", "require"), default="optional")
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-fit-max-samples", type=int, default=5000)
    parser.add_argument("--umap-transform-batch-size", type=int, default=10000)
    parser.add_argument("--umap-representations", choices=("all", "consensus", "none"), default="consensus")
    parser.add_argument("--max-analysis-samples", type=int, default=50000)
    parser.add_argument("--max-plot-points", type=int, default=30000)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--consensus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda:0")
    return torch.device(value)


def discover_members(root: Path) -> list[Path]:
    return [
        path for path in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
        if valid_status_marker(path / "COMPLETED.json") and (path / "checkpoints" / "best_checkpoint.pt").is_file()
    ]


def make_loader(dataset: MultiFileHDF5Dataset, batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset, "batch_size": batch_size, "shuffle": False,
        "num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
        "drop_last": False, "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        # Persistent workers are intentionally disabled here because a new loader is
        # created for every ensemble member and split. Keeping them alive can leave
        # many worker processes resident during long interpretability runs.
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def choose_indices(n: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or n <= maximum:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=maximum, replace=False))


def extract_member(
    member: Path, device: torch.device, embedding_layer: str,
    batch_size: int, num_workers: int, prefetch_factor: int, amp: bool,
    max_samples: int, seed: int,
) -> dict[str, Any]:
    config = load_json(member / "configuration.json")
    checkpoint = torch.load(member / "checkpoints" / "best_checkpoint.pt", map_location=device, weights_only=False)
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    normalization = NormalizationStats(**checkpoint["normalization_stats"])
    targets = list(config["task"]["targets"])
    target_ranges = config["task"]["target_ranges"]
    data_root = Path(config["data"]["root"])
    files, infos, manifest = inspect_dataset(
        data_root, config["data"]["split_directories"], targets, target_ranges,
        augmentation=False, require_square_for_augmentation=False,
        enforce_disjoint_files=bool(config["data"].get("enforce_disjoint_files", True)),
    )
    result: dict[str, Any] = {"config": config, "data_fingerprint": manifest["dataset_fingerprint"], "splits": {}}
    for split in ("validation", "test"):
        labels_physical = load_target_matrix(files[split], targets)
        labels_normalized = normalize_targets(labels_physical, targets, target_ranges)
        dataset = MultiFileHDF5Dataset(
            files[split], infos[split], labels_normalized, normalization,
            in_channels=int(config["model"]["in_channels"]),
            input_standardization=str(config["model"]["input_standardization"]),
            augment=False, augmentation_seed=int(config["seed"]),
        )
        selected = choose_indices(len(dataset), max_samples, seed + (0 if split == "validation" else 1)).astype(np.int64)
        # Read only the selected samples. The previous implementation extracted the
        # complete split and filtered afterward, which could exhaust RAM before PCA.
        subset = Subset(dataset, selected.tolist())
        loader = make_loader(subset, batch_size, num_workers, prefetch_factor)
        embeddings: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        with torch.inference_mode():
            for images, _, sample_index in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                    embedding = model.extract_embedding(images, layer=embedding_layer)
                    prediction = model(images)
                embeddings.append(embedding.float().cpu().numpy())
                predictions.append(prediction.float().cpu().numpy())
                indices.append(np.asarray(sample_index, dtype=np.int64))
        embedding_all = np.concatenate(embeddings).astype(np.float32, copy=False)
        prediction_all = np.concatenate(predictions).astype(np.float32, copy=False)
        index_all = np.concatenate(indices)
        order = np.argsort(index_all)
        embedding_all, prediction_all, index_all = embedding_all[order], prediction_all[order], index_all[order]
        prediction_physical = np.empty_like(prediction_all, dtype=np.float32)
        prediction_clamped = np.clip(prediction_all, 0.0, 1.0)
        for column, target in enumerate(targets):
            low, high = map(float, target_ranges[target])
            prediction_physical[:, column] = prediction_clamped[:, column] * (high - low) + low
        result["splits"][split] = {
            "sample_index": index_all,
            "sample_key": dataset.all_sample_keys()[index_all],
            "embedding": embedding_all,
            "prediction_physical": prediction_physical,
            "target_physical": labels_physical[index_all].astype(np.float32),
        }
    return result


def correlations_table(
    coordinates: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
    targets: Sequence[str], representation: str, split: str, n_components: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pc in range(min(n_components, coordinates.shape[1])):
        for column, target in enumerate(targets):
            for source, values in (("true", truth), ("predicted", prediction)):
                rows.append({
                    "representation": representation, "split": split,
                    "component": f"PC{pc + 1}", "target": target, "source": source,
                    "pearson": pearson_correlation(coordinates[:, pc], values[:, column]),
                    "spearman": spearman_correlation(coordinates[:, pc], values[:, column]),
                })
    return rows


def plot_coordinates(
    coordinates: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
    targets: Sequence[str], output_dir: Path, prefix: str, maximum: int, seed: int,
) -> None:
    selected = choose_indices(len(coordinates), maximum, seed)
    for column, target in enumerate(targets):
        for source, values in (("true", truth), ("predicted", prediction)):
            fig, ax = plt.subplots(figsize=(6.4, 5.4))
            scatter = ax.scatter(
                coordinates[selected, 0], coordinates[selected, 1],
                c=values[selected, column], s=8, alpha=0.5, rasterized=True,
            )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(f"{prefix}: colored by {source} {target}")
            fig.colorbar(scatter, ax=ax, label=target)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"{prefix}_{source}_{target}.png", dpi=180)
            plt.close(fig)


def fit_cca(
    validation_embedding: np.ndarray,
    test_embedding: np.ndarray,
    validation_truth: np.ndarray,
    test_truth: np.ndarray,
    components: int,
    pca_components: int,
    pca_variance: float,
    fit_max_samples: int,
    seed: int,
    args: argparse.Namespace,
    label: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    fit_indices = choose_indices(len(validation_embedding), fit_max_samples, seed + 4201)
    x_fit = np.asarray(validation_embedding[fit_indices], dtype=np.float32)
    y_fit = np.asarray(validation_truth[fit_indices], dtype=np.float32)
    x_scaler = StandardScaler(copy=True).fit(x_fit)
    x_fit_scaled = np.asarray(x_scaler.transform(x_fit), dtype=np.float32)
    max_pca = min(pca_components, x_fit_scaled.shape[1], max(1, x_fit_scaled.shape[0] - 1))
    if max_pca < 1:
        raise ValueError("No embedding dimensions are available before CCA.")

    def progress(message: str) -> None:
        print(f"    [{label}] {message}", flush=True)

    pca, _ = fit_robust_pca(
        x_fit_scaled,
        max_pca,
        backend=args.pca_backend,
        device=args.pca_device if args.pca_device != "auto" else args.device,
        seed=seed + 77,
        fit_max_samples=0,
        oversamples=args.pca_oversamples,
        power_iterations=args.pca_power_iterations,
        cpu_batch_size=args.pca_cpu_batch_size,
        linear_algebra_threads=args.linear_algebra_threads,
        progress=progress,
    )
    x_fit_pca_full = transform_in_batches(pca, x_fit_scaled, args.pca_transform_batch_size)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained = int(np.searchsorted(cumulative, pca_variance) + 1)
    retained = max(1, min(retained, max_pca))
    x_fit_pca = x_fit_pca_full[:, :retained]
    y_scaler = StandardScaler(copy=True).fit(y_fit)
    y_fit_scaled = np.asarray(y_scaler.transform(y_fit), dtype=np.float32)
    n_components = min(components, x_fit_pca.shape[1], y_fit_scaled.shape[1], max(1, x_fit_pca.shape[0] - 1))
    cca = CCA(n_components=n_components, max_iter=args.cca_max_iter, tol=args.cca_tol, scale=False)
    convergence_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        cca.fit(x_fit_pca, y_fit_scaled)
        convergence_messages = [str(item.message) for item in caught]

    def transform(embedding: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_scaled = np.asarray(x_scaler.transform(np.asarray(embedding, dtype=np.float32)), dtype=np.float32)
        x_pca = transform_in_batches(pca, x_scaled, args.pca_transform_batch_size)[:, :retained]
        y_scaled = np.asarray(y_scaler.transform(np.asarray(truth, dtype=np.float32)), dtype=np.float32)
        x_c, y_c = cca.transform(x_pca, y_scaled)
        return np.asarray(x_c, dtype=np.float32), np.asarray(y_c, dtype=np.float32)

    x_val_c, y_val_c = transform(validation_embedding, validation_truth)
    x_test_c, y_test_c = transform(test_embedding, test_truth)
    val_corr = [pearson_correlation(x_val_c[:, i], y_val_c[:, i]) for i in range(n_components)]
    test_corr = [pearson_correlation(x_test_c[:, i], y_test_c[:, i]) for i in range(n_components)]
    metadata = {
        "fit_samples_total": int(len(validation_embedding)),
        "fit_samples_used": int(len(fit_indices)),
        "pca_backend": pca.backend_,
        "pca_device": pca.device_,
        "pca_fit_seconds": pca.fit_seconds_,
        "pca_components_available": max_pca,
        "pca_components_retained": retained,
        "pca_variance_retained": float(cumulative[retained - 1]),
        "canonical_correlations_validation": val_corr,
        "canonical_correlations_test": test_corr,
        "y_loadings": cca.y_loadings_.tolist(),
        "x_loadings": cca.x_loadings_.tolist(),
        "convergence_warnings": convergence_messages,
    }
    return metadata, x_val_c, x_test_c

def maybe_umap(
    validation_embedding: np.ndarray, test_embedding: np.ndarray,
    policy: str, neighbors: int, min_dist: float, max_fit: int, seed: int,
    transform_batch_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if policy == "off":
        return None
    try:
        import umap
    except Exception:
        if policy == "require":
            raise
        return None
    fit_indices = choose_indices(len(validation_embedding), max_fit, seed)
    reducer = umap.UMAP(
        n_neighbors=min(neighbors, max(2, len(fit_indices) - 1)),
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        low_memory=True,
        n_jobs=1,
    )
    reducer.fit(np.asarray(validation_embedding[fit_indices], dtype=np.float32))

    def transform_chunks(array: np.ndarray) -> np.ndarray:
        parts = []
        for start in range(0, len(array), transform_batch_size):
            parts.append(np.asarray(reducer.transform(array[start:start + transform_batch_size]), dtype=np.float32))
        return np.concatenate(parts, axis=0)

    return transform_chunks(validation_embedding), transform_chunks(test_embedding)

def analyze_representation(
    name: str, validation_embedding: np.ndarray, test_embedding: np.ndarray,
    validation_truth: np.ndarray, test_truth: np.ndarray,
    validation_prediction: np.ndarray, test_prediction: np.ndarray,
    targets: Sequence[str], output: Path, args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    representation_dir = output / name
    print(
        f"Analyzing representation: {name} "
        f"(validation={len(validation_embedding)}, test={len(test_embedding)}, "
        f"dimension={validation_embedding.shape[1]})",
        flush=True,
    )

    n_components = min(args.pca_components, validation_embedding.shape[1], max(1, len(validation_embedding) - 1))
    print(
        f"  [{name}] PCA fit: backend={args.pca_backend}, device={args.pca_device}, "
        f"max_samples={args.pca_fit_max_samples or 'all'}...",
        flush=True,
    )
    step = time.time()
    pca, pca_fit_indices = fit_robust_pca(
        validation_embedding,
        n_components,
        backend=args.pca_backend,
        device=args.pca_device if args.pca_device != "auto" else args.device,
        seed=args.sample_seed + 3109,
        fit_max_samples=args.pca_fit_max_samples,
        oversamples=args.pca_oversamples,
        power_iterations=args.pca_power_iterations,
        cpu_batch_size=args.pca_cpu_batch_size,
        linear_algebra_threads=args.linear_algebra_threads,
        progress=lambda message: print(f"    [{name}] {message}", flush=True),
    )
    print(f"  [{name}] PCA transform validation...", flush=True)
    val_pc = transform_in_batches(pca, validation_embedding, args.pca_transform_batch_size)
    print(f"  [{name}] PCA transform test...", flush=True)
    test_pc = transform_in_batches(pca, test_embedding, args.pca_transform_batch_size)
    print(
        f"  [{name}] PCA completed in {time.time() - step:.1f} s "
        f"using backend={pca.backend_} and {len(pca_fit_indices)} fit samples.",
        flush=True,
    )
    atomic_csv_write([
        {"component": f"PC{i + 1}", "explained_variance_ratio": float(value), "cumulative_variance": float(np.sum(pca.explained_variance_ratio_[:i + 1]))}
        for i, value in enumerate(pca.explained_variance_ratio_)
    ], representation_dir / "pca_explained_variance.csv")
    corr_rows = correlations_table(val_pc, validation_truth, validation_prediction, targets, name, "validation", args.pc_correlation_components)
    corr_rows += correlations_table(test_pc, test_truth, test_prediction, targets, name, "test", args.pc_correlation_components)
    atomic_csv_write(corr_rows, representation_dir / "pc_parameter_correlations.csv")
    plot_split_pc = test_pc if args.plot_split == "test" else val_pc
    plot_truth = test_truth if args.plot_split == "test" else validation_truth
    plot_prediction = test_prediction if args.plot_split == "test" else validation_prediction
    plot_coordinates(plot_split_pc, plot_truth, plot_prediction, targets, representation_dir / "plots" / "pca", f"{name}_PCA", args.max_plot_points, args.sample_seed)

    print(f"  [{name}] CCA fit using at most {args.cca_fit_max_samples or 'all'} validation samples...", flush=True)
    step = time.time()
    cca_metadata, val_cca, test_cca = fit_cca(
        validation_embedding, test_embedding, validation_truth, test_truth,
        args.cca_components, args.cca_pca_components, args.cca_pca_variance,
        args.cca_fit_max_samples, args.sample_seed, args, name,
    )
    print(f"  [{name}] CCA completed in {time.time() - step:.1f} s.", flush=True)
    atomic_json_dump(cca_metadata, representation_dir / "cca_metadata.json")
    cca_rows = []
    for index, (validation_corr, test_corr) in enumerate(zip(
        cca_metadata["canonical_correlations_validation"], cca_metadata["canonical_correlations_test"]
    )):
        cca_rows.append({"component": f"CC{index + 1}", "validation_correlation": validation_corr, "test_correlation": test_corr})
    atomic_csv_write(cca_rows, representation_dir / "cca_correlations.csv")
    if val_cca.shape[1] >= 2:
        plot_cca = test_cca if args.plot_split == "test" else val_cca
        plot_coordinates(plot_cca, plot_truth, plot_prediction, targets, representation_dir / "plots" / "cca", f"{name}_CCA", args.max_plot_points, args.sample_seed)

    run_umap = args.umap_representations == "all" or (args.umap_representations == "consensus" and name == "consensus")
    umap_coordinates = None
    if run_umap:
        print(f"  [{name}] UMAP fit/transform...", flush=True)
        step = time.time()
        umap_coordinates = maybe_umap(
            validation_embedding, test_embedding, args.umap_policy,
            args.umap_neighbors, args.umap_min_dist, args.umap_fit_max_samples,
            args.sample_seed, args.umap_transform_batch_size,
        )
        print(f"  [{name}] UMAP stage completed in {time.time() - step:.1f} s.", flush=True)
    if umap_coordinates is not None:
        val_umap, test_umap = umap_coordinates
        plot_umap = test_umap if args.plot_split == "test" else val_umap
        plot_coordinates(plot_umap, plot_truth, plot_prediction, targets, representation_dir / "plots" / "umap", f"{name}_UMAP", args.max_plot_points, args.sample_seed)
        atomic_npz_save(representation_dir / "umap_coordinates.npz", validation=val_umap, test=test_umap)

    atomic_npz_save(representation_dir / "pca_coordinates.npz", validation=val_pc, test=test_pc)
    atomic_npz_save(representation_dir / "cca_coordinates.npz", validation=val_cca, test=test_cca)
    summary = {
        "name": name,
        "embedding_dimension": int(validation_embedding.shape[1]),
        "validation_samples": int(len(validation_embedding)),
        "test_samples": int(len(test_embedding)),
        "pca_components": n_components,
        "pca_fit_samples_used": int(len(pca_fit_indices)),
        "pca_backend": pca.backend_,
        "pca_device": pca.device_,
        "pca_fit_seconds": pca.fit_seconds_,
        "pca_variance_first_two": float(np.sum(pca.explained_variance_ratio_[:2])),
        "cca": cca_metadata,
        "umap_completed": umap_coordinates is not None,
        "runtime_seconds": float(time.time() - started),
    }
    atomic_json_dump(summary, representation_dir / "SUMMARY.json")
    print(f"  [{name}] completed in {time.time() - started:.1f} s.", flush=True)
    return summary

def main() -> None:
    args = build_parser().parse_args()
    for name in ("max_analysis_samples", "pca_fit_max_samples", "cca_fit_max_samples", "umap_fit_max_samples"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.umap_transform_batch_size < 1:
        raise ValueError("--umap-transform-batch-size must be positive.")
    root = args.members_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    members = discover_members(root)
    if len(members) != args.expected_members and not args.allow_partial:
        raise RuntimeError(f"Found {len(members)} members; expected {args.expected_members}.")
    if len(members) < 2:
        raise RuntimeError("At least two members are required for ensemble interpretability.")
    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)
    print(f"Members: {len(members)}", flush=True)
    print(f"Embedding layer: {args.embedding_layer}", flush=True)
    extracted = []
    for index, member in enumerate(members):
        print(f"[{index + 1}/{len(members)}] Extracting {member.name}...", flush=True)
        extracted.append(extract_member(
            member, device, args.embedding_layer, args.batch_size, args.num_workers,
            args.prefetch_factor, args.amp, args.max_analysis_samples,
            args.sample_seed,
        ))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reference = extracted[0]
    targets = list(reference["config"]["task"]["targets"])
    for member, item in zip(members[1:], extracted[1:]):
        if item["data_fingerprint"] != reference["data_fingerprint"]:
            raise ValueError(f"Dataset fingerprint mismatch for {member.name}.")
        for split in ("validation", "test"):
            for key in ("sample_index", "sample_key", "target_physical"):
                if not np.array_equal(item["splits"][split][key], reference["splits"][split][key]):
                    raise ValueError(f"{member.name}: {split} {key} mismatch.")

    summaries = []
    all_correlations: list[dict[str, Any]] = []
    standardized_validation: list[np.ndarray] = []
    standardized_test: list[np.ndarray] = []
    member_predictions_validation: list[np.ndarray] = []
    member_predictions_test: list[np.ndarray] = []
    for member, item in zip(members, extracted):
        validation = item["splits"]["validation"]
        test = item["splits"]["test"]
        summary = analyze_representation(
            member.name, validation["embedding"], test["embedding"],
            validation["target_physical"], test["target_physical"],
            validation["prediction_physical"], test["prediction_physical"],
            targets, output / "representations", args,
        )
        summaries.append(summary)
        scaler = StandardScaler().fit(validation["embedding"])
        standardized_validation.append(scaler.transform(validation["embedding"]))
        standardized_test.append(scaler.transform(test["embedding"]))
        member_predictions_validation.append(validation["prediction_physical"])
        member_predictions_test.append(test["prediction_physical"])
        if args.save_embeddings:
            atomic_npz_save(
                output / "embeddings" / f"{member.name}.npz",
                validation=validation["embedding"], test=test["embedding"],
                validation_sample_key=validation["sample_key"], test_sample_key=test["sample_key"],
            )

    if args.consensus:
        consensus_validation = np.concatenate(standardized_validation, axis=1)
        consensus_test = np.concatenate(standardized_test, axis=1)
        ensemble_prediction_validation = np.mean(member_predictions_validation, axis=0)
        ensemble_prediction_test = np.mean(member_predictions_test, axis=0)
        summaries.append(analyze_representation(
            "consensus", consensus_validation, consensus_test,
            reference["splits"]["validation"]["target_physical"],
            reference["splits"]["test"]["target_physical"],
            ensemble_prediction_validation, ensemble_prediction_test,
            targets, output / "representations", args,
        ))
        atomic_json_dump({
            "method": "member-wise StandardScaler fitted on validation, followed by concatenation",
            "reason": "Raw embeddings from independently trained networks are not averaged because their coordinate systems can differ by rotations, sign changes, and permutations.",
            "dimension": int(consensus_validation.shape[1]),
        }, output / "consensus_representation.json")

    result = {
        "status": "completed", "completed_at_utc": utc_now(),
        "members": [member.name for member in members], "targets": targets,
        "embedding_layer": args.embedding_layer,
        "fit_split": "validation", "plot_split": args.plot_split,
        "representations": summaries,
        "methodological_note": "PCA, CCA preprocessing, and UMAP are fitted on validation embeddings. Test embeddings are transformed afterward for held-out interpretation.",
    }
    atomic_json_dump(result, output / "FINAL_INTERPRETABILITY_SUMMARY.json")
    atomic_json_dump({"status": "completed", "completed_at_utc": utc_now()}, output / "COMPLETED.json")
    print(f"Interpretability analysis complete: {output}")


if __name__ == "__main__":
    main()
