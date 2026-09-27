#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    @contextlib.contextmanager
    def threadpool_limits(limits=None):
        yield


Progress = Optional[Callable[[str], None]]


def _emit(progress: Progress, message: str) -> None:
    if progress is not None:
        progress(message)


def choose_fit_indices(n_samples: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False)).astype(np.int64)


def choose_backend(requested: str, device: torch.device) -> str:
    requested = str(requested).lower()
    if requested not in {"auto", "torch", "incremental"}:
        raise ValueError("PCA backend must be one of: auto, torch, incremental")
    if requested == "auto":
        return "torch" if device.type == "cuda" and torch.cuda.is_available() else "incremental"
    if requested == "torch" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PCA backend 'torch' requested on CUDA, but CUDA is unavailable.")
    return requested


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

    def transform(self, x: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"PCA transform expected (*,{self.n_features_in_}), got {tuple(array.shape)}"
            )
        batch_size = max(1, int(batch_size))
        out = np.empty((array.shape[0], self.n_components_), dtype=np.float32)
        components_t = np.asarray(self.components_.T, dtype=np.float32)
        mean = np.asarray(self.mean_, dtype=np.float32)
        for start in range(0, array.shape[0], batch_size):
            stop = min(array.shape[0], start + batch_size)
            centered = np.asarray(array[start:stop], dtype=np.float32) - mean
            out[start:stop] = centered @ components_t
        return out


def _fit_torch_lowrank(
    x: np.ndarray,
    n_components: int,
    device: torch.device,
    seed: int,
    oversamples: int,
    niter: int,
    progress: Progress,
) -> RobustPCA:
    started = time.time()
    x_np = np.asarray(x, dtype=np.float32, order="C")
    n_samples, n_features = x_np.shape
    q = min(n_features, max(n_components, n_components + max(0, int(oversamples))))
    _emit(
        progress,
        f"PCA backend=torch_lowrank device={device} matrix={n_samples}x{n_features} "
        f"components={n_components} q={q} niter={niter}",
    )
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        matrix_bytes = x_np.nbytes
        _emit(
            progress,
            f"CUDA memory before PCA: free={free_bytes / 2**30:.2f} GiB, "
            f"total={total_bytes / 2**30:.2f} GiB, fit_matrix={matrix_bytes / 2**30:.2f} GiB",
        )
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        tensor = torch.from_numpy(x_np).to(device=device, dtype=torch.float32, non_blocking=False)
        mean_t = tensor.mean(dim=0)
        centered = tensor - mean_t
        # center=False because we already centered explicitly and retain the exact mean.
        _, singular_values, vectors = torch.pca_lowrank(
            centered,
            q=q,
            center=False,
            niter=max(1, int(niter)),
        )
        vectors = vectors[:, :n_components]
        singular_values = singular_values[:n_components]
        components = vectors.T.contiguous()
        explained_variance = singular_values.square() / max(1, n_samples - 1)
        total_variance = centered.var(dim=0, unbiased=True).sum()
        ratio = explained_variance / torch.clamp(total_variance, min=torch.finfo(torch.float32).eps)
        model = RobustPCA(
            mean_=mean_t.detach().cpu().numpy().astype(np.float32, copy=False),
            components_=components.detach().cpu().numpy().astype(np.float32, copy=False),
            explained_variance_=explained_variance.detach().cpu().numpy().astype(np.float32, copy=False),
            explained_variance_ratio_=ratio.detach().cpu().numpy().astype(np.float32, copy=False),
            singular_values_=singular_values.detach().cpu().numpy().astype(np.float32, copy=False),
            n_components_=int(n_components),
            n_features_in_=int(n_features),
            n_samples_fit_=int(n_samples),
            backend_="torch_lowrank",
            device_=str(device),
            fit_seconds_=float(time.time() - started),
        )
        del tensor, centered, mean_t, vectors, singular_values, components, explained_variance, total_variance, ratio
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _emit(progress, f"PCA torch_lowrank finished in {model.fit_seconds_:.2f} s")
    return model


def _fit_incremental(
    x: np.ndarray,
    n_components: int,
    batch_size: int,
    threads: int,
    progress: Progress,
) -> RobustPCA:
    started = time.time()
    x_np = np.asarray(x, dtype=np.float32)
    n_samples, n_features = x_np.shape
    batch_size = max(int(batch_size), 4 * n_components, n_components + 1)
    # np.array_split avoids a tiny final batch that IncrementalPCA cannot accept.
    n_batches = max(1, n_samples // batch_size)
    n_batches = min(n_batches, max(1, n_samples // max(1, n_components)))
    chunks = np.array_split(np.arange(n_samples, dtype=np.int64), n_batches)
    _emit(
        progress,
        f"PCA backend=incremental_cpu matrix={n_samples}x{n_features} components={n_components} "
        f"batches={len(chunks)} threads={threads}",
    )
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    with threadpool_limits(limits=max(1, int(threads))):
        for index, idx in enumerate(chunks, start=1):
            if idx.size < n_components:
                raise RuntimeError(
                    f"Incremental PCA batch {index} has only {idx.size} samples for {n_components} components."
                )
            pca.partial_fit(np.asarray(x_np[idx], dtype=np.float32))
            _emit(progress, f"Incremental PCA batch {index}/{len(chunks)} complete ({idx.size} samples)")
    fit_seconds = float(time.time() - started)
    model = RobustPCA(
        mean_=np.asarray(pca.mean_, dtype=np.float32),
        components_=np.asarray(pca.components_, dtype=np.float32),
        explained_variance_=np.asarray(pca.explained_variance_, dtype=np.float32),
        explained_variance_ratio_=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
        singular_values_=np.asarray(pca.singular_values_, dtype=np.float32),
        n_components_=int(n_components),
        n_features_in_=int(n_features),
        n_samples_fit_=int(n_samples),
        backend_="incremental_cpu",
        device_="cpu",
        fit_seconds_=fit_seconds,
    )
    _emit(progress, f"PCA incremental_cpu finished in {fit_seconds:.2f} s")
    return model


def fit_robust_pca(
    x: np.ndarray,
    n_components: int,
    *,
    backend: str = "auto",
    device: str | torch.device = "auto",
    seed: int = 2026,
    fit_max_samples: int = 20000,
    oversamples: int = 10,
    power_iterations: int = 3,
    cpu_batch_size: int = 4096,
    linear_algebra_threads: int = 4,
    progress: Progress = None,
) -> tuple[RobustPCA, np.ndarray]:
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"PCA expects a 2-D matrix; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("PCA input contains NaN or infinite values.")
    if array.shape[0] < 3 or array.shape[1] < 2:
        raise ValueError(f"PCA requires at least 3 samples and 2 features; got {array.shape}")
    max_components = min(array.shape[0] - 1, array.shape[1])
    n_components = min(int(n_components), max_components)
    if n_components < 2:
        raise ValueError("PCA requires at least 2 components.")
    indices = choose_fit_indices(array.shape[0], int(fit_max_samples), int(seed) + 104729)
    x_fit = np.asarray(array[indices], dtype=np.float32, order="C")
    if isinstance(device, torch.device):
        resolved = device
    else:
        value = str(device).lower()
        if value == "auto":
            resolved = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif value == "cuda":
            resolved = torch.device("cuda:0")
        else:
            resolved = torch.device(value)
    selected_backend = choose_backend(backend, resolved)
    _emit(progress, f"PCA fit subset: {len(indices)}/{array.shape[0]} samples")
    if selected_backend == "torch":
        try:
            model = _fit_torch_lowrank(
                x_fit, n_components, resolved, seed, oversamples, power_iterations, progress
            )
            return model, indices
        except RuntimeError as exc:
            if str(backend).lower() != "auto":
                raise
            _emit(progress, f"Torch PCA failed ({type(exc).__name__}: {exc}); falling back to incremental CPU PCA")
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
    model = _fit_incremental(
        x_fit, n_components, cpu_batch_size, linear_algebra_threads, progress
    )
    return model, indices


def transform_in_batches(model: RobustPCA, x: np.ndarray, batch_size: int = 10000) -> np.ndarray:
    return model.transform(x, batch_size=batch_size)
