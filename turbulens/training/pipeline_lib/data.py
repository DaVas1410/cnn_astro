from __future__ import annotations

import bisect
import contextlib
import hashlib
import logging
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from .common import object_sha256, to_jsonable

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class H5Info:
    path: str
    relative_path: str
    n_samples: int
    raw_image_shape: tuple[int, ...]
    height: int
    width: int
    stored_channels: int
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class NormalizationStats:
    lower_percentile: float
    upper_percentile: float
    lower_value: float
    upper_value: float
    value_range: float
    finite_pixels_examined: int
    images_examined: int
    sampling_seed: int
    source: str


def infer_image_layout(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1]), 1
    if len(shape) != 3:
        raise ValueError(
            f"Each image must have shape (H,W), (C,H,W), or (H,W,C); received {tuple(shape)}."
        )
    if int(shape[0]) in (1, 3):
        return int(shape[1]), int(shape[2]), int(shape[0])
    if int(shape[-1]) in (1, 3):
        return int(shape[0]), int(shape[1]), int(shape[-1])
    raise ValueError(
        f"Cannot identify channel axis in image shape {tuple(shape)}. Only 1 or 3 channels are supported."
    )


def discover_split_files(data_root: Path, split_directories: Mapping[str, str]) -> dict[str, list[Path]]:
    root = data_root.expanduser().resolve()
    discovered: dict[str, list[Path]] = {}
    for split in ("train", "validation", "test"):
        folder = root / str(split_directories[split])
        if not folder.is_dir():
            raise NotADirectoryError(f"Missing required {split} directory: {folder}")
        files = sorted((p.resolve() for p in folder.glob("*.h5") if p.is_file()), key=str)
        if not files:
            raise FileNotFoundError(f"No .h5 files found for {split}: {folder}")
        discovered[split] = files
    return discovered


def inspect_h5(path: Path, data_root: Path, targets: Sequence[str], target_ranges: Mapping[str, Sequence[float]]) -> H5Info:
    stat = path.stat()
    with h5py.File(path, "r") as handle:
        if "images" not in handle:
            raise KeyError(f"{path}: missing /images")
        images = handle["images"]
        if images.ndim not in (3, 4):
            raise ValueError(
                f"{path}: /images must be (N,H,W), (N,C,H,W), or (N,H,W,C), got {images.shape}."
            )
        n_samples = int(images.shape[0])
        if n_samples < 1:
            raise ValueError(f"{path}: no images found.")
        raw_image_shape = tuple(int(x) for x in images.shape[1:])
        height, width, stored_channels = infer_image_layout(raw_image_shape)
        if "parameters" not in handle:
            raise KeyError(f"{path}: missing /parameters group")
        for target in targets:
            key = f"parameters/{target}"
            if key not in handle:
                raise KeyError(f"{path}: missing /{key}")
            dataset = handle[key]
            if dataset.ndim != 1 or int(dataset.shape[0]) != n_samples:
                raise ValueError(f"{path}: /{key} must have shape ({n_samples},), got {dataset.shape}.")
            values = np.asarray(dataset[:], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: target {target} contains NaN or infinite values.")
            low, high = map(float, target_ranges[target])
            tolerance = max(1e-6, 1e-6 * (high - low))
            if float(values.min()) < low - tolerance or float(values.max()) > high + tolerance:
                raise ValueError(
                    f"{path}: target {target} lies outside configured range [{low}, {high}]. "
                    f"Observed [{float(values.min())}, {float(values.max())}]."
                )
    return H5Info(
        path=str(path),
        relative_path=str(path.relative_to(data_root.resolve())),
        n_samples=n_samples,
        raw_image_shape=raw_image_shape,
        height=height,
        width=width,
        stored_channels=stored_channels,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def inspect_dataset(
    data_root: Path,
    split_directories: Mapping[str, str],
    targets: Sequence[str],
    target_ranges: Mapping[str, Sequence[float]],
    *,
    augmentation: bool,
    require_square_for_augmentation: bool,
    enforce_disjoint_files: bool = True,
) -> tuple[dict[str, list[Path]], dict[str, list[H5Info]], dict[str, Any]]:
    files = discover_split_files(data_root, split_directories)
    if enforce_disjoint_files:
        seen: dict[Path, str] = {}
        for split, paths in files.items():
            for path in paths:
                resolved = path.resolve()
                previous = seen.get(resolved)
                if previous is not None:
                    raise ValueError(
                        f"The same resolved HDF5 file appears in both {previous} and {split}: {resolved}"
                    )
                seen[resolved] = split
    infos: dict[str, list[H5Info]] = {}
    all_infos: list[H5Info] = []
    for split, paths in files.items():
        infos[split] = [inspect_h5(path, data_root, targets, target_ranges) for path in paths]
        all_infos.extend(infos[split])
    shapes = {(item.height, item.width, item.stored_channels) for item in all_infos}
    if len(shapes) != 1:
        raise ValueError(f"All HDF5 files must use the same H/W/channel layout; found {sorted(shapes)}")
    height, width, stored_channels = next(iter(shapes))
    if augmentation and require_square_for_augmentation and height != width:
        raise ValueError(
            f"90-degree rotation augmentation requires square images; found {height}x{width}."
        )
    manifest = {
        "format_version": 2,
        "data_root": str(data_root.resolve()),
        "targets": list(targets),
        "target_ranges": to_jsonable(target_ranges),
        "image_shape": [height, width],
        "stored_channels": stored_channels,
        "splits": {
            split: {
                "total_samples": int(sum(item.n_samples for item in split_infos)),
                "files": [asdict(item) for item in split_infos],
            }
            for split, split_infos in infos.items()
        },
    }
    manifest["dataset_fingerprint"] = object_sha256(manifest)
    return files, infos, manifest


def total_samples(infos: Sequence[H5Info]) -> int:
    return int(sum(info.n_samples for info in infos))


def build_offsets(infos: Sequence[H5Info]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([item.n_samples for item in infos], dtype=np.int64)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    return starts, ends


def load_target_matrix(paths: Sequence[Path], targets: Sequence[str]) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            columns = [np.asarray(handle[f"parameters/{target}"][:], dtype=np.float32) for target in targets]
        matrices.append(np.stack(columns, axis=1).astype(np.float32, copy=False))
    if not matrices:
        raise ValueError("No target files supplied.")
    return np.concatenate(matrices, axis=0).astype(np.float32, copy=False)


def normalize_targets(values: np.ndarray, targets: Sequence[str], target_ranges: Mapping[str, Sequence[float]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(targets):
        raise ValueError(f"Expected targets shape (N,{len(targets)}), got {values.shape}.")
    output = np.empty_like(values, dtype=np.float32)
    for col, target in enumerate(targets):
        low, high = map(float, target_ranges[target])
        output[:, col] = (values[:, col] - low) / (high - low)
    if not np.all(np.isfinite(output)):
        raise ValueError("Target normalization produced non-finite values.")
    return output


def denormalize_targets(
    values: np.ndarray,
    targets: Sequence[str],
    target_ranges: Mapping[str, Sequence[float]],
    *,
    clamp: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(targets):
        raise ValueError(f"Expected predictions shape (N,{len(targets)}), got {values.shape}.")
    work = np.clip(values, 0.0, 1.0) if clamp else values
    output = np.empty_like(work, dtype=np.float32)
    for col, target in enumerate(targets):
        low, high = map(float, target_ranges[target])
        output[:, col] = work[:, col] * (high - low) + low
    return output


def to_channel_first_float32(raw: np.ndarray) -> np.ndarray:
    image = np.asarray(raw, dtype=np.float32)
    if image.ndim == 2:
        return image[None]
    if image.ndim != 3:
        raise ValueError(f"Single image must be 2D or 3D, got {image.shape}.")
    if image.shape[0] in (1, 3):
        return image
    if image.shape[-1] in (1, 3):
        return np.moveaxis(image, -1, 0)
    raise ValueError(f"Cannot infer channel axis for image shape {image.shape}.")


def pixels_for_percentiles(raw: np.ndarray) -> np.ndarray:
    return to_channel_first_float32(raw).reshape(-1)


def estimate_normalization(
    paths: Sequence[Path],
    infos: Sequence[H5Info],
    lower_percentile: float,
    upper_percentile: float,
    max_images: int,
    max_pixels: int,
    seed: int,
    epsilon: float,
    logger: logging.Logger,
) -> NormalizationStats:
    starts, ends = build_offsets(infos)
    total = int(ends[-1])
    rng = np.random.default_rng(seed)
    images_to_use = min(int(max_images), total)
    indices = np.arange(total, dtype=np.int64)
    if images_to_use < total:
        indices = np.sort(rng.choice(indices, size=images_to_use, replace=False))
    pixels_per_image = max(1, int(max_pixels // images_to_use))
    samples: list[np.ndarray] = []
    finite_count = 0
    logger.info(
        "Estimating image normalization percentiles %.4f/%.4f from %d training images (<=%d pixels).",
        lower_percentile, upper_percentile, images_to_use, max_pixels,
    )
    progress = tqdm(total=images_to_use, desc="Normalization", unit="image", leave=False)
    try:
        for file_index, path in enumerate(paths):
            mask = (indices >= starts[file_index]) & (indices < ends[file_index])
            local_indices = indices[mask] - starts[file_index]
            if local_indices.size == 0:
                continue
            with h5py.File(path, "r") as handle:
                images = handle["images"]
                for local_index in local_indices:
                    pixels = pixels_for_percentiles(images[int(local_index)])
                    finite = pixels[np.isfinite(pixels)]
                    if finite.size:
                        take = min(pixels_per_image, int(finite.size))
                        if take < finite.size:
                            positions = rng.choice(finite.size, size=take, replace=False)
                            finite = finite[positions]
                        finite = finite.astype(np.float32, copy=False)
                        samples.append(finite)
                        finite_count += int(finite.size)
                    progress.update(1)
    finally:
        progress.close()
    if finite_count < 2:
        raise ValueError("Fewer than two finite pixels were available for normalization.")
    merged = np.concatenate(samples)
    if merged.size > max_pixels:
        merged = merged[rng.choice(merged.size, size=max_pixels, replace=False)]
    lower, upper = np.percentile(merged, [lower_percentile, upper_percentile]).astype(float)
    value_range = float(upper - lower)
    if not np.isfinite(lower) or not np.isfinite(upper) or value_range <= epsilon:
        raise ValueError(
            f"Invalid normalization values: lower={lower}, upper={upper}, range={value_range}."
        )
    return NormalizationStats(
        lower_percentile=float(lower_percentile),
        upper_percentile=float(upper_percentile),
        lower_value=float(lower),
        upper_value=float(upper),
        value_range=value_range,
        finite_pixels_examined=finite_count,
        images_examined=images_to_use,
        sampling_seed=int(seed),
        source=f"training_images_only_sampled_global_pixels_across_{len(paths)}_h5_files",
    )


class EpochRandomSampler(Sampler[int]):
    def __init__(self, data_source: Dataset[Any], seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class MultiFileHDF5Dataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        paths: Sequence[Path],
        infos: Sequence[H5Info],
        labels_normalized: np.ndarray,
        normalization: NormalizationStats,
        *,
        in_channels: int,
        input_standardization: str,
        augment: bool,
        augmentation_seed: int,
        shared_epoch: Any | None = None,
    ) -> None:
        self.paths = [str(path) for path in paths]
        self.infos = list(infos)
        self.starts, self.ends = build_offsets(infos)
        self.ends_list = self.ends.tolist()
        self.n_samples = int(self.ends[-1])
        self.labels = np.asarray(labels_normalized, dtype=np.float32)
        if self.labels.shape[0] != self.n_samples:
            raise ValueError(f"Label count mismatch: {self.labels.shape[0]} versus {self.n_samples}.")
        self.normalization = normalization
        self.in_channels = int(in_channels)
        self.input_standardization = str(input_standardization)
        self.augment = bool(augment)
        self.augmentation_seed = int(augmentation_seed)
        self.shared_epoch = shared_epoch if shared_epoch is not None else mp.Value("i", 0)
        self._handles: list[h5py.File | None] = [None] * len(self.paths)
        if self.in_channels == 3:
            self.mean = IMAGENET_MEAN[:, None, None]
            self.std = IMAGENET_STD[:, None, None]
        elif self.in_channels == 1:
            self.mean = np.asarray([IMAGENET_MEAN.mean()], dtype=np.float32)[:, None, None]
            self.std = np.asarray([IMAGENET_STD.mean()], dtype=np.float32)[:, None, None]
        else:
            raise ValueError("in_channels must be 1 or 3.")

    def __len__(self) -> int:
        return self.n_samples

    def _locate(self, global_index: int) -> tuple[int, int]:
        file_index = bisect.bisect_right(self.ends_list, global_index)
        if file_index >= len(self.paths):
            raise IndexError(global_index)
        return file_index, global_index - int(self.starts[file_index])

    def _handle(self, file_index: int) -> h5py.File:
        handle = self._handles[file_index]
        if handle is None:
            handle = h5py.File(self.paths[file_index], "r")
            self._handles[file_index] = handle
        return handle

    def sample_key(self, global_index: int) -> str:
        file_index, local_index = self._locate(global_index)
        return f"{self.infos[file_index].relative_path}::{local_index}"

    def all_sample_keys(self) -> np.ndarray:
        return np.asarray([self.sample_key(i) for i in range(self.n_samples)], dtype=str)

    def _convert_channels(self, raw: np.ndarray) -> np.ndarray:
        image = to_channel_first_float32(raw)
        stored = image.shape[0]
        if stored == self.in_channels:
            return image
        if stored == 3 and self.in_channels == 1:
            return image.mean(axis=0, keepdims=True, dtype=np.float32)
        if stored == 1 and self.in_channels == 3:
            return np.repeat(image, 3, axis=0)
        raise ValueError(f"Unsupported channel conversion {stored} -> {self.in_channels}.")

    def _augment(self, image: np.ndarray, index: int) -> np.ndarray:
        if not self.augment:
            return image
        epoch = int(self.shared_epoch.value)
        rng = np.random.default_rng(np.random.SeedSequence([self.augmentation_seed, epoch, index]))
        k = int(rng.integers(0, 4))
        if k:
            image = np.rot90(image, k=k, axes=(-2, -1))
        if bool(rng.integers(0, 2)):
            image = np.flip(image, axis=-1)
        if bool(rng.integers(0, 2)):
            image = np.flip(image, axis=-2)
        return np.ascontiguousarray(image)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        file_index, local_index = self._locate(int(index))
        raw = self._handle(file_index)["images"][local_index]
        image = self._convert_channels(raw).astype(np.float32, copy=False)
        finite = np.isfinite(image)
        if not finite.all():
            image = np.where(finite, image, self.normalization.lower_value).astype(np.float32)
        image = (image - self.normalization.lower_value) / self.normalization.value_range
        image = np.clip(image, 0.0, 1.0)
        image = self._augment(image, int(index))
        if self.input_standardization == "imagenet":
            image = (image - self.mean) / self.std
        elif self.input_standardization != "none":
            raise ValueError(f"Unknown input standardization: {self.input_standardization}")
        return (
            torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(self.labels[index], dtype=np.float32)),
            int(index),
        )

    def close(self) -> None:
        for handle in self._handles:
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()

    def __del__(self) -> None:
        self.close()


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
