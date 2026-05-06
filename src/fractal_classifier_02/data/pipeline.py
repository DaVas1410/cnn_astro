"""
TensorFlow tf.data pipeline for Fractal Classifier v0.2.

Lazily loads images from HDF5 files, extracts non-overlapping 128×128
patches for images larger than the native resolution, applies optional
data augmentation, and returns (patch, k_min) batches.
"""

import logging
from typing import List, Optional

import numpy as np
import tensorflow as tf
import h5py

from .loader import DataIndexer

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Constructs train / val / test tf.data.Dataset objects from a DataIndexer.

    Patch strategy (applied per sample at load time):
      - 128×128 → 1 patch  (returned as-is)
      - 256×256 → 4 patches (2×2 non-overlapping 128×128)
      - 512×512 → 16 patches (4×4 non-overlapping 128×128)
    Each patch inherits the k_min label of its parent image.

    Args:
        indexer:        A built DataIndexer instance.
        native_res:     Native model input resolution (default 128).
        batch_size:     Samples per batch.
        augment:        Whether to apply data augmentation (training only).
        shuffle_buffer: Size of the shuffle buffer for training.
        num_parallel:   tf.data parallel workers.
        prefetch:       Number of batches to prefetch.
    """

    def __init__(
        self,
        indexer: DataIndexer,
        native_res: int = 128,
        batch_size: int = 32,
        augment: bool = False,
        shuffle_buffer: int = 4096,
        num_parallel: int = 4,
        prefetch: int = 2,
    ):
        self.indexer = indexer
        self.native_res = native_res
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle_buffer = shuffle_buffer
        self.num_parallel = num_parallel
        self.prefetch = prefetch

    # ------------------------------------------------------------------
    # Public factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, indexer: DataIndexer, config, training: bool = True) -> 'DataPipeline':
        """Build a DataPipeline from a Config object."""
        data_cfg = config.data
        train_cfg = config.training
        aug_cfg = train_cfg.augmentation if hasattr(train_cfg, 'augmentation') else None
        augment = training and (aug_cfg is not None and (
            getattr(aug_cfg, 'horizontal_flip', False) or
            getattr(aug_cfg, 'vertical_flip', False)
        ))
        return cls(
            indexer=indexer,
            native_res=int(getattr(data_cfg, 'native_resolution', 128)),
            batch_size=int(train_cfg.batch_size),
            augment=augment,
            shuffle_buffer=int(getattr(train_cfg, 'shuffle_buffer', 4096)),
            num_parallel=int(getattr(data_cfg, 'num_parallel_calls', 4)),
            prefetch=int(getattr(data_cfg, 'prefetch', 2)),
        )

    def build_train_dataset(self) -> tf.data.Dataset:
        return self._build(self.indexer.train_index, shuffle=True, augment=self.augment)

    def build_val_dataset(self) -> tf.data.Dataset:
        return self._build(self.indexer.val_index, shuffle=False, augment=False)

    def build_test_dataset(self) -> tf.data.Dataset:
        return self._build(self.indexer.test_index, shuffle=False, augment=False)

    # ------------------------------------------------------------------
    # Internal pipeline construction
    # ------------------------------------------------------------------

    def _build(
        self,
        index: np.ndarray,
        shuffle: bool,
        augment: bool,
    ) -> tf.data.Dataset:
        """Build a tf.data.Dataset from an index array."""
        file_paths = index['file_path'].tolist()
        resolutions = index['resolution'].tolist()
        image_idxs  = index['image_idx'].tolist()
        k_min_vals  = index['k_min'].tolist()

        dataset = tf.data.Dataset.from_tensor_slices((
            file_paths,
            resolutions,
            image_idxs,
            k_min_vals,
        ))

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(self.shuffle_buffer, len(index)),
                reshuffle_each_iteration=True,
            )

        native = self.native_res

        def load_and_patch(fpath, res, img_idx, k_min):
            patches, labels = tf.py_function(
                func=lambda fp, r, i, k: _load_and_extract_patches(
                    fp.numpy().decode(), int(r.numpy()), int(i.numpy()),
                    float(k.numpy()), native
                ),
                inp=[fpath, res, img_idx, k_min],
                Tout=[tf.float32, tf.float32],
            )
            patches.set_shape([None, native, native, 1])
            labels.set_shape([None])
            return tf.data.Dataset.from_tensor_slices((patches, labels))

        dataset = dataset.flat_map(load_and_patch)

        if augment:
            dataset = dataset.map(
                lambda img, lbl: (_augment(img), lbl),
                num_parallel_calls=self.num_parallel,
            )

        dataset = (
            dataset
            .batch(self.batch_size, drop_remainder=False)
            .prefetch(self.prefetch)
        )
        return dataset


# ---------------------------------------------------------------------------
# Helper functions (outside class for tf.py_function compatibility)
# ---------------------------------------------------------------------------

def _load_and_extract_patches(
    file_path: str,
    resolution: int,
    image_idx: int,
    k_min: float,
    native_res: int,
) -> tuple:
    """
    Load a single image from HDF5 and return patches + labels as numpy arrays.

    Returns:
        patches: (N_patches, native_res, native_res, 1) float32
        labels:  (N_patches,) float32
    """
    group_key = f"{resolution}x{resolution}"
    with h5py.File(file_path, 'r') as f:
        img = f[f"{group_key}/images"][image_idx]  # (H, W) float32

    patches = _extract_patches(img, native_res)          # (N, native_res, native_res)
    patches = patches[:, :, :, np.newaxis]               # add channel dim
    labels = np.full(len(patches), k_min, dtype=np.float32)
    return patches, labels


def _extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Slice a 2-D image into non-overlapping *patch_size × patch_size* tiles.

    If the image is already *patch_size × patch_size*, it is returned as a
    single-element batch.  Images whose side length is not a multiple of
    *patch_size* are centre-cropped before slicing.

    Args:
        image:      2-D numpy array (H, W).
        patch_size: Side length of each output patch.

    Returns:
        np.ndarray of shape (n_patches, patch_size, patch_size).
    """
    h, w = image.shape
    # Centre-crop to a multiple of patch_size
    h_crop = (h // patch_size) * patch_size
    w_crop = (w // patch_size) * patch_size
    row_start = (h - h_crop) // 2
    col_start = (w - w_crop) // 2
    image = image[row_start:row_start + h_crop, col_start:col_start + w_crop]

    n_rows = h_crop // patch_size
    n_cols = w_crop // patch_size
    patches = (
        image
        .reshape(n_rows, patch_size, n_cols, patch_size)
        .transpose(0, 2, 1, 3)              # (n_rows, n_cols, patch_size, patch_size)
        .reshape(-1, patch_size, patch_size) # (n_patches, patch_size, patch_size)
    )
    return patches.astype(np.float32)


def _augment(image: tf.Tensor) -> tf.Tensor:
    """Random horizontal and vertical flips."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image
