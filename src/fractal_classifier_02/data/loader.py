"""
Multi-file HDF5 data indexer for Fractal Classifier v0.2.

Scans a directory of HDF5 files (one per k_min value, produced by the
kmin_auto_batch generator) and builds an in-memory index of all available
samples across all requested resolutions.

Expected HDF5 structure per file:
    {res}x{res}/images          – (N, H, W) float32 images
    {res}x{res}/parameters/k_min – (N,) float32 labels
where {res} ∈ {128, 256, 512}.
"""

import re
import glob as _glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index record dtype
# ---------------------------------------------------------------------------
# Each row describes one *native* sample (i.e. one image at the native
# resolution; larger images are split into patches later by the pipeline).
_INDEX_DTYPE = np.dtype([
    ('file_path', 'U256'),   # absolute path to HDF5 file
    ('resolution', np.int32), # 128, 256, or 512
    ('image_idx', np.int32),  # index within the resolution group
    ('k_min', np.float32),   # regression label
])


class DataIndexer:
    """
    Builds and stores an index of all samples across all HDF5 files.

    Args:
        dataset_dir: Directory containing the extracted HDF5 files.
        resolutions: List of image resolutions to include (e.g. [128, 256, 512]).
        split_ratios: Tuple (train, val, test) that must sum to 1.
        stratify_bins: Number of bins used to stratify k_min into discrete
                       groups for the train/val/test split.
        random_seed: Seed for reproducible splits.
    """

    def __init__(
        self,
        dataset_dir: str,
        resolutions: Optional[List[int]] = None,
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        stratify_bins: int = 32,
        random_seed: int = 42,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.resolutions = resolutions or [128, 256, 512]
        self.split_ratios = split_ratios
        self.stratify_bins = stratify_bins
        self.random_seed = random_seed

        self._index: Optional[np.ndarray] = None
        self._splits: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Scan all HDF5 files and build the index."""
        h5_files = self._discover_files()
        if not h5_files:
            raise FileNotFoundError(
                f"No HDF5 files found in {self.dataset_dir}. "
                "Make sure you extracted kmin_auto_batch.tar first."
            )
        logger.info(f"Found {len(h5_files)} HDF5 file(s) in {self.dataset_dir}")

        records: List[Tuple[str, int, int, float]] = []
        for fpath in sorted(h5_files):
            records.extend(self._index_file(fpath))

        self._index = np.array(
            records,
            dtype=_INDEX_DTYPE,
        )
        logger.info(f"Total indexed samples: {len(self._index)}")
        self._splits = self._split_index()

    @property
    def train_index(self) -> np.ndarray:
        self._require_built()
        return self._splits['train']

    @property
    def val_index(self) -> np.ndarray:
        self._require_built()
        return self._splits['val']

    @property
    def test_index(self) -> np.ndarray:
        self._require_built()
        return self._splits['test']

    @property
    def full_index(self) -> np.ndarray:
        self._require_built()
        return self._index

    def steps_per_epoch(self, split: str = 'train', batch_size: int = 32, native_res: int = 128) -> int:
        """
        Compute the number of gradient steps per epoch for a given split.

        Each image of resolution R produces (R // native_res)^2 patches.
        This is useful for passing `steps_per_epoch` to `model.fit()` when
        using `.repeat()` on the training dataset.

        Args:
            split:      'train', 'val', or 'test'.
            batch_size: Batch size used in the DataPipeline.
            native_res: Native model input resolution (default 128).

        Returns:
            Number of batches (steps) per epoch.
        """
        self._require_built()
        idx = self._splits[split]
        total_patches = sum(
            int((int(r) // native_res) ** 2) for r in idx['resolution']
        )
        return max(1, total_patches // batch_size)

    def summary(self) -> str:
        """Return a human-readable summary string of the indexed dataset."""
        self._require_built()
        lines = [
            f"DataIndexer summary ({self.dataset_dir})",
            f"  Total samples : {len(self._index)}",
            f"  Train         : {len(self._splits['train'])}",
            f"  Val           : {len(self._splits['val'])}",
            f"  Test          : {len(self._splits['test'])}",
        ]
        for res in self.resolutions:
            mask = self._index['resolution'] == res
            lines.append(f"  Resolution {res}x{res}: {mask.sum()} samples")
        kmin_vals = self._index['k_min']
        lines.append(f"  k_min range   : [{kmin_vals.min():.1f}, {kmin_vals.max():.1f}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_files(self) -> List[Path]:
        pattern = str(self.dataset_dir / "*.h5")
        return [Path(p) for p in _glob.glob(pattern)]

    def _index_file(self, fpath: Path) -> List[Tuple[str, int, int, float]]:
        """Return a list of (file_path, resolution, image_idx, k_min) tuples."""
        records = []
        str_path = str(fpath)
        try:
            with h5py.File(fpath, 'r') as f:
                for res in self.resolutions:
                    group_key = f"{res}x{res}"
                    if group_key not in f:
                        logger.debug(f"  {fpath.name}: group '{group_key}' not found, skipping")
                        continue
                    n_images = f[group_key].attrs.get('num_images', 0)
                    k_min_arr = f[f"{group_key}/parameters/k_min"][:]
                    for idx in range(int(n_images)):
                        records.append((str_path, res, idx, float(k_min_arr[idx])))
        except Exception as e:
            logger.warning(f"Failed to index {fpath}: {e}")
        return records

    def _split_index(self) -> Dict[str, np.ndarray]:
        """Stratified train/val/test split on k_min."""
        train_r, val_r, test_r = self.split_ratios
        idx = np.arange(len(self._index))
        k_min_vals = self._index['k_min']

        # Bin k_min for stratification
        bins = np.linspace(k_min_vals.min(), k_min_vals.max() + 1e-6, self.stratify_bins + 1)
        strata = np.digitize(k_min_vals, bins)

        # First split: train vs (val+test)
        idx_train, idx_valtest = train_test_split(
            idx,
            test_size=(val_r + test_r),
            stratify=strata,
            random_state=self.random_seed,
        )
        # Second split: val vs test
        strata_valtest = strata[idx_valtest]
        val_fraction = val_r / (val_r + test_r)
        idx_val, idx_test = train_test_split(
            idx_valtest,
            test_size=(1.0 - val_fraction),
            stratify=strata_valtest,
            random_state=self.random_seed,
        )

        logger.info(
            f"Split sizes — train: {len(idx_train)}, "
            f"val: {len(idx_val)}, test: {len(idx_test)}"
        )
        return {
            'train': self._index[idx_train],
            'val':   self._index[idx_val],
            'test':  self._index[idx_test],
        }

    def _require_built(self) -> None:
        if self._index is None:
            raise RuntimeError("Call DataIndexer.build() before accessing the index.")
