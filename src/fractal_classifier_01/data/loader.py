"""
Data loading utilities for Fractal Classifier.

This module provides functions to build indices from HDF5 datasets and
perform stratified train/validation/test splits without loading all images
into memory.
"""

import numpy as np
import h5py as h5
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def build_index_array(h5_path: str) -> Tuple[List[Tuple[str, int, int]], Dict[int, int], List[int]]:
    """
    Build an in-memory index of HDF5 dataset without loading images.
    
    Scans the HDF5 file and creates a lightweight index of all images,
    storing only the group name, image index, and k-value for each sample.
    
    Args:
        h5_path: Path to HDF5 dataset file
    
    Returns:
        Tuple containing:
        - index_list: List of (group_name, image_index, k_value) for each image
        - label2class: Dictionary mapping k_value to class id (0..N-1)
        - label_names: Sorted list of distinct k-values
    
    Raises:
        FileNotFoundError: If HDF5 file doesn't exist
        ValueError: If HDF5 structure is invalid
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")
    
    index_list = []
    label_names = []
    
    logger.info(f"Building index from {h5_path}")
    
    try:
        with h5.File(h5_path, 'r') as f:
            # Find all groups starting with 'k_'
            groups = sorted([g for g in f.keys() if g.startswith('k_')])
            
            if not groups:
                raise ValueError(f"No groups starting with 'k_' found in {h5_path}")
            
            for g in groups:
                # Extract k value from group name (e.g., 'k_8' -> 8)
                try:
                    k_val = int(g.split('_')[1])
                except (IndexError, ValueError) as e:
                    logger.warning(f"Skipping group {g}: invalid format")
                    continue
                
                # Check if images dataset exists
                images_key = f"{g}/images"
                if images_key not in f:
                    logger.warning(f"Skipping group {g}: no 'images' dataset found")
                    continue
                
                # Get number of images in this group
                n = f[images_key].shape[0]
                
                # Build index entries for all images in this group
                for i in range(n):
                    index_list.append((g, int(i), int(k_val)))
                
                label_names.append(k_val)
                logger.info(f"Group {g}: {n} images, k={k_val}")
    
    except Exception as e:
        raise ValueError(f"Error reading HDF5 file {h5_path}: {e}")
    
    # Sort and deduplicate label names
    label_names = sorted(list(set(label_names)))
    
    # Create mapping from k-value to class id (0..N-1)
    label2class = {k: i for i, k in enumerate(label_names)}
    
    logger.info(f"Total samples: {len(index_list)}")
    logger.info(f"Label names: {label_names}")
    logger.info(f"Label to class mapping: {label2class}")
    
    return index_list, label2class, label_names


def read_image(h5_path: str, group: str, idx: int) -> np.ndarray:
    """
    Read a single image from HDF5 dataset.
    
    Opens the HDF5 file, reads the specified image, and returns it as a
    float32 numpy array. The file is opened and closed for each read,
    which is simple and robust but not optimized for maximum throughput.
    
    Args:
        h5_path: Path to HDF5 dataset file
        group: Group name (e.g., 'k_8')
        idx: Image index within the group
    
    Returns:
        Image as numpy array with shape (H, W) and dtype float32
    
    Raises:
        ValueError: If image cannot be read
    """
    try:
        with h5.File(h5_path, 'r') as f:
            img = f[f"{group}/images"][idx]
            img = np.array(img, dtype=np.float32)
        return img
    except Exception as e:
        raise ValueError(f"Error reading image from {h5_path}, group {group}, index {idx}: {e}")


class DataIndexer:
    """
    Manages dataset indexing and splitting for fractal classification.
    
    Provides methods to build train/validation/test splits with stratification
    and query dataset properties.
    """
    
    def __init__(self, h5_path: str):
        """
        Initialize DataIndexer with HDF5 dataset.
        
        Args:
            h5_path: Path to HDF5 dataset file
        """
        self.h5_path = h5_path
        self.indices, self.label2class, self.label_names = build_index_array(h5_path)
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
    
    def build_train_val_test_splits(self, 
                                    train_ratio: float = 0.7,
                                    val_ratio: float = 0.15,
                                    test_ratio: float = 0.15,
                                    random_state: int = 42,
                                    stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            train_ratio: Proportion of data for training (default 0.7)
            val_ratio: Proportion of data for validation (default 0.15)
            test_ratio: Proportion of data for testing (default 0.15)
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits by class
        
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
            Each is a numpy array of integer positions into self.indices
        
        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        # Build class labels for stratification
        y = np.array([self.label2class[t[2]] for t in self.indices])
        
        # First split: train vs (val + test)
        temp_ratio = val_ratio + test_ratio
        stratify_labels = y if stratify else None
        
        train_idx, temp_idx = train_test_split(
            np.arange(len(self.indices)),
            test_size=temp_ratio,
            stratify=stratify_labels,
            random_state=random_state
        )
        
        # Second split: validation vs test (split the temp set in half)
        val_proportion = val_ratio / temp_ratio
        test_proportion = test_ratio / temp_ratio
        
        temp_stratify = y[temp_idx] if stratify else None
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_proportion,
            stratify=temp_stratify,
            random_state=random_state
        )
        
        # Store splits
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        
        logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return train_idx, val_idx, test_idx
    
    def get_split_indices(self, split: str) -> np.ndarray:
        """
        Get indices for a specific split.
        
        Args:
            split: One of 'train', 'val', or 'test'
        
        Returns:
            Array of indices for the specified split
        
        Raises:
            ValueError: If split is invalid or splits haven't been built yet
        """
        if split == 'train':
            if self.train_idx is None:
                raise ValueError("Splits not built yet. Call build_train_val_test_splits() first.")
            return self.train_idx
        elif split == 'val':
            if self.val_idx is None:
                raise ValueError("Splits not built yet. Call build_train_val_test_splits() first.")
            return self.val_idx
        elif split == 'test':
            if self.test_idx is None:
                raise ValueError("Splits not built yet. Call build_train_val_test_splits() first.")
            return self.test_idx
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
    
    def get_class_distribution(self, indices: np.ndarray = None) -> Dict[int, int]:
        """
        Get class distribution for a set of indices.
        
        Args:
            indices: Array of indices to analyze. If None, uses all indices.
        
        Returns:
            Dictionary mapping k-value to count
        """
        if indices is None:
            indices = np.arange(len(self.indices))
        
        distribution = {}
        for idx in indices:
            k_val = self.indices[idx][2]
            distribution[k_val] = distribution.get(k_val, 0) + 1
        
        return distribution
    
    def get_subset(self, indices: np.ndarray) -> List[Tuple[str, int, int]]:
        """
        Get index tuples for a subset of indices.
        
        Args:
            indices: Array of integer positions
        
        Returns:
            List of (group, idx, k_value) tuples
        """
        return [self.indices[i] for i in indices]
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.label_names)
    
    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        return len(self.indices)
