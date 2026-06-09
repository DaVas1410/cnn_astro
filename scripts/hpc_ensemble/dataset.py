# scripts/hpc_ensemble/dataset.py
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class JointHDF5Dataset(Dataset):
    """HDF5 dataset with global p1/p99 image normalisation and [0,1] target normalisation."""

    def __init__(self, h5_path, indices: np.ndarray, params: Dict[str, np.ndarray], cfg: dict):
        self.h5_path = str(h5_path)
        self.indices = indices
        data = cfg['data']
        self.img_p1 = data['img_p1']
        self.img_range = float(data['img_p99'] - data['img_p1']) + 1e-8

        kmin_n = (params['k_min'][indices] - data['kmin_lo']) / (data['kmin_hi'] - data['kmin_lo'])
        kmax_n = (params['k_max'][indices] - data['kmax_lo']) / (data['kmax_hi'] - data['kmax_lo'])
        log_sig = np.log(np.clip(params['sigma'][indices].astype(np.float64), 1e-9, None))
        sig_n = (log_sig - data['log_sigma_lo']) / (data['log_sigma_hi'] - data['log_sigma_lo'])

        self.labels = np.stack([kmin_n, kmax_n, sig_n], axis=1).astype(np.float32)  # (N, 3)
        self._h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        img = self._h5['images'][self.indices[idx]].astype(np.float32)
        img = (img - self.img_p1) / self.img_range
        return torch.from_numpy(img[np.newaxis]), torch.from_numpy(self.labels[idx])

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def denorm(pred_norm: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """Convert (N, 3) normalised predictions to original parameter units.

    Returns dict with keys 'k_min', 'k_max', 'sigma'.
    """
    d = cfg['data']
    kmin = pred_norm[:, 0] * (d['kmin_hi'] - d['kmin_lo']) + d['kmin_lo']
    kmax = pred_norm[:, 1] * (d['kmax_hi'] - d['kmax_lo']) + d['kmax_lo']
    sigma = np.exp(pred_norm[:, 2] * (d['log_sigma_hi'] - d['log_sigma_lo']) + d['log_sigma_lo'])
    return {'k_min': kmin, 'k_max': kmax, 'sigma': sigma}


def sigma_uncertainty_to_orig(sigma_norm: np.ndarray, mu_norm: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert normalised σ for the 'sigma' parameter to original sigma units.

    Uses the lognormal approximation: σ_orig ≈ μ_sigma_orig * σ_log_sigma
    where σ_log_sigma = σ_norm * (log_sigma_hi - log_sigma_lo).
    """
    d = cfg['data']
    mu_sigma_orig = np.exp(mu_norm * (d['log_sigma_hi'] - d['log_sigma_lo']) + d['log_sigma_lo'])
    sigma_log_sigma = sigma_norm * (d['log_sigma_hi'] - d['log_sigma_lo'])
    return mu_sigma_orig * sigma_log_sigma


def uncertainty_to_orig(sigma_norm: np.ndarray, mu_norm: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """Convert (N, 3) normalised uncertainty to original parameter units.

    Returns dict with keys 'k_min', 'k_max', 'sigma'.
    """
    d = cfg['data']
    return {
        'k_min':  sigma_norm[:, 0] * (d['kmin_hi'] - d['kmin_lo']),
        'k_max':  sigma_norm[:, 1] * (d['kmax_hi'] - d['kmax_lo']),
        'sigma':  sigma_uncertainty_to_orig(sigma_norm[:, 2], mu_norm[:, 2], cfg),
    }


def load_splits(cfg: dict):
    """Load HDF5 and return (train_ds, val_ds, test_ds)."""
    data_cfg = cfg['data']
    h5_path = Path(cfg['hpc']['project_dir']) / data_cfg['file']
    seed = cfg['training']['base_seed']

    with h5py.File(h5_path, 'r') as hf:
        n = hf['images'].shape[0]
        params = {k: hf['parameters'][k][:] for k in ('k_min', 'k_max', 'sigma')}

    indices = np.arange(n)
    test_ratio = data_cfg['val_split'] + data_cfg['test_split']
    train_idx, tmp = train_test_split(indices, test_size=test_ratio, random_state=seed)
    val_ratio = data_cfg['val_split'] / test_ratio
    val_idx, test_idx = train_test_split(tmp, test_size=1.0 - val_ratio, random_state=seed)

    kwargs = dict(params=params, cfg=cfg)
    return (
        JointHDF5Dataset(h5_path, train_idx, **kwargs),
        JointHDF5Dataset(h5_path, val_idx,   **kwargs),
        JointHDF5Dataset(h5_path, test_idx,  **kwargs),
    )
