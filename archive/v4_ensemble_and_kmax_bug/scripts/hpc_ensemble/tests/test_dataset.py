# scripts/hpc_ensemble/tests/test_dataset.py
import numpy as np
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import JointHDF5Dataset, load_splits, denorm, uncertainty_to_orig
import h5py


def test_dataset_len(fake_cfg, tmp_path):
    h5_path = tmp_path / 'fake.h5'
    with h5py.File(h5_path, 'r') as hf:
        n = hf['images'].shape[0]
        params = {
            'k_min': hf['parameters']['k_min'][:],
            'k_max': hf['parameters']['k_max'][:],
            'sigma': hf['parameters']['sigma'][:],
        }
    indices = np.arange(n)
    ds = JointHDF5Dataset(h5_path, indices, params, fake_cfg)
    assert len(ds) == n


def test_dataset_getitem_shapes(fake_cfg, tmp_path):
    h5_path = tmp_path / 'fake.h5'
    with h5py.File(h5_path, 'r') as hf:
        params = {k: hf['parameters'][k][:] for k in ('k_min', 'k_max', 'sigma')}
    indices = np.arange(10)
    ds = JointHDF5Dataset(h5_path, indices, params, fake_cfg)
    img, label = ds[0]
    assert img.shape == (1, 128, 128)
    assert label.shape == (3,)
    assert img.dtype == torch.float32
    assert label.dtype == torch.float32


def test_dataset_labels_normalised(fake_cfg, tmp_path):
    h5_path = tmp_path / 'fake.h5'
    with h5py.File(h5_path, 'r') as hf:
        params = {k: hf['parameters'][k][:] for k in ('k_min', 'k_max', 'sigma')}
    indices = np.arange(40)
    ds = JointHDF5Dataset(h5_path, indices, params, fake_cfg)
    labels = ds.labels  # (N, 3)
    assert labels.min() >= -0.1, "labels should be close to [0, 1]"
    assert labels.max() <= 1.1, "labels should be close to [0, 1]"


def test_denorm_roundtrip(fake_cfg):
    norm = np.array([[0.5, 0.5, 0.5],
                     [0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0]])
    result = denorm(norm, fake_cfg)
    data = fake_cfg['data']
    # k_min
    np.testing.assert_allclose(result['k_min'][0], (data['kmin_lo'] + data['kmin_hi']) / 2, rtol=1e-5)
    np.testing.assert_allclose(result['k_min'][1], data['kmin_lo'], rtol=1e-5)
    np.testing.assert_allclose(result['k_min'][2], data['kmin_hi'], rtol=1e-5)
    # k_max
    np.testing.assert_allclose(result['k_max'][0], (data['kmax_lo'] + data['kmax_hi']) / 2, rtol=1e-5)
    np.testing.assert_allclose(result['k_max'][1], data['kmax_lo'], rtol=1e-5)
    np.testing.assert_allclose(result['k_max'][2], data['kmax_hi'], rtol=1e-5)
    # sigma — at norm=0 -> exp(log_sigma_lo), at norm=1 -> exp(log_sigma_hi)
    np.testing.assert_allclose(result['sigma'][1], np.exp(data['log_sigma_lo']), rtol=1e-5)
    np.testing.assert_allclose(result['sigma'][2], np.exp(data['log_sigma_hi']), rtol=1e-5)


def test_uncertainty_to_orig(fake_cfg):
    data = fake_cfg['data']
    # At mu_norm=0 for all, sigma_norm=1 for sigma column only
    mu_norm = np.array([[0.0, 0.0, 0.0]])
    sigma_norm_linear = np.array([[1.0, 1.0, 0.0]])
    result_linear = uncertainty_to_orig(sigma_norm_linear, mu_norm, fake_cfg)
    # k_min: 1.0 * (kmin_hi - kmin_lo)
    np.testing.assert_allclose(result_linear['k_min'][0], data['kmin_hi'] - data['kmin_lo'], rtol=1e-5)
    # k_max: 1.0 * (kmax_hi - kmax_lo)
    np.testing.assert_allclose(result_linear['k_max'][0], data['kmax_hi'] - data['kmax_lo'], rtol=1e-5)

    # sigma lognormal: mu_sigma_orig * sigma_norm * (log_sigma_hi - log_sigma_lo)
    sigma_norm_sig = np.array([[0.0, 0.0, 1.0]])
    result_sig = uncertainty_to_orig(sigma_norm_sig, mu_norm, fake_cfg)
    mu_sigma_orig = np.exp(data['log_sigma_lo'])  # mu_norm=0 -> log_sigma_lo
    expected_sigma_unc = mu_sigma_orig * 1.0 * (data['log_sigma_hi'] - data['log_sigma_lo'])
    np.testing.assert_allclose(result_sig['sigma'][0], expected_sigma_unc, rtol=1e-5)


def test_load_splits_sizes(fake_cfg, tmp_path):
    train_ds, val_ds, test_ds = load_splits(fake_cfg)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == 40
    assert len(train_ds) > len(val_ds)
    assert len(val_ds) > 0
    assert len(test_ds) > 0
