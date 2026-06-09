# scripts/hpc_ensemble/tests/conftest.py
import numpy as np
import h5py
import pytest


@pytest.fixture
def fake_cfg(tmp_path):
    h5_path = tmp_path / 'fake.h5'
    n = 40
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('images', data=np.random.randn(n, 128, 128).astype(np.float32))
        grp = hf.create_group('parameters')
        grp.create_dataset('k_min',  data=np.random.uniform(1, 62, n).astype(np.float32))
        grp.create_dataset('k_max',  data=np.random.uniform(5, 64, n).astype(np.float32))
        grp.create_dataset('sigma',  data=np.random.uniform(0.01, 5.0, n).astype(np.float32))

    return {
        'hpc': {'project_dir': str(tmp_path)},
        'training': {'base_seed': 42, 'batch_size': 8, 'lr': 1e-4,
                     'weight_decay': 1e-4, 'epochs': 2},
        'ensemble': {'n_members': 5},
        'loss_weights': {'k_min': 1.0, 'k_max': 2.0, 'sigma': 1.0},
        'data': {
            'file': 'fake.h5',
            'val_split': 0.1, 'test_split': 0.1,
            'img_p1': -2.0818, 'img_p99': 0.9409,
            'kmin_lo': 1.0, 'kmin_hi': 62.0,
            'kmax_lo': 5.0, 'kmax_hi': 64.0,
            'log_sigma_lo': -4.6052, 'log_sigma_hi': 1.6094,
        },
        'output': {'dir': 'outputs/hpc_ensemble'},
    }
