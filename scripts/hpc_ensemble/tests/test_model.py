import torch
import torch.nn.functional as F
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ResNet50HeteroJoint


@pytest.fixture
def model():
    return ResNet50HeteroJoint(in_channels=1, n_params=3)


def test_output_shape(model):
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    assert out.shape == (4, 3, 2), f"Expected (4, 3, 2), got {out.shape}"


def test_mu_bounded(model):
    x = torch.randn(8, 1, 128, 128)
    out = model(x)
    mu = out[..., 0]  # (B, 3)
    assert mu.min().item() >= 0.0, "μ must be >= 0 (Sigmoid applied)"
    assert mu.max().item() <= 1.0, "μ must be <= 1 (Sigmoid applied)"


def test_sigma_positive_after_softplus(model):
    x = torch.randn(8, 1, 128, 128)
    out = model(x)
    log_sigma = out[..., 1]  # (B, 3)
    sigma = F.softplus(log_sigma)
    assert (sigma > 0).all(), "σ must be > 0 after softplus"


def test_gradients_flow(model):
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found"


def test_param_count(model):
    n = sum(p.numel() for p in model.parameters())
    assert n > 20_000_000, "ResNet50 should have >20M params"
    assert n < 30_000_000, "ResNet50 should have <30M params"
