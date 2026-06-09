import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loss import WeightedGaussianNLL


@pytest.fixture
def loss_fn():
    return WeightedGaussianNLL(weights=[1.0, 2.0, 1.0])


def test_output_is_scalar(loss_fn):
    pred = torch.randn(8, 3, 2)
    target = torch.rand(8, 3)
    out = loss_fn(pred, target)
    assert out.shape == (), f"Expected scalar, got {out.shape}"


def test_loss_finite(loss_fn):
    pred = torch.randn(8, 3, 2)
    target = torch.rand(8, 3)
    out = loss_fn(pred, target)
    assert torch.isfinite(out), "Loss must be finite"


def test_k_max_upweighted():
    """k_max weight 2x should produce higher loss than k_min weight 1x for same error."""
    equal_weights = WeightedGaussianNLL(weights=[1.0, 1.0, 1.0])
    kmax_upweighted = WeightedGaussianNLL(weights=[1.0, 2.0, 1.0])
    # Create pred/target with equal errors across all params
    pred = torch.zeros(4, 3, 2)   # mu=sigmoid(0)=0.5, log_sigma=0
    target = torch.ones(4, 3) * 0.8  # constant offset from mu
    loss_equal = equal_weights(pred, target)
    loss_upweighted = kmax_upweighted(pred, target)
    assert loss_upweighted > loss_equal, "k_max upweighted loss should exceed equal-weight loss"


def test_gradients_flow(loss_fn):
    pred = torch.randn(4, 3, 2, requires_grad=True)
    target = torch.rand(4, 3)
    out = loss_fn(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_lower_sigma_increases_loss_for_large_error(loss_fn):
    """When prediction error is large, lower sigma should penalise more than larger sigma."""
    # Large error between target and mu
    target = torch.ones(4, 3) * 0.9
    # pred_a: mu=0.5 (sigmoid(0)), log_sigma=0 → sigma≈1.31 (softplus(0))
    pred_a = torch.zeros(4, 3, 2)
    # pred_b: mu=0.5, log_sigma=-2 → smaller sigma → larger NLL when error is large
    pred_b = torch.zeros(4, 3, 2)
    pred_b[..., 1] = -2.0
    loss_a = loss_fn(pred_a, target)
    loss_b = loss_fn(pred_b, target)
    assert loss_b > loss_a, "Smaller sigma with large error should have higher NLL"
