from __future__ import annotations

import numpy as np
import torch

from evaluate_ensemble import aggregate_member_predictions
from pipeline_lib.metrics import selection_scores
from pipeline_lib.model import ResNetPhysicalRegressor, resolve_resnet_weights


def test_model_forward_with_explicit_dropout_policy() -> None:
    model = ResNetPhysicalRegressor(
        architecture="resnet18",
        in_channels=1,
        pretrained=False,
        pretrained_weight=None,
        dropout=0.3,
        preserve_resolution=True,
        targets=["k_min", "k_max", "sigma", "beta"],
        shared_dimensions=[64, 32],
        head_hidden_dimension=16,
        dropout_policy={
            "shared_increment_per_layer": 0.1,
            "shared_max": 0.6,
            "head_multiplier": 0.5,
            "head_max": 0.35,
        },
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 64, 64))
    assert output.shape == (2, 4)


def test_mse_r2_selection_is_lower_is_better() -> None:
    mse = np.asarray([0.01, 0.02, 0.03])
    rmse = np.sqrt(mse)
    r2 = np.asarray([0.98, 0.95, 0.90])
    score, _ = selection_scores(
        mse,
        rmse,
        r2,
        metric="mse_r2",
        mse_weight=1.0,
        r2_weight=1.0,
        scaling="minmax",
    )
    assert np.argsort(score).tolist() == [0, 1, 2]


def test_pretrained_weight_recipe_is_explicit() -> None:
    weight = resolve_resnet_weights("resnet50", True, "IMAGENET1K_V2")
    assert weight.name == "IMAGENET1K_V2"


def test_ensemble_uses_mean_of_bounded_member_predictions() -> None:
    raw = np.asarray([[[-0.5]], [[0.8]], [[1.4]]], dtype=np.float64)
    # Physical range [0, 10], after member-wise clamping: [0, 8, 10].
    physical = np.asarray([[[0.0]], [[8.0]], [[10.0]]], dtype=np.float64)
    raw_mean, bounded_mean, physical_mean, _ = aggregate_member_predictions(raw, physical)
    assert np.allclose(raw_mean, [[( -0.5 + 0.8 + 1.4) / 3]])
    assert np.allclose(bounded_mean, [[(0.0 + 0.8 + 1.0) / 3]])
    assert np.allclose(physical_mean, [[6.0]])
    assert not np.allclose(bounded_mean, np.clip(raw_mean, 0.0, 1.0))


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    from compare_multitask_single import holm_adjust

    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] == 0.04
    assert adjusted[3] == 0.20
