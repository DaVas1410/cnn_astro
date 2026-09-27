import json

import numpy as np
import pytest
import torch

from turbulens.models.architecture import build_model_from_config
from turbulens.models.checkpoint import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Member,
    NormalizationStats,
    denormalize_targets,
    load_member,
    preprocess_image,
    real_observation_normalization,
)


def _write_member(member_dir, targets=("k_min",), target_ranges=None, input_standardization="none"):
    target_ranges = target_ranges or {"k_min": [1.0, 32.0]}
    member_dir.mkdir(parents=True)
    config = {
        "model": {
            "architecture": "resnet18", "in_channels": 1, "pretrained": False,
            "dropout": 0.1, "input_standardization": input_standardization,
        },
        "task": {"targets": list(targets), "target_ranges": target_ranges},
    }
    (member_dir / "configuration.json").write_text(json.dumps(config))

    model = build_model_from_config(config)
    checkpoint_dir = member_dir / "checkpoints"
    checkpoint_dir.mkdir()
    payload = {
        "model_state": model.state_dict(),
        "normalization_stats": {
            "lower_percentile": 0.5, "upper_percentile": 99.5,
            "lower_value": -1.0, "upper_value": 3.0, "value_range": 4.0,
            "finite_pixels_examined": 1000, "images_examined": 10,
            "sampling_seed": 424242, "source": "test_fixture",
        },
    }
    torch.save(payload, checkpoint_dir / "best_checkpoint.pt")
    return model


def test_load_member_returns_member_with_target_ranges_and_standardization(tmp_path):
    member_dir = tmp_path / "member_0"
    _write_member(member_dir, input_standardization="imagenet")

    member = load_member(member_dir)

    assert isinstance(member, Member)
    assert member.model.targets == ("k_min",)
    assert member.target_ranges == {"k_min": (1.0, 32.0)}
    assert member.input_standardization == "imagenet"
    assert isinstance(member.normalization, NormalizationStats)
    output = member.model(torch.zeros(1, 1, 64, 64))
    assert output.shape == (1, 1)


def test_load_member_builds_with_pretrained_false_and_does_not_mutate_config(tmp_path):
    member_dir = tmp_path / "member_0"
    _write_member(member_dir, input_standardization="none")
    config_path = member_dir / "configuration.json"
    original_config_text = config_path.read_text()
    config = json.loads(original_config_text)
    config["model"]["pretrained"] = True
    config_path.write_text(json.dumps(config))

    member = load_member(member_dir)

    assert member.model.pretrained is False
    # load_member must not mutate the on-disk config or any dict it re-reads later.
    assert json.loads(config_path.read_text())["model"]["pretrained"] is True


def test_preprocess_image_clips_and_reports_clipped_fraction():
    normalization = NormalizationStats(0.5, 99.5, 0.0, 4.0, 4.0, 1000, 10, 1, "test_fixture")
    image = np.array([[-4.0, 4.0], [8.0, 2.0]], dtype=np.float32)

    processed, clipped_fraction = preprocess_image(image, normalization, in_channels=1, input_standardization="none")

    np.testing.assert_allclose(processed, np.array([[[0.0, 1.0], [1.0, 0.5]]], dtype=np.float32))
    assert clipped_fraction == 0.75


def test_preprocess_image_replicates_channel_for_3_channel_models():
    normalization = NormalizationStats(0.5, 99.5, 0.0, 1.0, 1.0, 1000, 10, 1, "test_fixture")
    image = np.full((2, 2), 0.5, dtype=np.float32)

    processed, _ = preprocess_image(image, normalization, in_channels=3, input_standardization="none")

    assert processed.shape == (3, 2, 2)
    np.testing.assert_allclose(processed, np.full((3, 2, 2), 0.5, dtype=np.float32))


def test_preprocess_image_imagenet_standardization_single_channel():
    normalization = NormalizationStats(0.5, 99.5, 0.0, 1.0, 1.0, 1000, 10, 1, "test_fixture")
    image = np.full((2, 2), 0.5, dtype=np.float32)

    processed, clipped_fraction = preprocess_image(image, normalization, in_channels=1, input_standardization="imagenet")

    expected_value = (0.5 - float(IMAGENET_MEAN.mean())) / float(IMAGENET_STD.mean())
    assert processed.shape == (1, 2, 2)
    np.testing.assert_allclose(processed, np.full((1, 2, 2), expected_value, dtype=np.float32), rtol=1e-5)
    assert clipped_fraction == 0.0


def test_real_observation_normalization_log10_transforms_and_recenters():
    # log10 -> [0, 1, 2, 3], own [0, 100]-percentile center = (0+3)/2 = 1.5; member's center = (10+20)/2 = 15;
    # every pixel shifts by +13.5.
    image = np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float32)
    member_normalization = NormalizationStats(0.0, 100.0, 10.0, 20.0, 10.0, 1000, 10, 1, "test_fixture")

    recentered, stats = real_observation_normalization(image, member_normalization)

    np.testing.assert_allclose(recentered, np.array([[13.5, 14.5], [15.5, 16.5]], dtype=np.float32))
    # The reused scale is the member's own, unchanged -- not re-derived from this image.
    assert stats.lower_value == pytest.approx(10.0)
    assert stats.upper_value == pytest.approx(20.0)
    assert stats.value_range == pytest.approx(10.0)
    assert stats.images_examined == 1
    assert stats.source == "real_observation_recentered_fixed_training_scale"


def test_real_observation_normalization_floors_nonpositive_pixels():
    image = np.array([[-5.0, 0.0], [1.0, 100.0]], dtype=np.float32)
    member_normalization = NormalizationStats(0.5, 99.5, -1.0, 3.0, 4.0, 1000, 10, 1, "test_fixture")

    recentered, _ = real_observation_normalization(image, member_normalization)

    # Non-positive pixels are floored to the image's own smallest positive value (1.0), not to
    # an arbitrary constant, so they map to the same log10 value as that smallest positive pixel,
    # before the shared per-image shift is applied.
    assert np.all(np.isfinite(recentered))
    assert recentered[0, 0] == pytest.approx(recentered[0, 1])


def test_real_observation_normalization_preserves_relative_spread_across_images():
    # Two images with genuinely different log-density spread (one wide, one narrow) must stay
    # relatively different after recentering -- a naive per-image rescale-to-fill-[0,1] would
    # instead erase this difference, which is exactly the signal the model's `sigma` head reads.
    member_normalization = NormalizationStats(0.0, 100.0, -1.0, 1.0, 2.0, 1000, 10, 1, "test_fixture")
    narrow_image = np.array([[10.0, 10.0], [100.0, 100.0]], dtype=np.float32)  # log10 spread: 0 to 1
    wide_image = np.array([[1.0, 1.0], [10000.0, 10000.0]], dtype=np.float32)  # log10 spread: 0 to 4

    narrow_recentered, _ = real_observation_normalization(narrow_image, member_normalization)
    wide_recentered, _ = real_observation_normalization(wide_image, member_normalization)

    assert (wide_recentered.max() - wide_recentered.min()) > (narrow_recentered.max() - narrow_recentered.min())


def test_real_observation_normalization_fed_through_preprocess_image_does_not_saturate():
    # A real image whose linear scale is wildly outside a member's training-time range would clip
    # 100% of pixels if fed to preprocess_image directly with the member's own NormalizationStats;
    # log10 + per-image recentering should avoid that.
    image = np.array([[1.0, 500.0], [50.0, 300.0]], dtype=np.float32)
    member_normalization = NormalizationStats(0.5, 99.5, -1.0, 3.0, 4.0, 1000, 10, 1, "test_fixture")

    recentered, stats = real_observation_normalization(image, member_normalization)
    _, clipped_fraction = preprocess_image(recentered, stats, in_channels=1, input_standardization="none")

    assert clipped_fraction < 1.0


def test_denormalize_targets_maps_zero_one_range_to_physical_range():
    values = np.array([[0.0], [1.0], [0.5]], dtype=np.float32)

    physical = denormalize_targets(values, ["k_min"], {"k_min": (1.0, 32.0)})

    np.testing.assert_allclose(physical, np.array([[1.0], [32.0], [16.5]], dtype=np.float32))
