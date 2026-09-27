import json

import numpy as np
import pytest
import torch

from turbulens.inference import Inferencer
from turbulens.models.architecture import build_model_from_config


def _write_member(root, name, targets=("k_min",), target_ranges=None):
    target_ranges = target_ranges or {"k_min": [1.0, 32.0]}
    member_dir = root / name
    member_dir.mkdir(parents=True)
    config = {
        "model": {
            "architecture": "resnet18", "in_channels": 1, "pretrained": False,
            "dropout": 0.1, "input_standardization": "none",
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
    (member_dir / "COMPLETED.json").write_text("{}")


def test_predict_returns_physical_unit_prediction(tmp_path):
    root = tmp_path / "ensemble_root"
    _write_member(root, "member_0")
    _write_member(root, "member_1")

    table = Inferencer(root).predict(np.zeros((64, 64), dtype=np.float32))

    assert len(table) == 1
    row = table[0]
    assert row["target"] == "k_min"
    assert 1.0 <= float(row["value"]) <= 32.0
    assert float(row["epistemic_std"]) >= 0.0
    assert "clipped_pixel_fraction_max" in table.meta


def test_predict_accepts_a_batch_of_images(tmp_path):
    root = tmp_path / "ensemble_root"
    _write_member(root, "member_0")
    _write_member(root, "member_1")

    table = Inferencer(root).predict(np.zeros((3, 64, 64), dtype=np.float32))

    assert len(table) == 3  # 3 images x 1 target


def test_predict_rejects_invalid_image_shape(tmp_path):
    root = tmp_path / "ensemble_root"
    _write_member(root, "member_0")
    _write_member(root, "member_1")

    with pytest.raises(ValueError):
        Inferencer(root).predict(np.zeros((2, 3, 64, 64), dtype=np.float32))


def test_inferencer_requires_at_least_two_members(tmp_path):
    root = tmp_path / "ensemble_root"
    _write_member(root, "member_0")

    with pytest.raises(ValueError):
        Inferencer(root)


def test_predict_warns_when_input_is_far_out_of_training_distribution(tmp_path):
    root = tmp_path / "ensemble_root"
    _write_member(root, "member_0")
    _write_member(root, "member_1")

    with pytest.warns(UserWarning):
        Inferencer(root).predict(np.full((64, 64), 1000.0, dtype=np.float32))


def test_predict_real_data_avoids_saturation_warning_on_wildly_out_of_scale_input(tmp_path):
    root = tmp_path / "ensemble_root"
    _write_member(root, "member_0")
    _write_member(root, "member_1")
    # Training normalization range is [-1.0, 3.0]; a raw real-observation-scale image (hundreds
    # of physical units) saturates every pixel without --real-data (see the sibling warning test).
    real_image = np.linspace(100.0, 500.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = Inferencer(root).predict(real_image, real_data=True)

    assert len(table) == 1


def test_inferencer_rejects_members_with_mismatched_target_ordering(tmp_path):
    root = tmp_path / "ensemble_root"
    ranges = {"k_min": [1.0, 32.0], "k_max": [1.0, 32.0]}
    _write_member(root, "member_0", targets=("k_min", "k_max"), target_ranges=ranges)
    _write_member(root, "member_1", targets=("k_max", "k_min"), target_ranges=ranges)

    with pytest.raises(ValueError):
        Inferencer(root)


def test_predict_averages_per_member_denormalized_values_not_denormalized_average(tmp_path):
    from turbulens.models.checkpoint import denormalize_targets, load_member, preprocess_image

    root = tmp_path / "ensemble_root"
    _write_member(root, "member_0", target_ranges={"k_min": [1.0, 32.0]})
    _write_member(root, "member_1", target_ranges={"k_min": [100.0, 200.0]})

    image = np.zeros((64, 64), dtype=np.float32)
    table = Inferencer(root).predict(image)

    expected_physical_values = []
    for name in ("member_0", "member_1"):
        member = load_member(root / name)
        stacked, _ = preprocess_image(
            image, member.normalization, member.model.in_channels, member.input_standardization,
        )
        tensor = torch.from_numpy(stacked[np.newaxis, :, :, :])
        with torch.no_grad():
            raw_output = member.model(tensor).numpy()
        physical = denormalize_targets(raw_output, member.model.targets, member.target_ranges, clamp=True)
        expected_physical_values.append(float(physical[0, 0]))

    expected_value = sum(expected_physical_values) / len(expected_physical_values)

    assert float(table[0]["value"]) == pytest.approx(expected_value, abs=1e-5)
