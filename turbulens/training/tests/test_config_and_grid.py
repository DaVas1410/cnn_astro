from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pipeline import enumerate_grid_jobs
from pipeline_lib.config import configuration_count, load_and_resolve_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_refined_multitask_grid_is_exactly_108_runs() -> None:
    config = load_and_resolve_config(CONFIGS / "config_multitask.yaml")
    assert config["experiment"]["version"] == "local_v2"
    assert config["model"]["pretrained"] == [True]
    assert config["model"]["pretrained_weights"] == {
        "resnet18": "IMAGENET1K_V1",
        "resnet34": "IMAGENET1K_V1",
        "resnet50": "IMAGENET1K_V2",
    }
    assert config["search"]["space"]["learning_rate"] == [5e-4, 1e-3, 2e-3]
    assert config["search"]["space"]["batch_size"] == [64, 128]
    assert configuration_count(config) == 108

    jobs = enumerate_grid_jobs(config)
    assert len(jobs) == 108
    assert {job["optimization"]["batch_size"] for job in jobs} == {64, 128}
    assert {job["optimization"]["learning_rate"] for job in jobs} == {5e-4, 1e-3, 2e-3}
    assert {job["model"]["pretrained"] for job in jobs} == {True}
    assert {job["model"]["pretrained_weight"] for job in jobs} == {
        "IMAGENET1K_V1", "IMAGENET1K_V2"
    }


def test_single_target_configs_share_refined_search_space() -> None:
    for target in ("k_min", "k_max", "sigma", "beta"):
        config = load_and_resolve_config(CONFIGS / f"config_single_{target}.yaml")
        assert config["task"]["targets"] == [target]
        assert configuration_count(config) == 108
        assert config["experiment"]["version"] == "local_v2"


def test_test_set_is_disabled_during_selection_stages() -> None:
    config = load_and_resolve_config(CONFIGS / "config_multitask.yaml")
    assert config["evaluation"]["grid"]["evaluate_test"] is False
    assert config["evaluation"]["candidate"]["evaluate_test"] is False
    assert config["evaluation"]["final"]["evaluate_test"] is True


def test_candidate_retraining_count_is_48() -> None:
    config = load_and_resolve_config(CONFIGS / "config_multitask.yaml")
    architectures = len(config["model"]["architectures"])
    candidates = architectures * config["search"]["selection"]["top_per_group"]
    repeats = config["candidate_training"]["repetitions_per_configuration"]
    assert candidates == 24
    assert candidates * repeats == 48


def test_grid_model_contains_explicit_dropout_policy() -> None:
    config = load_and_resolve_config(CONFIGS / "config_multitask.yaml")
    job = enumerate_grid_jobs(config)[0]
    assert job["model"]["dropout_policy"] == config["model"]["dropout_policy"]


def test_batch_size_validation_does_not_silently_truncate_float(tmp_path: Path) -> None:
    # Use YAML inheritance so the test only overrides the field under examination.
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"extends: {str((CONFIGS / 'config_multitask.yaml').resolve())}\n"
        "search:\n"
        "  space:\n"
        "    batch_size: [64.5]\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError):
        load_and_resolve_config(bad)


def test_search_rejects_removed_weighted_loss(tmp_path: Path) -> None:
    bad = tmp_path / "bad_loss.yaml"
    bad.write_text(
        f"extends: {str((CONFIGS / 'config_multitask.yaml').resolve())}\n"
        "search:\n"
        "  space:\n"
        "    loss: [weighted_smooth_l1]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only smooth_l1"):
        load_and_resolve_config(bad)


def test_selection_stage_test_isolation_is_enforced(tmp_path: Path) -> None:
    bad = tmp_path / "bad_test_leakage.yaml"
    bad.write_text(
        f"extends: {str((CONFIGS / 'config_multitask.yaml').resolve())}\n"
        "evaluation:\n"
        "  grid:\n"
        "    evaluate_test: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="held-out test isolation"):
        load_and_resolve_config(bad)


def test_child_config_can_narrow_architectures_with_inherited_weight_map(tmp_path: Path) -> None:
    narrowed = tmp_path / "narrowed.yaml"
    narrowed.write_text(
        f"extends: {str((CONFIGS / 'config_multitask.yaml').resolve())}\n"
        "model:\n"
        "  architectures: [resnet18]\n",
        encoding="utf-8",
    )
    config = load_and_resolve_config(narrowed)
    assert config["model"]["architectures"] == ["resnet18"]
    assert config["model"]["pretrained_weights"] == {"resnet18": "IMAGENET1K_V1"}
