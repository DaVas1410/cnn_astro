import os
import time

import pytest

from turbulens.models.registry import EnsembleRegistry


def test_discover_members_returns_only_completed_directories(tmp_path):
    root = tmp_path / "members"
    root.mkdir()
    completed = root / "member_0"
    completed.mkdir()
    (completed / "COMPLETED.json").write_text("{}")
    (root / "member_1").mkdir()
    (root / "not_a_dir.txt").write_text("stray file")

    members = EnsembleRegistry.discover_members(root)

    assert members == [completed]


def test_discover_members_raises_when_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        EnsembleRegistry.discover_members(tmp_path / "missing")


def test_discover_members_raises_when_none_completed(tmp_path):
    root = tmp_path / "members"
    root.mkdir()
    (root / "member_0").mkdir()

    with pytest.raises(FileNotFoundError):
        EnsembleRegistry.discover_members(root)


def _make_members_dir(output_root, mode, version):
    members_dir = output_root / mode / version / "ensemble" / "members"
    members_dir.mkdir(parents=True)
    return members_dir


def test_resolve_finds_multitask_ensemble_by_default(tmp_path):
    expected = _make_members_dir(tmp_path, "multitask", "v1")

    resolved = EnsembleRegistry.resolve(tmp_path)

    assert resolved == expected


def test_resolve_finds_single_target_ensemble(tmp_path):
    expected = _make_members_dir(tmp_path, "single_k_min", "v1")

    resolved = EnsembleRegistry.resolve(tmp_path, target="k_min")

    assert resolved == expected


def test_resolve_latest_picks_most_recently_modified_version(tmp_path):
    _make_members_dir(tmp_path, "multitask", "v1")
    time.sleep(0.01)
    expected = _make_members_dir(tmp_path, "multitask", "v2")
    now = time.time()
    os.utime(tmp_path / "multitask" / "v2", (now, now))

    resolved = EnsembleRegistry.resolve(tmp_path, version="latest")

    assert resolved == expected


def test_resolve_explicit_version(tmp_path):
    _make_members_dir(tmp_path, "multitask", "v1")
    expected = _make_members_dir(tmp_path, "multitask", "v2")

    resolved = EnsembleRegistry.resolve(tmp_path, version="v2")

    assert resolved == expected


def test_resolve_raises_when_mode_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        EnsembleRegistry.resolve(tmp_path, target="k_min")
