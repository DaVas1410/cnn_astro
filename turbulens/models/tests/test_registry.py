import os
import time

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from turbulens.models import registry
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


def _fake_snapshot_download(tmp_path, calls, modes=("multitask",), sha="0123abcd"):
    """Stand-in for `huggingface_hub.snapshot_download` that lays out a fake cache snapshot."""
    def fake(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        snapshot = tmp_path / "hub" / "snapshots" / sha
        for mode in modes:
            member = snapshot / mode / "local_v2" / "ensemble" / "members" / "member_01_seed42"
            member.mkdir(parents=True, exist_ok=True)
            (member / "COMPLETED.json").write_text("{}")
        return str(snapshot)
    return fake


def test_fetch_downloads_only_the_multitask_ensemble_by_default(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "snapshot_download", _fake_snapshot_download(tmp_path, calls))

    fetched = EnsembleRegistry.fetch()

    assert calls[0]["repo_id"] == registry.DEFAULT_HUB_REPO
    assert calls[0]["allow_patterns"] == ["multitask/*"]
    assert fetched.members_dir == (
        tmp_path / "hub" / "snapshots" / "0123abcd" / "multitask" / "local_v2" / "ensemble" / "members"
    )
    assert fetched.repo_id == registry.DEFAULT_HUB_REPO
    assert fetched.revision == "0123abcd"


def test_fetch_downloads_only_the_requested_single_target_ensemble(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        registry, "snapshot_download", _fake_snapshot_download(tmp_path, calls, modes=("single_k_min",))
    )

    fetched = EnsembleRegistry.fetch(target="k_min")

    assert calls[0]["allow_patterns"] == ["single_k_min/*"]
    assert fetched.members_dir.parts[-4] == "single_k_min"


def test_fetch_narrows_the_download_to_an_explicit_version(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "snapshot_download", _fake_snapshot_download(tmp_path, calls))

    EnsembleRegistry.fetch(version="local_v2")

    assert calls[0]["allow_patterns"] == ["multitask/local_v2/*"]


def test_fetch_passes_repo_and_revision_through(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "snapshot_download", _fake_snapshot_download(tmp_path, calls))

    EnsembleRegistry.fetch(repo_id="someone/fork", revision="v1.0")

    assert calls[0]["repo_id"] == "someone/fork"
    assert calls[0]["revision"] == "v1.0"


def test_fetch_explains_an_offline_cache_miss(tmp_path, monkeypatch):
    def offline(repo_id, **kwargs):
        raise LocalEntryNotFoundError("no network and nothing cached")
    monkeypatch.setattr(registry, "snapshot_download", offline)

    with pytest.raises(FileNotFoundError, match="turbulens download"):
        EnsembleRegistry.fetch()


def test_fetch_raises_when_the_repo_has_no_such_ensemble(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "snapshot_download", _fake_snapshot_download(tmp_path, calls, modes=()))

    with pytest.raises(FileNotFoundError):
        EnsembleRegistry.fetch(target="sigma")
