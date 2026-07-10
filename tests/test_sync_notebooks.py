import sys
from pathlib import Path

import nbformat

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sync_notebooks as sn  # noqa: E402


def _write_nb(path: Path, source: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(source)]
    nbformat.write(nb, str(path))


def test_discover_finds_active_and_skips_checkpoints(tmp_path):
    _write_nb(tmp_path / "notebooks" / "comparison" / "a.ipynb")
    _write_nb(tmp_path / "notebooks" / "flexible_dataset_generation.ipynb")
    _write_nb(tmp_path / "notebooks" / "archive" / "old.ipynb")
    _write_nb(tmp_path / "notebooks" / "comparison" / ".ipynb_checkpoints" / "a-checkpoint.ipynb")

    found = sn.discover_notebooks(tmp_path)
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in found)

    assert "notebooks/comparison/a.ipynb" in rel
    assert "notebooks/flexible_dataset_generation.ipynb" in rel
    assert not any("archive" in r for r in rel)
    assert not any("checkpoint" in r for r in rel)
