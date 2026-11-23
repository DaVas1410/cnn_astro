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


def test_convert_writes_mirrored_script_with_header(tmp_path):
    nb = tmp_path / "notebooks" / "comparison" / "demo.ipynb"
    _write_nb(nb, "def add(a, b):\n    return a + b\n")
    out_root = tmp_path / "notebooks" / "_scripts"

    written = sn.convert_notebook(nb, tmp_path, out_root)

    assert written == out_root / "comparison" / "demo.py"
    text = written.read_text(encoding="utf-8")
    assert text.startswith("# AUTO-GENERATED from notebooks/comparison/demo.ipynb")
    assert "def add(a, b):" in text


def test_run_skips_bad_notebook_and_reports(tmp_path):
    good = tmp_path / "notebooks" / "comparison" / "good.ipynb"
    _write_nb(good, "y = 2\n")
    bad = tmp_path / "notebooks" / "sigma_research.ipynb"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{ this is not valid notebook json", encoding="utf-8")

    out_root = tmp_path / "notebooks" / "_scripts"
    converted, skipped = sn.run(tmp_path, out_root, run_graph=False)

    conv_rel = [p.relative_to(tmp_path).as_posix() for p in converted]
    skip_rel = [p.relative_to(tmp_path).as_posix() for p, _ in skipped]
    assert "notebooks/_scripts/comparison/good.py" in conv_rel
    assert "notebooks/sigma_research.ipynb" in skip_rel


def test_run_survives_missing_graphify(tmp_path, monkeypatch, capsys):
    _write_nb(tmp_path / "notebooks" / "comparison" / "ok.ipynb", "z = 3\n")

    def _raise_missing(*args, **kwargs):
        raise FileNotFoundError("graphify")

    monkeypatch.setattr(sn.subprocess, "run", _raise_missing)
    out_root = tmp_path / "notebooks" / "_scripts"

    # graphify absent must NOT crash the run — conversions and summary survive.
    converted, skipped = sn.run(tmp_path, out_root, run_graph=True)

    assert len(converted) == 1 and skipped == []
    assert "graphify` not found" in capsys.readouterr().out
