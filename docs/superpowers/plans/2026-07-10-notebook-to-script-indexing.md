# Notebook-to-Script Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert active Jupyter notebooks to mirrored `.py` scripts so graphify (tree-sitter) can index their code, driven by one on-demand helper.

**Architecture:** A single Python module `scripts/sync_notebooks.py` exposes pure functions — `discover_notebooks`, `convert_notebook`, `run` — plus a `main` CLI. It uses nbconvert's `PythonExporter` to write `notebooks/_scripts/<mirrored-path>.py`, then calls `graphify update .`. Run via `uv run --with nbconvert` so nothing is installed permanently.

**Tech Stack:** Python 3.11+, nbconvert (`PythonExporter`), nbformat (tests only), pytest, graphify CLI, uv.

## Global Constraints

- Invocation is always `uv run --with nbconvert scripts/sync_notebooks.py` — never assume nbconvert is on the base interpreter. Copy verbatim.
- Active notebook set (exact — 26 files, mirror subfolders):
  - `notebooks/comparison/` (13 `.ipynb`)
  - `notebooks/data_generation/` (4 `.ipynb`)
  - `notebooks/kmax_experiments/` (3 `.ipynb`)
  - `notebooks/kmin_experiments/` (3 `.ipynb`, incl. `resnet_nopatchs _discrete.ipynb` — name contains a space)
  - `notebooks/flexible_dataset_generation.ipynb`, `notebooks/sigma_regression.ipynb`, `notebooks/sigma_research.ipynb`
- Excluded: `notebooks/archive/**`, `experiments/**`, any `.ipynb_checkpoints/`.
- Output tree: `notebooks/_scripts/`, version-controlled, mirrors `notebooks/` structure.
- Each generated `.py` starts with: `# AUTO-GENERATED from <relative/path.ipynb> — do not edit by hand`.
- A failed/corrupt notebook is skipped (never aborts the batch); final summary reports converted N, skipped M with names.
- Tests run with: `uv run --with nbconvert --with pytest python -m pytest <path> -v`.
- Follow repo convention: scripts add `src/` paths via `sys.path.insert` where needed (not needed here, but do not break it).

---

### Task 1: Notebook discovery

**Files:**
- Create: `scripts/sync_notebooks.py`
- Test: `tests/test_sync_notebooks.py`

**Interfaces:**
- Produces:
  - `ACTIVE_SPECS: list[str]` — relative dir/file globs defining the active set.
  - `discover_notebooks(repo_root: Path) -> list[Path]` — returns sorted absolute `.ipynb` paths under the active set, excluding any path containing `.ipynb_checkpoints`. Directory specs (ending `/`) are searched one level deep; file specs are included directly if they exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sync_notebooks.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with nbconvert --with pytest python -m pytest tests/test_sync_notebooks.py::test_discover_finds_active_and_skips_checkpoints -v`
Expected: FAIL (`ModuleNotFoundError: sync_notebooks` or `AttributeError: discover_notebooks`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sync_notebooks.py
"""Convert active notebooks to mirrored .py scripts and refresh the graphify graph.

Run with:  uv run --with nbconvert scripts/sync_notebooks.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Relative specs defining the "active" notebook set. Dir specs end with "/".
ACTIVE_SPECS: list[str] = [
    "notebooks/comparison/",
    "notebooks/data_generation/",
    "notebooks/kmax_experiments/",
    "notebooks/kmin_experiments/",
    "notebooks/flexible_dataset_generation.ipynb",
    "notebooks/sigma_regression.ipynb",
    "notebooks/sigma_research.ipynb",
]


def discover_notebooks(repo_root: Path) -> list[Path]:
    found: set[Path] = set()
    for spec in ACTIVE_SPECS:
        if spec.endswith("/"):
            base = repo_root / spec.rstrip("/")
            if base.is_dir():
                for nb in base.glob("*.ipynb"):
                    found.add(nb)
        else:
            nb = repo_root / spec
            if nb.is_file():
                found.add(nb)
    return sorted(p for p in found if ".ipynb_checkpoints" not in p.parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with nbconvert --with pytest python -m pytest tests/test_sync_notebooks.py::test_discover_finds_active_and_skips_checkpoints -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/sync_notebooks.py tests/test_sync_notebooks.py
git commit -m "feat(sync-notebooks): discover active notebooks"
```

---

### Task 2: Convert a notebook to a mirrored script

**Files:**
- Modify: `scripts/sync_notebooks.py`
- Test: `tests/test_sync_notebooks.py`

**Interfaces:**
- Consumes: `discover_notebooks`, `ACTIVE_SPECS`.
- Produces:
  - `convert_notebook(nb_path: Path, repo_root: Path, out_root: Path) -> Path` — converts one notebook with `PythonExporter`, writes `<out_root>/<nb_path relative to repo_root/notebooks>.py` (mirroring the sub-tree under `notebooks/`), prepends the AUTO-GENERATED header, returns the written `.py` path. Raises on unreadable/invalid notebooks.

- [ ] **Step 1: Write the failing test**

```python
def test_convert_writes_mirrored_script_with_header(tmp_path):
    nb = tmp_path / "notebooks" / "comparison" / "demo.ipynb"
    _write_nb(nb, "def add(a, b):\n    return a + b\n")
    out_root = tmp_path / "notebooks" / "_scripts"

    written = sn.convert_notebook(nb, tmp_path, out_root)

    assert written == out_root / "comparison" / "demo.py"
    text = written.read_text(encoding="utf-8")
    assert text.startswith("# AUTO-GENERATED from notebooks/comparison/demo.ipynb")
    assert "def add(a, b):" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with nbconvert --with pytest python -m pytest tests/test_sync_notebooks.py::test_convert_writes_mirrored_script_with_header -v`
Expected: FAIL (`AttributeError: convert_notebook`).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/sync_notebooks.py`:

```python
def convert_notebook(nb_path: Path, repo_root: Path, out_root: Path) -> Path:
    from nbconvert import PythonExporter  # local import: only needed under `uv run --with nbconvert`

    rel_to_notebooks = nb_path.relative_to(repo_root / "notebooks")
    out_path = out_root / rel_to_notebooks.with_suffix(".py")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exporter = PythonExporter()
    body, _ = exporter.from_filename(str(nb_path))

    rel_nb = nb_path.relative_to(repo_root).as_posix()
    header = f"# AUTO-GENERATED from {rel_nb} — do not edit by hand\n"
    out_path.write_text(header + body, encoding="utf-8")
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with nbconvert --with pytest python -m pytest tests/test_sync_notebooks.py::test_convert_writes_mirrored_script_with_header -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/sync_notebooks.py tests/test_sync_notebooks.py
git commit -m "feat(sync-notebooks): convert notebook to mirrored script"
```

---

### Task 3: Orchestration with skip-on-error and summary

**Files:**
- Modify: `scripts/sync_notebooks.py`
- Test: `tests/test_sync_notebooks.py`

**Interfaces:**
- Consumes: `discover_notebooks`, `convert_notebook`.
- Produces:
  - `run(repo_root: Path, out_root: Path, run_graph: bool = True) -> tuple[list[Path], list[tuple[Path, str]]]` — converts every discovered notebook; on exception per-notebook, records `(nb_path, error_str)` and continues. When `run_graph` is True, runs `graphify update .` (cwd=repo_root) after conversions. Returns `(converted, skipped)`.
  - `main(argv: list[str] | None = None) -> int` — arg parse (`--no-graph`), calls `run`, prints `converted N / skipped M` summary with skipped names, returns 0.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with nbconvert --with pytest python -m pytest tests/test_sync_notebooks.py::test_run_skips_bad_notebook_and_reports -v`
Expected: FAIL (`AttributeError: run`).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/sync_notebooks.py`:

```python
def run(
    repo_root: Path, out_root: Path, run_graph: bool = True
) -> tuple[list[Path], list[tuple[Path, str]]]:
    converted: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for nb in discover_notebooks(repo_root):
        try:
            converted.append(convert_notebook(nb, repo_root, out_root))
        except Exception as exc:  # noqa: BLE001 — one bad notebook must not abort the batch
            skipped.append((nb, str(exc)))
    if run_graph:
        subprocess.run(["graphify", "update", "."], cwd=str(repo_root), check=False)
    return converted, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert active notebooks to scripts and refresh graphify.")
    parser.add_argument("--no-graph", action="store_true", help="skip the `graphify update .` step")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    out_root = repo_root / "notebooks" / "_scripts"
    converted, skipped = run(repo_root, out_root, run_graph=not args.no_graph)

    print(f"converted {len(converted)} / skipped {len(skipped)}")
    for nb, err in skipped:
        print(f"  SKIPPED {nb.relative_to(repo_root).as_posix()}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with nbconvert --with pytest python -m pytest tests/test_sync_notebooks.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/sync_notebooks.py tests/test_sync_notebooks.py
git commit -m "feat(sync-notebooks): orchestrate conversion with skip-on-error"
```

---

### Task 4: Run for real, document, and index

**Files:**
- Modify: `CLAUDE.md` (add generated-scripts note)
- Create: `notebooks/_scripts/**` (generated output — committed)

**Interfaces:**
- Consumes: the full `scripts/sync_notebooks.py` CLI.

- [ ] **Step 1: Generate the scripts for real**

Run: `uv run --with nbconvert scripts/sync_notebooks.py`
Expected: prints `converted N / skipped M`. Investigate any skip that is not an intentionally-broken notebook before continuing.

- [ ] **Step 2: Verify graph now contains notebook code**

Run: `graphify query "mc dropout uncertainty" --budget 600`
Expected: at least one NODE whose `src=` points under `notebooks/_scripts/`.

- [ ] **Step 3: Add the generated-scripts note to CLAUDE.md**

Under the existing "graphify" section in `CLAUDE.md`, append:

```markdown
- Notebook code is indexed via generated scripts: `notebooks/_scripts/**` are AUTO-GENERATED from the `.ipynb` files. Edit the notebook, then run `uv run --with nbconvert scripts/sync_notebooks.py` to refresh them and the graph. Never edit `_scripts/` by hand.
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/_scripts CLAUDE.md
git commit -m "feat(sync-notebooks): index active notebooks; document workflow"
```

- [ ] **Step 5 (optional): Refresh semantic community names**

Run: `graphify label . --backend claude-cli --missing-only`
Expected: only new/placeholder communities get named; existing labels preserved.

---

## Notes for the implementer

- `PythonExporter().from_filename` raises `nbformat.reader.NotJSONError` (a subclass of `Exception`) on invalid JSON — that is exactly the skip path Task 3 tests.
- The active set includes a notebook whose filename has a space (`resnet_nopatchs _discrete.ipynb`); `Path.glob("*.ipynb")` handles it — do not shell-split filenames anywhere.
- Do not add nbconvert to any `requirements*.txt`; it is supplied at runtime by `uv run --with nbconvert`.
