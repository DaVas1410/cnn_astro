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
