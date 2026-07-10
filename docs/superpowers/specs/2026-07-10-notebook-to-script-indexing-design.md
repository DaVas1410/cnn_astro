# Design: Index Active Notebooks via Converted Scripts

**Date:** 2026-07-10
**Status:** Approved (design), pending implementation plan
**Author:** DaVas1410 + Claude Code

## Problem

Graphify builds the repo knowledge graph with tree-sitter, which does **not**
parse Jupyter `.ipynb` files. A large share of active work in this project lives
in notebooks (`notebooks/comparison/`, `kmax_experiments/`, etc.), so that code
is currently invisible to the graph. We want the code inside active notebooks to
be indexable without polluting the normal workflow.

## Goal

Convert active notebooks to plain `.py` scripts (nbconvert), store them as a
mirrored tree under version control, and refresh the graphify graph — via a
single explicit helper the user runs on demand.

## Scope

### In scope — "active" notebooks (~27)

- `notebooks/comparison/*.ipynb`
- `notebooks/data_generation/*.ipynb`
- `notebooks/kmax_experiments/*.ipynb`
- `notebooks/kmin_experiments/*.ipynb`
- `notebooks/flexible_dataset_generation.ipynb`
- `notebooks/sigma_regression.ipynb`
- `notebooks/sigma_research.ipynb`

### Out of scope

- `notebooks/archive/` (retired experiments)
- `experiments/**` (all archived first-steps work)
- Re-executing notebook cells (we extract static code only — what the graph needs)
- Git hooks / filesystem watchers (YAGNI)

## Components

### 1. `scripts/sync_notebooks.py`

Single responsibility: notebooks → scripts → refresh graph.

Behavior:
- Walks the active-notebook directory list; finds `.ipynb`, skipping
  `.ipynb_checkpoints/`.
- Converts each with nbconvert's `PythonExporter` to
  `notebooks/_scripts/<mirrored-subpath>/<name>.py`. Subfolder structure is
  **mirrored** to avoid name collisions (multiple `joint_regression*`, `eval`,
  etc.).
- Prepends an auto-generated header to each `.py`:
  `# AUTO-GENERATED from <path.ipynb> — do not edit by hand`.
- After conversion, runs `graphify update .` to refresh the graph.
- Intended invocation: `uv run --with nbconvert scripts/sync_notebooks.py`
  (no dependency on the project venvs; nothing installed permanently — aligns
  with the user's uv-only preference).

### 2. `notebooks/_scripts/`

Version-controlled output directory (user's choice). Mirrors the `notebooks/`
tree. Committed to git so converted code is visible in diffs.

### 3. Documentation note

One line in `CLAUDE.md` (or README): the `.py` files in `notebooks/_scripts/`
are generated — edit the `.ipynb` and run `scripts/sync_notebooks.py`.

## Flow

```
edit notebook
  → uv run --with nbconvert scripts/sync_notebooks.py
      → convert active notebooks → notebooks/_scripts/**
      → graphify update .
      → (optional) graphify label . --backend claude-cli --missing-only
```

## Error Handling

- A corrupt/unreadable notebook is reported and **skipped** — the batch does not
  abort.
- Final summary: converted N, skipped M, with the list of skipped files.

## Future Consideration

When the final model migrates to a real Python package (`src/<package>/`), that
code is indexed natively by graphify. At that point the `_scripts/` copies of
notebooks already covered by the package can be **retired** to avoid duplicate
nodes in the graph. This design does not block that path; noted for later.

## Success Criteria

- Running the helper produces `.py` files for all active notebooks under
  `notebooks/_scripts/`, mirroring structure.
- `graphify query` surfaces symbols/functions defined in notebook code
  (e.g. a training loop or model definition living in a notebook).
- Re-running the helper is idempotent (same inputs → same outputs, no spurious
  git churn beyond genuinely changed notebooks).
