# turbulens documentation

Sphinx source for the `turbulens` documentation site.

## Build

From this directory (`docs/turbulens`):

```bash
pip install -r requirements.txt
make html
```

The rendered site lands in `_build/html`; open `_build/html/index.html`.

`turbulens` itself must be importable, since the API reference is built
with `autodoc`. Install it from a checkout of the repository root first:

```bash
pip install -e .[image]
```

## Strict build

CI-equivalent build with warnings treated as errors:

```bash
rm -rf _build
python -m sphinx -b html . _build/html -W
```

This must finish with zero warnings. On Windows (no `make`), run the same
`python -m sphinx ...` command.

## Content checks

`tools/check_docs.py` catches what a Sphinx build cannot: results numbers
(the docs explain the method, the paper reports results), links into
private or untracked repository paths, and bibliography entries without a
`doi`, `eprint` or `url`. From the repository root:

```bash
python docs/turbulens/tools/check_docs.py --package turbulens            # docs + package docstrings
python docs/turbulens/tools/check_docs.py --package turbulens/training   # training-module docstrings
python -m pytest docs/turbulens/tools -q                                  # the checker's own tests
```

Each `check_docs.py` run must print `check_docs: OK`. Docstrings are
checked because autodoc renders them into the API pages.

## Citations

Cite with `:cite:t:` (Author (year)) or `:cite:p:` ((Author year)) using
keys from `references.bib`. Only add an entry after checking it against
its DOI record, arXiv entry or publisher page, and give it a `doi`,
`eprint` or `url` field; the checker rejects entries without one. The
bibliography lists only cited entries, labelled author-year (see the
style registered in `conf.py`).

## Layout

- `index.rst`: landing page, install, quickstart, reading guide by audience.
- `getting_started.rst`: a first run on a real GASS cube.
- `theory/`: notation and conventions, turbulence and power spectra,
  log-normal fractal fields, projection and HI observations.
- `method/`: synthetic data, preprocessing, architecture, training,
  uncertainty, classical spectral estimators, evaluation metrics.
- `training_pipeline/`: overview, configuration, hyperparameter search,
  scientific invariants, running, outputs.
- `user_guide/`: CLI (generated from the argparse parser), Python API,
  inputs, outputs and provenance, pretrained models, interpreting results.
- `limitations.rst`, `citing.rst`, `references.rst` (+ `references.bib`).
- `api/index.rst` and `api/generated/*.rst`: one hand-maintained
  `automodule` page per module, plus one for the training-pipeline
  modules (importable because `conf.py` adds `turbulens/training` to
  `sys.path`). Add a file and list it in `api/index.rst` when a module is
  added.
- `_static/architecture.svg`: the network diagram.
- `tools/`: `check_docs.py` and its tests (excluded from the build).

The documentation explains the method and gives no results numbers;
those belong to the paper. `check_docs.py` enforces this.
