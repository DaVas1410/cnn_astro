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

This must finish with zero warnings.

## Layout

- `index.rst` — landing page, installation, how to obtain an ensemble, quickstart.
- `concepts.rst` — physical background for the predicted parameters.
- `cli.rst` — command-line reference, generated from the argparse parser.
- `api/index.rst` — API reference table of contents.
- `api/generated/*.rst` — one page per module, each an `automodule` with
  `:members:`. These are hand-maintained, not autosummary output; add a
  new file here and list it in `api/index.rst` when a module is added.
- `references.rst` — external and in-repository references.
