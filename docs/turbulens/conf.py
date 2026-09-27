from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "turbulens"
copyright = "2026, turbulens contributors"
author = "turbulens contributors"
version = release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinxarg.ext",
]

numpydoc_show_class_members = False
autosummary_generate = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# "qe" keeps quote and ellipsis prettification but disables dash
# substitution, so ``--flag`` in prose (notably argparse help text
# rendered by sphinx-argparse) stays copy-pasteable instead of being
# rewritten into an en dash.
smartquotes_action = "qe"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
