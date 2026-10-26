from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# The training pipeline is a script directory with flat imports (`from pipeline_lib...`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "turbulens" / "training"))

project = "turbulens"
copyright = "2026, turbulens contributors"
author = "turbulens contributors"
try:
    version = release = importlib.metadata.version("turbulens")
except importlib.metadata.PackageNotFoundError:
    version = release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinxarg.ext",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"


def _register_author_year_bibliography_style() -> None:
    """Label bibliography entries "Author et al. YEAR" to match the author-year citations."""
    from pybtex.plugin import register_plugin
    from pybtex.style.formatting.unsrt import Style as UnsrtStyle
    from pybtex.style.labels import BaseLabelStyle

    class AuthorYearLabelStyle(BaseLabelStyle):
        def format_labels(self, sorted_entries):
            for entry in sorted_entries:
                persons = entry.persons.get("author", [])
                names = ["".join(p.last_names).replace("{", "").replace("}", "") for p in persons[:2]]
                if len(persons) > 2:
                    author = f"{names[0]} et al."
                elif len(persons) == 2:
                    author = f"{names[0]} & {names[1]}"
                else:
                    author = names[0] if names else entry.key
                yield f"{author} {entry.fields.get('year', '')}".strip()

    class AuthorYearStyle(UnsrtStyle):
        default_label_style = "authoryear"
        default_sorting_style = "author_year_title"

    register_plugin("pybtex.style.labels", "authoryear", AuthorYearLabelStyle)
    register_plugin("pybtex.style.formatting", "authoryear", AuthorYearStyle)


_register_author_year_bibliography_style()
bibtex_default_style = "authoryear"

numpydoc_show_class_members = False
autosummary_generate = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# "qe" keeps quote and ellipsis prettification but disables dash
# substitution, so ``--flag`` in prose (notably argparse help text
# rendered by sphinx-argparse) stays copy-pasteable instead of being
# rewritten into an en dash.
smartquotes_action = "qe"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "tools"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
