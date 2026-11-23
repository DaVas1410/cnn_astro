"""The terminal color theme shared by every `turbulens` CLI output.

Console output refers to these semantic style names (``label``, ``value``,
``unc``, ...) and never to raw colors, so the look is defined here, once.
`rich` drops all of it for ``NO_COLOR``, non-terminal streams and
terminals without color support; nothing here forces color on.
"""

from __future__ import annotations

from rich.theme import Theme

ACCENT = "#6da7ec"
"""str: The turbulens blue, readable on both light and dark terminal backgrounds."""

THEME = Theme(
    {
        "label": "dim",
        "value": "bold",
        "unc": "cyan",
        "title": f"bold {ACCENT}",
        "desc": "dim italic",
        "ok": "green",
        "warn": "yellow",
        "bad": "bold red",
        "path": f"underline {ACCENT}",
    }
)
"""rich.theme.Theme: Semantic styles for every table, panel and message the CLI prints."""
