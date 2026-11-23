"""Presentation functions for `turbulens.io.breakdown.CubeBreakdown`: CLI text, JSON, HTML."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import RenderableType
from rich.text import Text

from turbulens import __version__
from turbulens.io.breakdown import CubeBreakdown
from turbulens.io.display import (
    MISSING,
    POOR_FIT_R_SQUARED,
    format_number,
    key_value_table,
    parameter_name,
    render_text,
    titled_table,
)

_POOR_FIT_NOTE = "poor fit, not meaningful"


@dataclass
class _Section:
    """One block of a breakdown: a list of label/value pairs, or a grid with a header row."""

    title: str
    columns: list[str] = field(default_factory=list)  # empty for label/value pairs
    rows: list[list[str]] = field(default_factory=list)
    right_aligned: set[int] = field(default_factory=set)
    flagged_rows: set[int] = field(default_factory=set)  # rows to show in the warn style


def _sections(breakdown: CubeBreakdown, unicode: bool = True) -> list[_Section]:
    """The breakdown's blocks, formatted exactly once for every output (terminal and HTML)."""
    unit = breakdown.unit or None
    data = _Section("Data", rows=[
        ["Source", breakdown.source_path],
        ["Shape", f"{' x '.join(str(n) for n in breakdown.shape)}  (channels x height x width)"],
        ["Unit", unit or MISSING],
    ])
    if breakdown.has_spectral_axis and breakdown.velocity_range is not None:
        low, high = breakdown.velocity_range
        rest = breakdown.rest_frequency_ghz
        data.rows += [
            ["Spectral axis", breakdown.spectral_ctype or MISSING],
            ["Velocity range", f"{low:.2f} to {high:.2f} km/s"],
            ["Rest frequency", MISSING if rest is None else f"{rest:.6f} GHz"],
        ]
    else:
        data.rows.append(["Spectral axis", "none (already-projected image)"])
    sections = [data]

    wcs = breakdown.wcs_summary
    if wcs:
        crvals, scales, units = wcs.get("crval") or [], wcs.get("pixel_scales") or [], wcs.get("cunit") or []
        sections.append(_Section(
            "WCS", columns=["Axis", "CTYPE", "CRVAL", "Pixel scale", "Unit"], right_aligned={0, 2, 3},
            rows=[
                [
                    str(index + 1), ctype or MISSING,
                    format_number(crvals[index] if index < len(crvals) else None),
                    format_number(scales[index] if index < len(scales) else None),
                    (units[index] if index < len(units) else "") or MISSING,
                ]
                for index, ctype in enumerate(wcs.get("ctype") or [])
            ],
        ))

    stats = breakdown.pixel_stats
    nan_fraction = stats.get("nan_fraction")
    sections.append(_Section(
        f"Pixel statistics ({unit})" if unit else "Pixel statistics",
        columns=["Min", "Max", "Mean", "Std", "NaN fraction"], right_aligned={0, 1, 2, 3, 4},
        rows=[[*(format_number(stats.get(key)) for key in ("min", "max", "mean", "std")),
               MISSING if nan_fraction is None else f"{nan_fraction:.2%}"]],
    ))

    band, power = breakdown.spectral_estimate_band_edge, breakdown.spectral_estimate_power_law
    if band is not None or power is not None:
        estimates = _Section(
            "Classical FFT estimates",
            columns=["Method", *(parameter_name(name, unicode) for name in ("k_min", "k_max", "alpha", "r_squared")),
                     "Appropriate for"],
            right_aligned={1, 2, 3, 4},
        )
        if band is not None:
            estimates.rows.append(["Band edge", format_number(band.get("k_min")), format_number(band.get("k_max")),
                                   MISSING, MISSING, "synthetic, band-limited data"])
        if power is not None:
            r_squared = power.get("r_squared")
            poor_fit = r_squared is not None and r_squared < POOR_FIT_R_SQUARED
            if poor_fit:
                estimates.flagged_rows.add(len(estimates.rows))
            estimates.rows.append([
                "Power law", format_number(power.get("k_min")), format_number(power.get("k_max")),
                format_number(power.get("alpha")), format_number(r_squared),
                f"{'⚠' if unicode else '!'} {_POOR_FIT_NOTE}" if poor_fit else "real clouds (inertial range)",
            ])
        sections.append(estimates)
    return sections


def breakdown_renderables(breakdown: CubeBreakdown, unicode: bool = True) -> list[RenderableType]:
    """
    Build the terminal sections for a `CubeBreakdown`.

    Parameters
    ----------
    breakdown : CubeBreakdown
        The breakdown to present.
    unicode : bool, optional
        Use symbols (α, R², ⚠) in the classical-estimates table; `False`
        falls back to ASCII. Default `True`.

    Returns
    -------
    list of rich.console.RenderableType
        Tables for the data summary, WCS axes (if any), pixel
        statistics, and the classical FFT estimates (only for a
        projected image). Any value that was not computed shows as
        ``"-"``. A power-law fit with R² below `POOR_FIT_R_SQUARED` is
        shown in the warn style and marked as not meaningful.
    """
    renderables = []
    for section in _sections(breakdown, unicode):
        if not section.columns:
            renderables.append(key_value_table(section.title, [tuple(row) for row in section.rows]))
            continue
        table = titled_table(section.title)
        for index, name in enumerate(section.columns):
            style = "value" if section.title.startswith("Classical") and index == 0 else None
            style = "desc" if section.title.startswith("Classical") and index == len(section.columns) - 1 else style
            table.add_column(name, justify="right" if index in section.right_aligned else "left", style=style,
                             no_wrap=index == 0)
        for index, row in enumerate(section.rows):
            flagged = index in section.flagged_rows
            cells = [*row[:-1], Text(row[-1], style="warn") if flagged else row[-1]]
            table.add_row(*cells, style="warn" if flagged else None)
        renderables.append(table)
    return renderables


def format_breakdown_cli(breakdown: CubeBreakdown) -> str:
    """
    Format a `CubeBreakdown` as human-readable, uncolored multi-line text.

    Parameters
    ----------
    breakdown : CubeBreakdown
        The breakdown to format.

    Returns
    -------
    str
        The sections of `breakdown_renderables`, rendered as plain
        text. The ``turbulens inspect`` CLI command prints the same
        sections directly, with color when writing to a terminal.
    """
    return render_text(*breakdown_renderables(breakdown))


def write_breakdown_json(breakdown: CubeBreakdown, path) -> None:
    """
    Write a `CubeBreakdown` to disk as indented JSON.

    Parameters
    ----------
    breakdown : CubeBreakdown
        The breakdown to serialize.
    path : str or pathlib.Path
        Output path, overwritten if it already exists.

    Notes
    -----
    Serialized with ``allow_nan=False``, so a breakdown containing a
    non-finite float (which should not occur, since `pixel_stats` is
    computed over finite pixels only) raises rather than silently
    producing invalid JSON (standard JSON has no NaN/Infinity literal).
    Values keep full precision; the terminal and HTML outputs round them.
    """
    path = Path(path)
    path.write_text(json.dumps(asdict(breakdown), indent=2, allow_nan=False))


_HTML_STYLE = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --head: #f6f8fa;
  --accent: #184f95; --warn: #9a6700; color-scheme: light dark; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1;
  --line: #30363d; --head: #151b23; --accent: #6da7ec; --warn: #d29922; } }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 880px; margin: 0 auto; padding: 32px 20px 48px; display: grid; gap: 28px; }
header h1 { margin: 0; font-size: 22px; color: var(--accent); }
header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
h2 { margin: 0 0 8px; font-size: 15px; color: var(--accent); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 12px; text-align: left; border-bottom: 1px solid var(--line); }
thead th { background: var(--head); font-weight: 600; }
.num { text-align: right; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 20px; margin: 0; }
dt { color: var(--muted); } dd { margin: 0; font-weight: 600; overflow-wrap: anywhere; }
tr.flag td { color: var(--warn); }
footer { color: var(--muted); font-size: 12px; }
"""


def _html_cell(tag: str, text: str, numeric: bool) -> str:
    """One escaped table cell; numeric cells are right-aligned."""
    css = ' class="num"' if numeric else ""
    return f"<{tag}{css}>{html.escape(text)}</{tag}>"


def write_breakdown_html(breakdown: CubeBreakdown, path) -> None:
    """
    Write a `CubeBreakdown` to disk as a small, self-contained HTML report.

    Parameters
    ----------
    breakdown : CubeBreakdown
        The breakdown to render.
    path : str or pathlib.Path
        Output path, overwritten if it already exists.

    Notes
    -----
    The report shows the same sections and the same rounded values as
    ``turbulens inspect`` in the terminal (both come from one formatter),
    as real tables, with light and dark styles that follow the viewer's
    ``prefers-color-scheme``. It needs no network: no scripts, fonts or
    external styles. Full-precision values are in the JSON report.

    Every value is passed through `html.escape`, since
    `CubeBreakdown.source_path` and other fields can originate from an
    untrusted filesystem path or FITS header value; unescaped
    interpolation into HTML would be an XSS risk if the output is ever
    served or opened from an untrusted source.
    """
    escape = html.escape
    blocks = []
    for section in _sections(breakdown):
        title = f"<h2>{escape(section.title)}</h2>"
        if not section.columns:
            pairs = "".join(f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in section.rows)
            blocks.append(f"<section>{title}<dl>{pairs}</dl></section>")
            continue

        head = "".join(_html_cell("th", name, index in section.right_aligned)
                       for index, name in enumerate(section.columns))
        body = "".join(
            f'<tr{" class=flag" if row_index in section.flagged_rows else ""}>'
            + "".join(_html_cell("td", value, index in section.right_aligned) for index, value in enumerate(row))
            + "</tr>"
            for row_index, row in enumerate(section.rows)
        )
        blocks.append(f'<section>{title}<div class="scroll"><table><thead><tr>{head}</tr></thead>'
                      f"<tbody>{body}</tbody></table></div></section>")
    content = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>turbulens inspect: {escape(Path(breakdown.source_path).name)}</title>"
        f"<style>{_HTML_STYLE}</style></head><body><main>"
        f"<header><h1>turbulens inspect</h1><p>{escape(breakdown.source_path)}</p></header>"
        + "".join(blocks)
        + f"<footer>turbulens {escape(__version__)} · full-precision values are in the --json report</footer>"
        "</main></body></html>\n"
    )
    Path(path).write_text(content, encoding="utf-8")
