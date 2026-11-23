"""Terminal presentation helpers shared by the `turbulens` CLI subcommands.

Every value that was not actually computed (a missing estimate, an
uncertainty a method does not provide, a non-finite float) is shown as
`MISSING` (``"-"``) -- never ``nan``/``None`` -- and numbers are rounded
to a fixed number of significant figures, or, for a value with an
uncertainty, to the uncertainty's second significant figure.

Text is kept ASCII-only (``x``, ``+/-``, ``R^2``) so output stays legible
when redirected through a non-UTF-8 stream (e.g. a cp1252 Windows pipe);
`rich` itself falls back to ASCII box borders in that case.
"""

from __future__ import annotations

import io
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.table import QTable
from astropy.wcs import FITSFixedWarning
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from turbulens._style import THEME

MISSING = "-"
"""str: Placeholder shown for any value that was not computed."""

TARGET_DESCRIPTIONS = {
    "k_min": "Lower band edge (cycles per image side)",
    "k_max": "Upper band edge (cycles per image side)",
    "sigma": "Std. dev. of linear density",
    "beta": "Spectral index",
    "alpha": "Fitted power-law slope",
    "r_squared": "Fit R^2",
}
"""dict: One-line descriptions of every target a results table can hold."""

_NON_TTY_WIDTH = 160


def make_console(stderr: bool = False) -> Console:
    """
    Build a `rich.console.Console` for CLI output.

    Parameters
    ----------
    stderr : bool, optional
        Write to standard error instead of standard output. Default `False`.

    Returns
    -------
    rich.console.Console
        A console bound to the *current* ``sys.stdout``/``sys.stderr`` (so
        pytest's ``capsys`` sees its output). When that stream is not a
        terminal (a pipe or file), the width is widened to
        ``_NON_TTY_WIDTH`` so long paths are not wrapped mid-name.
    """
    stream = sys.stderr if stderr else sys.stdout
    is_tty = hasattr(stream, "isatty") and stream.isatty()
    return Console(file=stream, width=None if is_tty else _NON_TTY_WIDTH, highlight=False, theme=THEME)


def render_text(*renderables: RenderableType, width: int = 100) -> str:
    """Render ``renderables`` to plain, uncolored text, e.g. for tests or logs."""
    buffer = io.StringIO()
    Console(file=buffer, width=width, color_system=None, highlight=False, theme=THEME).print(*renderables)
    return buffer.getvalue()


def _is_missing(value) -> bool:
    if value is None or value is np.ma.masked:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def format_number(value, sig: int = 4) -> str:
    """
    Format a number to ``sig`` significant figures, or `MISSING` if not computed.

    Integer-valued floats print without a decimal point (``48.0`` ->
    ``"48"``) and trailing zeros are dropped (``0.08000`` -> ``"0.08"``);
    magnitudes outside ``[1e-3, 1e6)`` switch to scientific notation.
    """
    if _is_missing(value):
        return MISSING
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e6:
        return str(int(value))
    magnitude = abs(value)
    if not 1e-3 <= magnitude < 1e6:
        return f"{value:.{sig - 1}e}"
    decimals = max(0, sig - 1 - math.floor(math.log10(magnitude)))
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_measurement(value, std) -> tuple[str, str]:
    """
    Format a value and its uncertainty to matching precision.

    Returns
    -------
    tuple of str
        ``(value, std)``. The uncertainty is rounded to two significant
        figures and the value to the same decimal place (``18.9305``,
        ``5.1429`` -> ``"18.9"``, ``"5.1"``). If ``std`` was not computed
        (non-finite, or not positive), the value falls back to
        `format_number` and the uncertainty is `MISSING`.
    """
    if _is_missing(value):
        return MISSING, MISSING
    if _is_missing(std) or float(std) <= 0:
        return format_number(value), MISSING
    decimals = max(0, 1 - math.floor(math.log10(float(std))))
    return f"{float(value):.{decimals}f}", f"{float(std):.{decimals}f}"


def titled_table(title: str, **kwargs) -> Table:
    """An empty rounded-border table with a bold, left-aligned title."""
    kwargs.setdefault("border_style", "bright_black")
    return Table(
        title=title, title_justify="left", title_style="title", box=box.ROUNDED,
        header_style="value", **kwargs,
    )


def key_value_table(title: str, rows) -> Table:
    """
    A titled list of ``label  value`` pairs, without grid lines.

    Labels are dim and values bold; `None` values show as `MISSING`.
    Each value may be a plain string or a `rich.text.Text` carrying its
    own styles (e.g. a dim qualifier after the value).
    """
    table = Table(
        title=title, title_justify="left", title_style="title", box=box.SIMPLE_HEAD,
        show_header=False, pad_edge=False, border_style="bright_black",
    )
    table.add_column(style="label", no_wrap=True)
    table.add_column(style="value", overflow="fold")
    for label, value in rows:
        table.add_row(label, MISSING if value is None else value if isinstance(value, Text) else str(value))
    return table


OUT_OF_RANGE_FRACTION = 0.5
"""float: Clipped-pixel fraction above which a CNN run is shown as out of the training distribution."""

POOR_FIT_R_SQUARED = 0.9
"""float: Power-law fit R² below which the fit, and the values read from it, are flagged."""

_UNICODE_NAMES = {"sigma": "σ", "beta": "β", "alpha": "α", "r_squared": "R²"}
_ASCII_NAMES = {"sigma": "sigma", "beta": "beta", "alpha": "alpha", "r_squared": "R^2"}
_FIT_TARGETS = ("k_min", "k_max", "alpha", "r_squared")


def supports_unicode(console: Console) -> bool:
    """Whether ``console`` can print the symbols the tables use (σ, ±, ✓, ⚠, ...)."""
    try:
        "σβαR²±✓⚠✗≥".encode(console.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def parameter_name(target: str, unicode: bool = True) -> str:
    """Display name of a target: ``σ``, ``β``, ``α``, ``R²`` (or their ASCII spellings); others as is."""
    return (_UNICODE_NAMES if unicode else _ASCII_NAMES).get(target, target)


def measurement_text(value, std, unicode: bool = True) -> Text:
    """
    ``value ± std`` as styled text: bold value, dim ``±``, spread in the ``unc`` style.

    The precision comes from `format_measurement` (the spread's second
    significant digit); with no spread the value alone, to 4 significant
    digits, without ``±``.
    """
    value_str, std_str = format_measurement(value, std)
    text = Text(value_str, style="value")
    if std_str != MISSING:
        text.append(" ± " if unicode else " +/- ", style="dim")
        text.append(std_str, style="unc")
    return text


def _quality(target: str, value, std, r_squared, out_of_range: bool, unicode: bool) -> Text:
    """The short per-row trust flag of `results_table`."""
    ok, warn, bad = ("✓", "⚠", "✗") if unicode else ("ok", "!", "x")
    if _is_missing(value):
        return Text("")
    if out_of_range:
        return Text(f"{bad} out of range", style="bad")
    if target in _FIT_TARGETS and r_squared is not None:
        if r_squared < POOR_FIT_R_SQUARED:
            return Text(f"{warn} poor fit", style="warn")
        if target == "r_squared":
            return Text(f"{ok} good fit", style="ok")
    if not _is_missing(std) and abs(float(std)) >= abs(float(value)):
        return Text(f"{warn} spread {'≥' if unicode else '>='} value", style="warn")
    return Text(ok, style="ok")


def results_table(table: QTable, unicode: bool = True, suggest_real_data: bool = False) -> Table:
    """
    Build a display table from a prediction `QTable`.

    Parameters
    ----------
    table : astropy.table.QTable
        Output of `turbulens.inference.Inferencer.predict` or
        `turbulens.spectral.SpectralEstimator.predict` (columns
        ``image_index``, ``target``, ``value``, ``epistemic_std``). Its
        ``meta["clipped_pixel_fraction_max"]``, when present, decides
        whether the run is out of the training distribution.
    unicode : bool, optional
        Use symbols (σ, β, ±, ✓, ...). Pass `supports_unicode` of the
        target console; `False` falls back to ASCII. Default `True`.
    suggest_real_data : bool, optional
        In the out-of-distribution caption, ask whether ``--real-data``
        was meant (for FITS input run without it). Default `False`.

    Returns
    -------
    rich.table.Table
        One row per ``(image, target)``: the parameter, ``value ± 1σ``
        (precision set by the spread), a quality flag and a description.
        The ``Image`` column is only shown for a multi-image batch. An
        out-of-distribution run (more than `OUT_OF_RANGE_FRACTION` of
        pixels clipped) gets a red border, dim rows and a red caption.
    """
    clipped = table.meta.get("clipped_pixel_fraction_max")
    out_of_range = clipped is not None and clipped > OUT_OF_RANGE_FRACTION
    r_squared = {
        int(row["image_index"]): float(row["value"])
        for row in table if str(row["target"]) == "r_squared" and not _is_missing(row["value"])
    }
    multi_image = len(set(np.asarray(table["image_index"]))) > 1
    caption = None
    if out_of_range:
        caption = Text(f"{clipped:.0%} of pixels clipped: input is outside the training range.", style="bad")
        if suggest_real_data:
            caption.append(" Did you mean --real-data?", style="bad")
    display = titled_table(
        "Predictions", border_style="bad" if out_of_range else "bright_black",
        caption=caption, caption_justify="left",
    )
    if multi_image:
        display.add_column("Image", justify="right")
    display.add_column("Parameter", style="value", no_wrap=True)
    display.add_column("Value ± 1σ" if unicode else "Value +/- 1 std", justify="right", no_wrap=True)
    display.add_column("", no_wrap=True)
    display.add_column("Description", style="desc")
    for row in table:
        target, image = str(row["target"]), int(row["image_index"])
        description = TARGET_DESCRIPTIONS.get(target, "")
        if unicode:
            description = description.replace("R^2", "R²")
        cells = [
            parameter_name(target, unicode),
            measurement_text(row["value"], row["epistemic_std"], unicode),
            _quality(target, row["value"], row["epistemic_std"], r_squared.get(image), out_of_range, unicode),
            description,
        ]
        if multi_image:
            cells.insert(0, str(image))
        display.add_row(*cells, style="dim" if out_of_range else None)
    return display


def results_rows(table: QTable) -> list[dict[str, str]]:
    """
    Format a prediction `QTable` as display-ready string rows.

    Returns
    -------
    list of dict
        One dict per row, with keys ``Image``, ``Parameter``, ``Value``,
        ``+/- (1 std)``, and ``Description``; values and uncertainties
        are rounded by `format_measurement`, and masked (not computed)
        cells are `MISSING`. Used by `results_table` and by any non-terminal
        front end (e.g. a notebook) that should show the same values.
    """
    rows = []
    for row in table:
        value, std = format_measurement(row["value"], row["epistemic_std"])
        target = str(row["target"])
        rows.append({
            "Image": str(int(row["image_index"])), "Parameter": target, "Value": value,
            "+/- (1 std)": std, "Description": TARGET_DESCRIPTIONS.get(target, ""),
        })
    return rows


def _warning_text(record: warnings.WarningMessage) -> str:
    return f"{record.category.__name__}: {' '.join(str(record.message).split())}"


def _is_routine(record: warnings.WarningMessage) -> bool:
    """Whether a warning only reports astropy normalizing a FITS header (e.g. ``'DEG' -> 'deg'``)."""
    return issubclass(record.category, FITSFixedWarning)


def message_panels(records: list[warnings.WarningMessage], verbose: bool = False,
                   skip: tuple[type[Warning], ...] = ()) -> list[Panel]:
    """
    Sort captured warnings into a Warnings panel and a Notes panel.

    Parameters
    ----------
    records : list of warnings.WarningMessage
        Warnings captured while a command ran.
    verbose : bool, optional
        List every routine note instead of counting them. Default `False`.
    skip : tuple of type, optional
        Warning categories to leave out because the output already shows
        them (e.g. clipping, when the results table is marked out of
        range). Default none.

    Returns
    -------
    list of rich.panel.Panel
        At most two panels, in this order: "Warnings" (yellow border),
        the messages the user should act on; and "Notes" (dim border),
        routine messages such as astropy's FITS header fixes
        (`astropy.wcs.FITSFixedWarning`: unit spelling, legacy ``CTYPE``
        translation), counted unless ``verbose``. Each message is
        deduplicated and collapsed onto one line, without the source-line
        echo Python's default warning display adds. Empty when nothing
        was captured.
    """
    records = [record for record in records if not issubclass(record.category, skip)]
    actionable = list(dict.fromkeys(_warning_text(r) for r in records if not _is_routine(r)))
    routine = list(dict.fromkeys(_warning_text(r) for r in records if _is_routine(r)))
    panels = []
    if actionable:
        body = Group(*(Text(f"- {message}") for message in actionable))
        panels.append(Panel(body, title="[warn]Warnings[/warn]", title_align="left", border_style="warn",
                            box=box.ROUNDED, padding=(0, 1), expand=False))
    if routine:
        if verbose:
            body = Group(*(Text(f"- {message}", style="label") for message in routine))
        else:
            plural = "fix" if len(routine) == 1 else "fixes"
            body = Text(f"{len(routine)} routine FITS header {plural} applied (--verbose to list)", style="label")
        panels.append(Panel(body, title="[label]Notes[/label]", title_align="left", border_style="bright_black",
                            box=box.ROUNDED, padding=(0, 1), expand=False))
    return panels


def file_written_text(label: str, path, summary: str = "") -> Text:
    """
    A "<label> <path>" line with a styled path and an optional dim summary.

    The path links to its ``file://`` URI, which terminals that support
    hyperlinks make clickable.
    """
    text = Text(f"{label} ")
    start = len(text)
    text.append(str(path), style="path")
    text.stylize(Style(link=Path(path).resolve().as_uri()), start, len(text))
    if summary:
        text.append(f"  · {summary}", style="label")
    return text


def results_written_text(path, table: QTable) -> Text:
    """
    The "Results written to <path>" line: a clickable path and a dim summary.

    The summary counts the parameters and, for a CNN run, the ensemble
    members; see `file_written_text`.
    """
    n_parameters = len({str(target) for target in table["target"]})
    summary = f"{n_parameters} parameter{'s' if n_parameters != 1 else ''}"
    if table.meta.get("n_members"):
        summary += f" · {table.meta['n_members']} members"
    return file_written_text("Results written to", path, summary)
