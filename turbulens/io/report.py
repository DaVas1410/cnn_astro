"""Presentation functions for `turbulens.io.breakdown.CubeBreakdown`: CLI text, JSON, HTML."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from turbulens.io.breakdown import CubeBreakdown


def format_breakdown_cli(breakdown: CubeBreakdown) -> str:
    """
    Format a `CubeBreakdown` as human-readable multi-line text.

    Parameters
    ----------
    breakdown : CubeBreakdown
        The breakdown to format.

    Returns
    -------
    str
        A multi-line summary: source path, shape, unit, spectral
        coverage (or a note that none exists), the classical band-edge
        and power-law-inertial-range FFT k_min/k_max estimates (only
        for a projected image; omitted otherwise), and pixel
        statistics. Used directly by the ``turbulens inspect`` CLI
        command.
    """
    lines = [
        f"Source: {breakdown.source_path}",
        f"Shape: {breakdown.shape} (n_channels={breakdown.n_channels})",
        f"Unit: {breakdown.unit or '(none)'}",
    ]
    if breakdown.has_spectral_axis and breakdown.velocity_range is not None:
        low, high = breakdown.velocity_range
        lines.append(f"Velocity range: {low:.2f} to {high:.2f} km/s (CTYPE={breakdown.spectral_ctype})")
        if breakdown.rest_frequency_ghz is not None:
            lines.append(f"Rest frequency used: {breakdown.rest_frequency_ghz:.6f} GHz")
    else:
        lines.append("No spectral axis (already-projected image).")
    if breakdown.spectral_estimate_band_edge is not None:
        estimate = breakdown.spectral_estimate_band_edge
        lines.append(
            f"Classical FFT estimate (band edge, synthetic data): "
            f"k_min={estimate['k_min']}, k_max={estimate['k_max']}"
        )
    if breakdown.spectral_estimate_power_law is not None:
        fit = breakdown.spectral_estimate_power_law
        lines.append(
            f"Classical FFT estimate (power-law inertial range, real clouds): "
            f"alpha={fit['alpha']}, r_squared={fit['r_squared']}, "
            f"k_min={fit['k_min']}, k_max={fit['k_max']}"
        )
    stats = breakdown.pixel_stats
    lines.append(
        f"Pixel stats: min={stats['min']}, max={stats['max']}, mean={stats['mean']}, "
        f"std={stats['std']}, nan_fraction={stats['nan_fraction']:.4f}"
    )
    return "\n".join(lines)


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
    """
    path = Path(path)
    path.write_text(json.dumps(asdict(breakdown), indent=2, allow_nan=False))


def write_breakdown_html(breakdown: CubeBreakdown, path) -> None:
    """
    Write a `CubeBreakdown` to disk as a minimal HTML table.

    Parameters
    ----------
    breakdown : CubeBreakdown
        The breakdown to render.
    path : str or pathlib.Path
        Output path, overwritten if it already exists.

    Notes
    -----
    Every field name and value is passed through `html.escape` before
    being written, since `CubeBreakdown.source_path` and other fields
    can originate from an untrusted filesystem path or FITS header
    value; unescaped interpolation into HTML would be an XSS risk if the
    output is ever served or opened from an untrusted source.
    """
    path = Path(path)
    rows = "".join(
        f"<tr><th>{html.escape(str(field))}</th><td>{html.escape(str(value))}</td></tr>"
        for field, value in asdict(breakdown).items()
    )
    content = (
        "<html><head><title>turbulens cube breakdown</title></head>"
        f"<body><table border=\"1\">{rows}</table></body></html>"
    )
    path.write_text(content)
