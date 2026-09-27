"""Result table serialization."""

from __future__ import annotations

import warnings
from pathlib import Path

from astropy.table import QTable

_FORMATS = {".ecsv": "ascii.ecsv", ".csv": "ascii.csv", ".fits": "fits"}


def write_results(table: QTable, path: Path | str) -> None:
    """
    Write a results table to disk, choosing the format from the path suffix.

    Parameters
    ----------
    table : astropy.table.QTable
        The table to write, typically the output of
        `turbulens.inference.Inferencer.predict`. Its ``.meta`` dict
        holds run provenance (input path, model path, channel
        selection, turbulens version).
    path : pathlib.Path or str
        Output path. The suffix selects the format: ``.ecsv``, ``.csv``,
        or ``.fits``.

    Raises
    ------
    ValueError
        If ``path`` does not end in one of the supported suffixes.

    Warns
    -----
    UserWarning
        If the chosen format is not ``.ecsv`` and ``table.meta`` is
        non-empty, since only ECSV round-trips table metadata; CSV and
        FITS table writers silently drop it.

    Notes
    -----
    ECSV (Enhanced Character-Separated Values) is the recommended output
    format for `turbulens` results, since it is the only one of the
    three that preserves ``table.meta`` (the run provenance) on write
    and read-back via `astropy.table.QTable.read`.

    References
    ----------
    astropy unified table I/O
        https://docs.astropy.org/en/stable/io/unified.html
    """
    path = Path(path)
    fmt = _FORMATS.get(path.suffix)
    if fmt is None:
        raise ValueError(f"Unsupported output format {path.suffix!r}; use one of {sorted(_FORMATS)}.")
    if fmt != "ascii.ecsv" and table.meta:
        warnings.warn(
            f"{path.suffix} does not preserve table metadata; use .ecsv to keep run provenance.",
            stacklevel=2,
        )
    table.write(path, format=fmt, overwrite=True)
