"""Result table serialization."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io.fits.verify import VerifyWarning
from astropy.table import MaskedColumn, QTable

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
        FITS table writers silently drop it. Also if writing ``.fits``
        and the table has missing (masked) values; see Notes.

    Notes
    -----
    A value that was never computed (a masked cell, such as the
    ``epistemic_std`` of a classical estimate; any stray non-finite float
    is masked too) is written as an empty cell in ``.ecsv``/``.csv``,
    never as ``nan``; ECSV reads it back as masked. FITS binary tables
    have no missing-value marker for floats other than ``NaN``, so a
    table with missing values triggers a warning recommending ``.ecsv``
    when written as ``.fits``.

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
    table = _mask_non_finite(table)
    if fmt == "fits" and any(getattr(table[name], "mask", np.False_).any() for name in table.colnames):
        warnings.warn(
            "FITS cannot mark a float as missing except with NaN, so values that were not computed "
            "are stored as NaN in this file; use .ecsv to keep them as empty (masked) cells.",
            stacklevel=2,
        )
    with warnings.catch_warnings():
        # Long metadata keys become FITS HIERARCH cards, which astropy announces once per key: expected.
        warnings.filterwarnings("ignore", message="Keyword name .* HIERARCH card", category=VerifyWarning)
        table.write(path, format=fmt, overwrite=True)


def _mask_non_finite(table: QTable) -> QTable:
    """Return a copy of `table` with every non-finite float cell masked."""
    table = table.copy(copy_data=False)
    for name in table.colnames:
        column = table[name]
        if getattr(column, "unit", None) is None and column.dtype.kind == "f":
            data = np.ma.getdata(column)
            missing = ~np.isfinite(data) | np.ma.getmaskarray(column)
            if missing.any():
                table[name] = MaskedColumn(data, mask=missing)
    return table
