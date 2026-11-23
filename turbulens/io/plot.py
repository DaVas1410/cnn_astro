"""Optional diagnostic figure for ``turbulens infer --plot`` and ``turbulens inspect --plot``.

Needs matplotlib, which is not a core dependency: ``pip install "turbulens[plot]"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from turbulens.spectral import radial_power_spectrum

MISSING_MATPLOTLIB = '--plot needs matplotlib, which is not installed. Install it with: pip install "turbulens[plot]"'
"""str: The message shown when ``--plot`` is used without matplotlib."""

_MAP_COLORS = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
_SPECTRUM, _EDGE, _CLIPPED, _FIT = "#2a78d6", "#eb6834", "#eb6834", "#52514e"


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(MISSING_MATPLOTLIB) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def diagnostic_figure(
    image: np.ndarray,
    path: str | Path,
    edges: dict[str, tuple[float, float | None]] | None = None,
    fit: dict[str, float] | None = None,
    clipped: np.ndarray | None = None,
    map_title: str = "Input map",
) -> None:
    """
    Save a two-panel diagnostic figure: the map and its power spectrum.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D map, exactly as it went into the estimate.
    path : str or pathlib.Path
        Output file; the format follows the suffix (``.png``, ``.pdf``, ``.svg``, ...).
    edges : dict, optional
        ``{"k_min": (value, std), "k_max": (value, std)}`` (``std`` may be
        `None`): drawn as vertical lines on the spectrum, with a band of
        ±1σ where a spread is given.
    fit : dict, optional
        A power-law fit ``{"alpha", "k_min", "k_max"}``: its line is drawn
        over the fitted range.
    clipped : numpy.ndarray, optional
        Boolean mask of pixels that hit the normalization clip boundary,
        drawn over the map.
    map_title : str, optional
        Title of the map panel. Default ``"Input map"``.

    Raises
    ------
    ImportError
        If matplotlib is not installed (message `MISSING_MATPLOTLIB`).
    """
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    image = np.asarray(image, dtype=float)
    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})

    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("turbulens", _MAP_COLORS)
    shown = ax_map.imshow(image, origin="lower", cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
    fig.colorbar(shown, ax=ax_map, fraction=0.046, pad=0.03)
    if clipped is not None and clipped.any():
        overlay = np.ma.masked_where(~clipped, np.ones_like(image))
        ax_map.imshow(overlay, origin="lower", cmap=ListedColormap([_CLIPPED]), alpha=0.75, interpolation="nearest")
        map_title += f" · {clipped.mean():.1%} clipped (orange)"
    ax_map.set_title(map_title, loc="left", fontsize=11)
    ax_map.set_xlabel("pixel")
    ax_map.set_ylabel("pixel")

    k, power = radial_power_spectrum(image if np.isfinite(image).all() else np.nan_to_num(image))
    positive = power > 0
    ax_spec.loglog(k[positive], power[positive], color=_SPECTRUM, lw=2, label="power spectrum")
    if fit is not None and all(fit.get(key) is not None for key in ("alpha", "k_min", "k_max")):
        in_range = positive & (k >= fit["k_min"]) & (k <= fit["k_max"])
        if in_range.any():
            log_k = np.log10(k[in_range].astype(float))
            intercept = float(np.mean(np.log10(power[in_range]) - fit["alpha"] * log_k))
            ax_spec.loglog(k[in_range], 10 ** (intercept + fit["alpha"] * log_k), color=_FIT, lw=1.5,
                           ls=(0, (5, 3)), label=f"power-law fit, slope {fit['alpha']:.2f}")
    for name, (value, std) in (edges or {}).items():
        if value is None or not np.isfinite(value):
            continue
        if std is not None and np.isfinite(std) and std > 0:
            ax_spec.axvspan(max(value - std, k[0] * 0.9), value + std, color=_EDGE, alpha=0.15, lw=0)
        ax_spec.axvline(value, color=_EDGE, lw=1.8, label=f"{name} = {value:.3g}" + (
            f" ± {std:.2g}" if std is not None and np.isfinite(std) and std > 0 else ""))
    ax_spec.set_title("Power spectrum", loc="left", fontsize=11)
    ax_spec.set_xlabel("k (cycles per image side)")
    ax_spec.set_ylabel("P(k)")
    ax_spec.grid(True, which="major", alpha=0.3)
    ax_spec.legend(loc="upper right", fontsize=9, frameon=False)  # a falling spectrum leaves this corner empty

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def edges_from_table(table, image_index: int = 0) -> tuple[dict, dict | None]:
    """
    Read the band edges and any power-law fit for one image out of a results `QTable`.

    Returns
    -------
    edges : dict
        ``{"k_min": (value, std), "k_max": (value, std)}`` for the targets present.
    fit : dict or None
        ``{"alpha", "k_min", "k_max"}`` when the table holds a power-law fit, else `None`.
    """
    values = {}
    for row in table:
        if int(row["image_index"]) != image_index:
            continue
        value, std = row["value"], row["epistemic_std"]
        value = None if np.ma.is_masked(value) else float(value)
        std = None if np.ma.is_masked(std) else float(std)
        values[str(row["target"])] = (value, std)
    edges = {name: values[name] for name in ("k_min", "k_max") if name in values}
    fit = None
    if "alpha" in values and values["alpha"][0] is not None:
        fit = {"alpha": values["alpha"][0], "k_min": values.get("k_min", (None,))[0],
               "k_max": values.get("k_max", (None,))[0]}
    return edges, fit
