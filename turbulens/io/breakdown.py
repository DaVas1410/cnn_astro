"""Structured, human- and machine-readable summaries of a `Cube`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from turbulens.io.cube import Cube, _FREQUENCY_CTYPE_PREFIXES, _find_spectral_axis
from turbulens.spectral import estimate_band_edges, fit_inertial_range


@dataclass
class CubeBreakdown:
    """
    A structured summary of a `Cube`'s shape, WCS, spectral coverage, and pixel statistics.

    Produced by `describe_cube` and consumed by
    `turbulens.io.report.format_breakdown_cli`,
    `turbulens.io.report.write_breakdown_json`, and
    `turbulens.io.report.write_breakdown_html`. Every field is a plain
    Python type (`tuple`, `str`, `float`, `dict`, or `None`), by design,
    so the dataclass round-trips through `dataclasses.asdict` and
    `json.dumps` without a custom encoder.

    Parameters
    ----------
    shape : tuple of int
        The cube's ``data.shape``.
    n_channels : int
        Number of channels, i.e. ``shape[0]``.
    has_spectral_axis : bool
        Whether the cube has a recognized spectral WCS axis.
    is_projected_image : bool
        The logical negation of `has_spectral_axis`: whether this cube
        is already a 2D projection (or a plain image with no WCS at
        all) rather than a full spectral cube.
    unit : str or None
        The cube's physical unit, stringified, or `None` if unset.
    wcs_summary : dict
        CTYPE, CRVAL, and pixel scale for each WCS axis, or ``{}`` if
        the cube has no WCS.
    velocity_range : tuple of float, or None
        ``(min, max)`` velocity in km/s spanned by the cube's channels,
        or `None` if there is no spectral axis.
    rest_frequency_ghz : float or None
        Rest frequency used to convert a frequency axis to velocity, in
        GHz, or `None` if the spectral axis was already in velocity
        units or absent.
    spectral_ctype : str or None
        The FITS CTYPE string of the spectral axis, or `None` if absent.
    pixel_stats : dict
        ``min``, ``max``, ``mean``, ``std``, and ``nan_fraction`` over
        the finite pixel values.
    spectral_estimate_band_edge : dict or None
        Classical (non-CNN) ``k_min``/``k_max`` estimate from a sharp
        spectral band edge (see `turbulens.spectral.estimate_band_edges`),
        appropriate only for synthetic, band-limited pyFC images. `None`
        if `n_channels` is greater than one -- see
        `spectral_estimate_power_law` for why.
    spectral_estimate_power_law : dict or None
        Classical (non-CNN) ``alpha``/``r_squared``/``k_min``/``k_max`` estimate from a
        fitted power-law inertial range (see
        `turbulens.spectral.fit_inertial_range`), appropriate for real
        observational clouds. `None` if `n_channels` is greater than
        one, since real telescope cubes are ``(x, y, velocity)`` and
        must be integrated down to one 2D image
        (`turbulens.io.cube.Cube.project` or
        `turbulens.io.cube.Cube.project_velocity`, as the ``infer``
        CLI's ``--integrate``/``--velocity`` flags do) before either
        Fourier estimate is meaningful; `describe_cube` never performs
        that projection itself, so a multi-channel cube gets `None`
        here rather than a misleading estimate from an arbitrary single
        channel.
    source_path : str
        The path the cube was loaded from, for provenance.
    """

    shape: tuple[int, ...]
    n_channels: int
    has_spectral_axis: bool
    is_projected_image: bool
    unit: str | None
    wcs_summary: dict
    velocity_range: tuple[float, float] | None
    rest_frequency_ghz: float | None
    spectral_ctype: str | None
    pixel_stats: dict
    spectral_estimate_band_edge: dict | None
    spectral_estimate_power_law: dict | None
    source_path: str


def _summarize_wcs(wcs) -> dict:
    """Return a plain-dict summary of a WCS: CTYPE, CRVAL, and pixel scales per axis."""
    summary = {
        "ctype": list(wcs.wcs.ctype),
        "crval": [float(value) for value in wcs.wcs.crval],
    }
    try:
        summary["pixel_scales"] = [float(scale) for scale in proj_plane_pixel_scales(wcs)]
    except Exception:
        summary["pixel_scales"] = None
    return summary


def describe_cube(cube: Cube, source_path: str) -> CubeBreakdown:
    """
    Build a `CubeBreakdown` summarizing a `Cube`'s shape, WCS, and pixel statistics.

    Parameters
    ----------
    cube : Cube
        The cube to summarize.
    source_path : str
        The path the cube was loaded from, recorded for provenance in
        the returned breakdown.

    Returns
    -------
    CubeBreakdown
        The structured summary. See `CubeBreakdown` for field
        descriptions.

    Notes
    -----
    When the spectral axis is a frequency axis rather than velocity, the
    rest frequency used for the velocity conversion (reported in
    `CubeBreakdown.rest_frequency_ghz`) follows the same fallback order
    as `turbulens.io.cube.Cube.velocity_axis`: the WCS ``RESTFRQ`` header
    value if present, otherwise the HI 21 cm line default
    (1420.405751 MHz).

    `pixel_stats` is computed over finite pixels only; if a cube has no
    finite pixels at all, every statistic is reported as `None` except
    ``nan_fraction``, which is ``1.0``.

    Both spectral estimates feed `numpy.nan_to_num`-cleaned pixels (non-finite
    values replaced with zero) into their FFT, which requires finite input;
    a cube with no power at any wavenumber (e.g. all zero) yields ``None``
    for every key in `spectral_estimate_band_edge` and
    `spectral_estimate_power_law`.
    """
    has_spectral = cube.has_spectral_axis if cube.wcs is not None else False

    velocity_range = None
    rest_frequency_ghz = None
    spectral_ctype = None
    if has_spectral:
        velocities = u.Quantity(cube.velocity_axis()).to_value(u.km / u.s)
        velocity_range = (float(velocities.min()), float(velocities.max()))
        _, spectral_ctype = _find_spectral_axis(cube.wcs)
        if spectral_ctype[:4].upper() in _FREQUENCY_CTYPE_PREFIXES:
            rest_hz = cube.wcs.wcs.restfrq if cube.wcs.wcs.restfrq else 1420.405751e6
            rest_frequency_ghz = float(rest_hz) / 1e9

    wcs_summary = _summarize_wcs(cube.wcs) if cube.wcs is not None else {}

    data = cube.data
    finite = np.isfinite(data)
    finite_values = data[finite]
    if finite_values.size:
        pixel_stats = {
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "std": float(finite_values.std()),
            "nan_fraction": float(1.0 - finite.sum() / data.size),
        }
    else:
        pixel_stats = {"min": None, "max": None, "mean": None, "std": None, "nan_fraction": 1.0}

    is_projected_image = not has_spectral
    spectral_estimate_band_edge = None
    spectral_estimate_power_law = None
    if cube.n_channels == 1:
        image = np.nan_to_num(data[0])

        edges = estimate_band_edges(image)
        spectral_estimate_band_edge = {
            "k_min": edges["k_min"] if np.isfinite(edges["k_min"]) else None,
            "k_max": edges["k_max"] if np.isfinite(edges["k_max"]) else None,
        }

        fit = fit_inertial_range(image)
        spectral_estimate_power_law = {
            "alpha": fit["alpha"] if np.isfinite(fit["alpha"]) else None,
            "r_squared": fit["r_squared"] if np.isfinite(fit["r_squared"]) else None,
            "k_min": fit["k_min"] if np.isfinite(fit["k_min"]) else None,
            "k_max": fit["k_max"] if np.isfinite(fit["k_max"]) else None,
        }

    return CubeBreakdown(
        shape=tuple(cube.data.shape),
        n_channels=cube.n_channels,
        has_spectral_axis=has_spectral,
        is_projected_image=is_projected_image,
        unit=str(cube.unit) if cube.unit is not None else None,
        wcs_summary=wcs_summary,
        velocity_range=velocity_range,
        rest_frequency_ghz=rest_frequency_ghz,
        spectral_ctype=spectral_ctype,
        pixel_stats=pixel_stats,
        spectral_estimate_band_edge=spectral_estimate_band_edge,
        spectral_estimate_power_law=spectral_estimate_power_law,
        source_path=source_path,
    )
