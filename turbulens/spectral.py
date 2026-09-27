"""Classical (non-CNN) k_min/k_max estimation from an image's azimuthal Fourier power spectrum.

Two estimators, for two different kinds of image:

- `estimate_band_edges` ("band_edge" method): pyFC synthetic fractal
  images are band-limited, so power lives only within a sharp
  ``[k_min, k_max]`` cutoff. This is the same estimator used in
  ``archive/scripts/verify_spectra.py`` to validate `Inferencer`'s CNN
  predictions against a classical ground truth on synthetic data.
- `fit_inertial_range` ("power_law" method): real turbulent clouds have
  a continuous power-law inertial range, ``P(k) ~ k^alpha``, not a
  sharp cutoff, so band-edge detection is physically inappropriate for
  them. Instead, `k_min`/`k_max` are read off as the injection/
  dissipation scales where the spectrum departs from the fitted power
  law, following the method developed in
  ``archive/observational_images/io-fits.ipynb`` for real GASS HI data.

Both estimators warn rather than silently reporting a boundary value:
`estimate_band_edges` requires power to exceed an adaptively-set noise
floor (not a fixed absolute threshold, which real or quantized images
can exceed everywhere) and warns when its answer sits at the edge of
the available wavenumber grid; `fit_inertial_range` reports the fitted
line's :math:`R^2` alongside `alpha`, so a meaningless fit (e.g. on
white noise) is visibly distinguishable from a real inertial range
rather than silently producing a plausible-looking number.

`SpectralEstimator` wraps either estimator behind the same
`Inferencer`-shaped ``.predict()`` interface.
"""

from __future__ import annotations

import warnings

import numpy as np
from astropy.table import QTable


def radial_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the azimuthally averaged 2D power spectrum P(k) of an image.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D image of shape ``(H, W)``.

    Returns
    -------
    k : numpy.ndarray
        Integer radial wavenumbers, ``1`` to ``min(H, W) // 2 - 1``.
    power : numpy.ndarray
        Mean squared FFT magnitude at each radius in `k`.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    power_2d = np.abs(spectrum) ** 2
    ny, nx = power_2d.shape
    cy, cx = ny // 2, nx // 2
    y_i, x_i = np.mgrid[:ny, :nx]
    radius = np.sqrt((x_i - cx) ** 2 + (y_i - cy) ** 2).astype(int)
    k_max_radius = min(cx, cy)
    k = np.arange(1, k_max_radius)
    power = np.array([power_2d[radius == r].mean() if (radius == r).any() else 0.0 for r in k])
    return k, power


def estimate_band_edges(
    image: np.ndarray, lo_frac: float = 0.1, hi_floor: float = 1e-12,
    noise_percentile: float = 3.0, noise_multiplier: float = 3.0,
) -> dict[str, float]:
    """
    Estimate k_min and k_max from a sharp spectral band edge (synthetic data).

    Appropriate for pyFC fractal images, whose power is exactly zero
    outside ``[k_min, k_max]`` by construction. Not appropriate for real
    observational data; see `fit_inertial_range`.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D image of shape ``(H, W)``.
    lo_frac : float, optional
        `k_min` is the lowest wavenumber whose power is at least this
        fraction of the peak power. Default ``0.1``.
    hi_floor : float, optional
        A floor on `k_max`'s power threshold, relative to peak power.
        The threshold actually used is ``max(hi_floor, noise_multiplier
        * measured_noise_floor)`` -- see `noise_percentile`/`noise_multiplier`.
        Default ``1e-12``.
    noise_percentile : float, optional
        The `noise_percentile`-th percentile of the *entire* normalized
        power spectrum (not a wavenumber-position window) is used as the
        measured noise floor. A percentile over the whole spectrum is
        insensitive to exactly where the true edge falls -- unlike a
        window over the highest wavenumbers, which real `pyFC` images
        frequently have signal inside (their labeled `k_max` is often
        close to Nyquist) -- since it is typically the many genuinely
        near-zero bins (from any exact cutoff, or from natural spectral
        dips) that dominate a low percentile regardless of where in `k`
        they occur. See Notes. Default ``3.0``.
    noise_multiplier : float, optional
        The measured noise floor is multiplied by this factor to set
        the effective `hi_floor`. Default ``3.0``.

    Returns
    -------
    dict
        ``{"k_min": float, "k_max": float}``, or ``nan`` for both if the
        image has no power (e.g. is uniformly constant) or is too small
        to have any radial bins (`radial_power_spectrum` needs
        ``min(H, W) >= 3``).

    Warns
    -----
    UserWarning
        If `k_min` or `k_max` lands on the first or last available
        wavenumber: this means no real departure from peak/noise-floor
        power was found within the image's wavenumber range, so the
        reported value is the edge of the FFT grid, not a measured
        spectral feature. Common on data that is not actually
        band-limited (e.g. real observational images, or heavy noise).

    Notes
    -----
    A wavenumber-position window (e.g. "the top 10% of k") was tried
    first and rejected: since a meaningful fraction of real `pyFC`
    images have a true `k_max` inside that same window, that approach
    mistook their real signal for noise and measurably regressed
    accuracy (`k_max` R^2 dropped from 0.91 to 0.27 on
    ``data/raw/flat_4param_128x128_2000_disjoint_val.h5``). The
    whole-spectrum percentile used here does not have this failure
    mode -- see ``archive/scripts/tune_band_edges.py`` for the sweep this
    default was chosen from, and to re-tune against other data.
    """
    k, power = radial_power_spectrum(image)
    if power.size == 0:
        warnings.warn(
            "estimate_band_edges: image too small to have any radial wavenumber bins "
            "(radial_power_spectrum needs min(H, W) >= 3); returning nan for k_min/k_max.",
            stacklevel=2,
        )
        return {"k_min": float("nan"), "k_max": float("nan")}
    peak = power.max()
    if peak <= 0:
        warnings.warn(
            "estimate_band_edges: image has no power (uniformly constant, or all-zero spectrum); "
            "returning nan for k_min/k_max.",
            stacklevel=2,
        )
        return {"k_min": float("nan"), "k_max": float("nan")}
    normalized = power / peak

    noise_floor = float(np.percentile(normalized, noise_percentile))
    effective_hi_floor = max(hi_floor, noise_multiplier * noise_floor)

    above_lo = np.where(normalized >= lo_frac)[0]
    above_hi_floor = np.where(normalized > effective_hi_floor)[0]
    if len(above_lo) == 0 or len(above_hi_floor) == 0:
        warnings.warn(
            "estimate_band_edges: no wavenumber bin exceeded the lo_frac/noise-floor thresholds "
            "anywhere in the spectrum; returning nan for k_min/k_max. The image may be pure noise "
            "or far outside the range this estimator was tuned for.",
            stacklevel=2,
        )
        return {"k_min": float("nan"), "k_max": float("nan")}

    k_min, k_max = k[above_lo[0]], k[above_hi_floor[-1]]
    if k_min == k[0] or k_max == k[-1]:
        warnings.warn(
            "estimate_band_edges: k_min/k_max landed on the edge of the available wavenumber "
            f"grid (k in [{k[0]}, {k[-1]}]); no real spectral edge was found within range, so "
            "this is the FFT grid limit, not a measurement. The image may not actually be "
            "band-limited -- band-edge detection is only appropriate for synthetic, sharply "
            "band-limited data (see fit_inertial_range for real observational data).",
            stacklevel=2,
        )
    return {"k_min": float(k_min), "k_max": float(k_max)}


def fit_inertial_range(
    image: np.ndarray, k_lo_frac: float = 0.05, k_hi_frac: float = 0.70, threshold: float = 0.5,
) -> dict[str, float]:
    """
    Estimate the spectral index and inertial-range k_min/k_max of a real turbulent image.

    A power law ``P(k) ~ k^alpha`` is fit in log-log space over
    ``k in [k_lo_frac, k_hi_frac] * min(H, W) // 2`` (excluding the DC
    component and the highest-k noise). ``k_min``/``k_max`` are then the
    edges of the contiguous run of wavenumbers -- containing the fit
    range -- that stays within `threshold` log10 units (dex) of that
    fitted line: i.e. the injection and dissipation/noise scales
    bounding the inertial range, rather than a sharp cutoff.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D image of shape ``(H, W)``.
    k_lo_frac, k_hi_frac : float, optional
        Fraction of the maximum radial wavenumber defining the fit
        range, ``[k_lo_frac, k_hi_frac] * min(H, W) // 2``. Defaults
        ``0.05`` and ``0.70``, matching
        ``archive/observational_images/io-fits.ipynb``.
    threshold : float, optional
        Residual threshold, in dex, defining the inertial range around
        the fitted power law. Default ``0.5``.

    Returns
    -------
    dict
        ``{"alpha": float, "r_squared": float, "k_min": float, "k_max":
        float}``. All four are ``nan`` if fewer than 3 points fall in
        the fit range or the image is too small to have any radial
        bins. `r_squared` is the fitted line's coefficient of
        determination over the fit range: it is close to ``0`` (or
        negative) for data with no real power-law structure (e.g. white
        noise), so a caller can distinguish a physically meaningful
        `alpha` from a meaningless one fit to noise -- `alpha` alone
        looks like a real number either way.

    Warns
    -----
    UserWarning
        If no point in the fit range departs the fitted line by more
        than `threshold`: `k_min`/`k_max` then fall back to the fit
        range's own edges (``k_lo``/``k_hi``), which are configuration,
        not a measured injection/dissipation scale.

    References
    ----------
    Real HI cloud spectral analysis
        ``archive/observational_images/io-fits.ipynb``, cells F4-F5.
    """
    from scipy.stats import linregress

    k, power = radial_power_spectrum(image)
    if power.size == 0:
        return {"alpha": float("nan"), "r_squared": float("nan"), "k_min": float("nan"), "k_max": float("nan")}

    k_radius_max = k[-1] + 1  # min(cx, cy) from radial_power_spectrum
    k_lo = max(2, int(k_lo_frac * k_radius_max))
    k_hi = int(k_hi_frac * k_radius_max)
    in_fit_range = (k >= k_lo) & (k <= k_hi) & (power > 0)
    if in_fit_range.sum() < 3:
        return {"alpha": float("nan"), "r_squared": float("nan"), "k_min": float("nan"), "k_max": float("nan")}

    slope, intercept, r_value, _, _ = linregress(np.log10(k[in_fit_range].astype(float)), np.log10(power[in_fit_range]))
    fitted_log_power = intercept + slope * np.log10(k.astype(float))
    residuals = np.log10(power + 1e-30) - fitted_log_power
    in_inertial_range = np.abs(residuals) < threshold

    if not in_inertial_range.any():
        warnings.warn(
            "fit_inertial_range: no point in the fit range departs the fitted power law by more "
            f"than {threshold} dex; k_min/k_max fall back to the fit range's own edges "
            f"(k_lo={k_lo}, k_hi={k_hi}), not a measured injection/dissipation scale.",
            stacklevel=2,
        )
        return {"alpha": float(slope), "r_squared": float(r_value**2), "k_min": float(k_lo), "k_max": float(k_hi)}

    # Take the contiguous run of in-threshold wavenumbers containing the fit range, not the
    # global first/last in-threshold index: residuals can dip back under `threshold` far past
    # k_hi (e.g. where an extrapolated fit happens to re-cross a flat noise plateau), and a
    # global reduction would report that as k_max instead of the actual departure point.
    fit_indices = np.where(in_fit_range)[0]
    anchor = fit_indices[len(fit_indices) // 2]
    run_start = anchor
    while run_start > 0 and in_inertial_range[run_start - 1]:
        run_start -= 1
    run_end = anchor
    while run_end < len(in_inertial_range) - 1 and in_inertial_range[run_end + 1]:
        run_end += 1

    k_min, k_max = k[run_start], k[run_end]
    return {"alpha": float(slope), "r_squared": float(r_value**2), "k_min": float(k_min), "k_max": float(k_max)}


class SpectralEstimator:
    """
    Estimate k_min and k_max from images via classical Fourier analysis.

    A model-free counterpart to `turbulens.inference.Inferencer`: no
    ensemble or training is involved, so `predict` is deterministic and
    the returned table's ``epistemic_std`` column is always ``nan``.

    Parameters
    ----------
    method : {"band_edge", "power_law"}, optional
        Which classical estimator to use. ``"band_edge"`` (default,
        `estimate_band_edges`) is for synthetic, sharply band-limited
        pyFC images. ``"power_law"`` (`fit_inertial_range`) is for real
        observational data with a continuous power-law spectrum; its
        table gains two extra rows per image, ``alpha`` and
        ``r_squared``.
    **kwargs
        Passed to the underlying estimator function (`lo_frac`/
        `hi_floor`/`noise_frac`/`noise_multiplier` for ``"band_edge"``;
        `k_lo_frac`/`k_hi_frac`/`threshold` for ``"power_law"``).

    Raises
    ------
    ValueError
        If `method` is not ``"band_edge"`` or ``"power_law"``.
    """

    def __init__(self, method: str = "band_edge", **kwargs) -> None:
        if method not in ("band_edge", "power_law"):
            raise ValueError(f"method must be 'band_edge' or 'power_law', got {method!r}.")
        self.method = method
        self.kwargs = kwargs

    def predict(self, images: np.ndarray) -> QTable:
        """
        Estimate k_min and k_max for one image or a batch of images.

        Parameters
        ----------
        images : numpy.ndarray
            A single 2D image of shape ``(H, W)``, or a batch of shape
            ``(N, H, W)``.

        Returns
        -------
        astropy.table.QTable
            One row per ``(image, target)`` pair, with columns
            ``image_index``, ``target``, ``value``, and
            ``epistemic_std`` (always ``nan``), matching the schema of
            `turbulens.inference.Inferencer.predict`. ``target`` is
            ``"k_min"``/``"k_max"`` for ``method="band_edge"``, and
            ``"k_min"``/``"k_max"``/``"alpha"``/``"r_squared"`` for
            ``method="power_law"`` -- `alpha` and `r_squared` are
            included as rows, not only in ``table.meta``, so they
            survive non-ECSV output formats
            (`turbulens.io.results.write_results`) that drop metadata.

        Raises
        ------
        ValueError
            If ``images`` is not 2D or 3D.
        """
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[np.newaxis, :, :]
        if images.ndim != 3:
            raise ValueError(f"Expected a 2D image or a (N,H,W) batch, got shape {images.shape}.")

        estimator = estimate_band_edges if self.method == "band_edge" else fit_inertial_range
        targets = ("k_min", "k_max") if self.method == "band_edge" else ("k_min", "k_max", "alpha", "r_squared")

        image_indices = []
        target_names = []
        values = []
        epistemic_stds = []
        for image_index, image in enumerate(images):
            edges = estimator(image, **self.kwargs)
            for target in targets:
                image_indices.append(image_index)
                target_names.append(target)
                values.append(edges[target])
                epistemic_stds.append(float("nan"))

        table = QTable()
        table["image_index"] = image_indices
        table["target"] = target_names
        table["value"] = values
        table["epistemic_std"] = epistemic_stds
        return table
