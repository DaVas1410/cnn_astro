"""
FITS and in-memory cube handling.

`Cube` wraps `astropy.nddata.NDDataRef` to give every array flowing
through `turbulens` a uniform 3D ``(n_channels, H, W)`` shape, whether it
originated from a spectral FITS cube, a plain 2D FITS image, or a
plain image file. A 2D input is always promoted to a single-channel
``(1, H, W)`` cube on construction, so downstream code never has to
special-case 2D vs 3D data.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SpectralCoord
from astropy.io import fits
from astropy.nddata import NDDataRef
from astropy.wcs import WCS

_VELOCITY_CTYPE_PREFIXES = ("VELO", "VOPT", "VRAD")
_FREQUENCY_CTYPE_PREFIXES = ("FREQ",)
_HI_21CM_REST_FREQUENCY = 1420.405751 * u.MHz


def _find_spectral_axis(wcs: WCS) -> tuple[int, str] | None:
    """Return the (axis index, CTYPE) of the cube's single spectral WCS axis, or None."""
    matches = []
    for index, ctype in enumerate(wcs.wcs.ctype):
        prefix = ctype[:4].upper()
        if prefix in _VELOCITY_CTYPE_PREFIXES or prefix in _FREQUENCY_CTYPE_PREFIXES:
            matches.append((index, ctype))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Multiple spectral-like WCS axes found ({matches}); turbulens requires exactly one."
        )
    index, ctype = matches[0]
    if index != wcs.naxis - 1:
        raise ValueError(
            f"turbulens requires the spectral axis to be the last WCS axis; "
            f"found {ctype!r} at axis {index} of {wcs.naxis}."
        )
    return index, ctype


def _drop_channel_axis(wcs: WCS | None, data_ndim: int) -> WCS | None:
    """Drop the trailing WCS axis for the channel dimension, if the WCS actually has one.

    A WCS built from a 2D FITS image (naxis == 2) that was promoted to a 3D cube of
    shape (1, H, W) does not describe a channel axis at all, so it is returned unchanged
    rather than having a spatial axis incorrectly dropped.
    """
    if wcs is None or wcs.naxis != data_ndim:
        return wcs
    return wcs.dropaxis(wcs.naxis - 1)


def _squeeze_degenerate_axes(data: np.ndarray, wcs: WCS | None) -> tuple[np.ndarray, WCS | None]:
    """Repeatedly drop size-1 axes from `data` (and the matching WCS axis) until it is 3D."""
    while data.ndim > 3:
        degenerate_axes = [axis for axis, size in enumerate(data.shape) if size == 1]
        if not degenerate_axes:
            raise ValueError(
                f"Cannot reduce {data.ndim}D FITS data with no size-1 axis to squeeze; shape={data.shape}."
            )
        axis = degenerate_axes[0]
        if wcs is not None:
            # WCS axis order is the reverse of the numpy array axis order.
            wcs = wcs.dropaxis(data.ndim - 1 - axis)
        data = np.squeeze(data, axis=axis)
    return data, wcs


class Cube(NDDataRef):
    """A 3D ``(n_channels, H, W)`` array with optional WCS, unit, and FITS metadata.

    `Cube` is the data structure every `turbulens` entry point (the
    `turbulens.io.image_loaders.load_input` loader, the `infer` and
    `inspect` CLI commands) normalizes input into. Its only structural
    invariant is that ``data`` is always 3D: 2D input is promoted to a
    single-channel cube of shape ``(1, H, W)`` on construction, by the
    class itself (not by `read`/`from_array` individually), so every
    code path that builds a `Cube` -- including `slice_channel` and
    `project`, which each build one directly -- gets the promotion for
    free and callers never branch on 2D vs 3D.
    """

    def __init__(self, data, *args, **kwargs) -> None:
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        super().__init__(data, *args, **kwargs)

    @classmethod
    def read(cls, path: str | Path, hdu: int | str = 0, memmap: bool = True) -> "Cube":
        """
        Read a FITS file into a `Cube`, promoting a 2D image to a single-channel cube.

        Every `turbulens` entry point (the `turbulens.io.image_loaders.load_input`
        loader, the `infer` and `inspect` CLI commands) normalizes input through
        this method or `from_array`. Both guarantee that output data is always 3D,
        so downstream code never has to special-case 2D vs 3D input: a 2D FITS
        image is promoted to a single-channel ``(1, H, W)`` cube.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the FITS file to read, including support for gzip-compressed
            ``.fits.gz`` files.
        hdu : int or str, optional
            Header data unit (HDU) index or name to read. Default is ``0`` (the
            primary HDU).
        memmap : bool, optional
            Whether to use memory-mapping when reading the FITS file. Default is
            ``True``. Set to ``False`` to force the entire array into memory
            immediately (needed for scaled data with BZERO/BSCALE/BLANK headers).

        Returns
        -------
        Cube
            A cube wrapping the data from the specified HDU, with WCS, unit, and
            metadata extracted from the FITS header. 2D input is promoted to
            shape ``(1, H, W)``; higher-dimensional input is squeezed down to 3D
            by dropping degenerate axes.

        Raises
        ------
        ValueError
            If the data is not 2D or 3D after squeezing.

        Warns
        -----
        UserWarning
            If ``BUNIT`` is present in the header but cannot be parsed as an
            `astropy.units.Unit`.

        Notes
        -----
        The WCS is constructed with ``translate_units="S"`` to handle radio data
        that writes velocity units as bare ``S`` (seconds) rather than the
        standard ``m/s``, which is common in AIPS-legacy FITS files. This is
        necessary because astropy's own `unitfix` only lowercases the recognized
        ``'M/S'`` substring to ``'m/S'``, leaving the trailing ``S`` capitalized,
        which wcslib would otherwise reject as an invalid (electrical) unit for
        a velocity axis.

        References
        ----------
        `astropy.io.fits.open`
            https://docs.astropy.org/en/stable/io/fits/api/files.html
        `astropy.wcs.WCS`
            https://docs.astropy.org/en/stable/wcs/index.html
        """
        # `memmap=True` (the default) is translated to astropy's own auto-detect
        # value (`None`) rather than passed through literally: forcing real
        # memory-mapping unconditionally makes astropy raise on scaled data
        # (BZERO/BSCALE/BLANK present, as in the real GASS sample files) even
        # though astropy's own default handles that case fine by silently
        # reading such files into memory.
        with fits.open(path, memmap=None if memmap else False) as hdul:
            raw = np.asarray(hdul[hdu].data, dtype=np.float32)
            header = hdul[hdu].header.copy()

        # `translate_units="S"` tells wcslib to read a bare 'S' unit symbol as
        # seconds of time rather than siemens. This matters for real radio data:
        # astropy's own unitfix only lowercases the recognized 'M/S' -> 'm/S'
        # substring, leaving the trailing 'S' capitalized, which wcslib would
        # otherwise reject as an invalid (electrical) unit for a velocity axis.
        wcs = WCS(header, translate_units="S")
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        elif raw.ndim > 3:
            raw, wcs = _squeeze_degenerate_axes(raw, wcs)
        if raw.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D FITS image after squeezing, got shape {raw.shape}.")

        unit = None
        bunit = header.get("BUNIT")
        if bunit:
            try:
                unit = u.Unit(bunit)
            except ValueError:
                warnings.warn(
                    f"Could not parse BUNIT={bunit!r} as an astropy unit; proceeding without a unit.",
                    stacklevel=2,
                )

        return cls(data=raw, wcs=wcs, unit=unit, meta=header)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        unit=None,
        wcs: WCS | None = None,
        meta=None,
    ) -> "Cube":
        """
        Build a `Cube` directly from an in-memory array.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D ``(H, W)`` or 3D ``(n_channels, H, W)`` array. 2D input is
            promoted to a single-channel ``(1, H, W)`` cube, matching `read`.
        unit : astropy.units.Unit or str, optional
            Physical unit of the pixel values. Default is `None`.
        wcs : astropy.wcs.WCS, optional
            World coordinate system, if known. Default is `None`.
        meta : dict, optional
            Arbitrary metadata to attach. Default is `None`.

        Returns
        -------
        Cube
            A cube wrapping ``data`` with the given ``unit``, ``wcs``, and
            ``meta``.

        Raises
        ------
        ValueError
            If ``data`` is not 2D or 3D after the promotion step.

        Notes
        -----
        Unlike `read`, there is no FITS file to derive ``unit``, ``wcs``, or
        ``meta`` from: callers supply whatever they have, which is often
        nothing at all for a plain PNG/TIFF/``.npy`` image (see
        `turbulens.io.image_loaders.load_input`).
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D array, got shape {data.shape}.")
        return cls(data=data, wcs=wcs, unit=unit, meta=meta)

    @property
    def n_channels(self) -> int:
        """int: Number of channels, i.e. ``data.shape[0]``."""
        return self.data.shape[0]

    @property
    def has_spectral_axis(self) -> bool:
        """bool: Whether the WCS has exactly one recognized spectral axis (velocity or frequency)."""
        if self.wcs is None:
            return False
        return _find_spectral_axis(self.wcs) is not None

    def velocity_axis(self, rest_frequency: u.Quantity | None = None) -> SpectralCoord:
        """
        Compute one velocity value per channel.

        Parameters
        ----------
        rest_frequency : astropy.units.Quantity, optional
            Rest frequency to use when the spectral axis is in frequency
            rather than velocity units. If not given, falls back to the
            WCS ``RESTFRQ`` header value, and if that is also absent, to
            the HI 21 cm line rest frequency (1420.405751 MHz).

        Returns
        -------
        astropy.coordinates.SpectralCoord
            One velocity value per channel, in km/s.

        Raises
        ------
        ValueError
            If the cube has no WCS, or no recognizable spectral WCS axis.

        Warns
        -----
        UserWarning
            Whenever a frequency axis is converted to velocity, naming which
            rest frequency source was used (explicit argument, WCS
            ``RESTFRQ``, or the HI 21 cm default), since this conversion is
            physically meaningful and worth surfacing to the caller.

        Notes
        -----
        The spectral axis's world values come from `astropy.wcs.WCS.pixel_to_world_values`
        rather than a hand-rolled ``crval + (pixel - crpix) * cdelt`` formula, since the
        latter silently ignores a ``CDi_j`` matrix (common in survey FITS): astropy
        leaves ``wcs.wcs.cdelt`` at its default ``1.0`` per axis when a CD matrix is
        present, so reading ``cdelt`` directly would return a spacing off by whatever
        factor the CD matrix actually encodes, with no error.
        ``cunit`` (astropy-normalized) is still read from `wcs.wcs.cunit` directly rather
        than the raw FITS header string, since ``'M/S'`` is not the same string as
        ``'m/s'`` to `astropy.units.Unit`. Other axes are evaluated at their own
        (0-indexed) reference pixel, so this is exact even when WCS axes are not purely
        diagonal (e.g. a nonzero off-diagonal `PC`/`CD` term coupling the spectral axis
        to a spatial one), a case a per-axis-only formula could not handle at all. This
        still does not require wcslib to have recognized a non-standard CTYPE: real
        radio data commonly uses the AIPS-legacy 9-character form (e.g.
        ``CTYPE='VELO-LSRK'``), which is not WCS Paper III conformant, so the spectral
        axis is located via `_find_spectral_axis`'s own CTYPE-prefix matching rather
        than relying on `wcs.wcs_pix2world`'s built-in axis-type dispatch.

        References
        ----------
        `astropy.units.doppler_radio`
            https://docs.astropy.org/en/stable/api/astropy.units.doppler_radio.html
        """
        if self.wcs is None:
            raise ValueError("Cube has no WCS; cannot compute a velocity axis.")
        found = _find_spectral_axis(self.wcs)
        if found is None:
            raise ValueError("Cube has no recognizable spectral WCS axis.")
        index, ctype = found

        cunit = self.wcs.wcs.cunit[index]
        crpix0 = self.wcs.wcs.crpix - 1.0  # 0-indexed reference pixel, per axis
        pixel_args = [np.full(self.n_channels, crpix0[axis]) for axis in range(self.wcs.naxis)]
        pixel_args[index] = np.arange(self.n_channels, dtype=np.float64)
        native_values = self.wcs.pixel_to_world_values(*pixel_args)[index] * cunit

        prefix = ctype[:4].upper()
        if prefix in _VELOCITY_CTYPE_PREFIXES:
            return SpectralCoord(native_values.to(u.km / u.s))

        if rest_frequency is not None:
            rest = rest_frequency
            source = "the explicit rest_frequency argument"
        elif self.wcs.wcs.restfrq:
            rest = self.wcs.wcs.restfrq * u.Hz
            source = "the WCS RESTFRQ header value"
        else:
            rest = _HI_21CM_REST_FREQUENCY
            source = "the HI 21cm line default"
        warnings.warn(
            f"velocity_axis: using rest frequency {rest} from {source} to convert the "
            f"frequency axis to velocity.",
            stacklevel=2,
        )
        velocities = native_values.to(u.km / u.s, equivalencies=u.doppler_radio(rest))
        return SpectralCoord(velocities, doppler_rest=rest, doppler_convention="radio")

    def slice_channel(self, index: int) -> "Cube":
        """
        Return a new 2D `Cube` holding a single channel.

        Parameters
        ----------
        index : int
            0-indexed channel to select.

        Returns
        -------
        Cube
            A cube of shape ``(1, H, W)`` holding channel ``index``, with
            the channel WCS axis dropped.

        Raises
        ------
        IndexError
            If ``index`` is out of range for `n_channels`.
        """
        if not 0 <= index < self.n_channels:
            raise IndexError(f"Channel {index} out of range for cube with {self.n_channels} channels.")
        wcs = _drop_channel_axis(self.wcs, self.data.ndim)
        meta = self.meta.copy() if self.meta is not None else None
        return Cube(data=self.data[index], wcs=wcs, unit=self.unit, meta=meta)

    def project(self, low: int, high: int, mode: str = "sum") -> "Cube":
        """
        Collapse channels ``[low, high)`` into a 2D `Cube`.

        Parameters
        ----------
        low : int
            0-indexed first channel of the range (inclusive).
        high : int
            0-indexed last channel of the range (exclusive).
        mode : {"sum", "mean"}, optional
            Reduction to apply across the selected channels. Default is
            ``"sum"``.

        Returns
        -------
        Cube
            A cube of shape ``(1, H, W)`` holding the reduced channels, with
            the channel WCS axis dropped.

        Raises
        ------
        ValueError
            If ``low``, ``high`` do not satisfy ``0 <= low < high <=
            n_channels``, or if ``mode`` is not ``"sum"`` or ``"mean"``.

        Notes
        -----
        ``mode="sum"``/``"mean"`` are plain pixel-count operations over the
        selected channels, not a unit-correct integrated-intensity
        (moment-0) map: a real integrated intensity in K km/s would require
        multiplying by the channel width, which this method does not do.
        """
        if not (0 <= low < high <= self.n_channels):
            raise ValueError(f"Invalid channel range [{low}, {high}) for cube with {self.n_channels} channels.")
        selected = self.data[low:high]
        if mode == "sum":
            projected = selected.sum(axis=0)
        elif mode == "mean":
            projected = selected.mean(axis=0)
        else:
            raise ValueError(f"Unsupported projection mode: {mode!r}.")
        wcs = _drop_channel_axis(self.wcs, self.data.ndim)
        meta = self.meta.copy() if self.meta is not None else None
        return Cube(data=projected, wcs=wcs, unit=self.unit, meta=meta)

    def project_velocity(
        self, v_low: u.Quantity, v_high: u.Quantity, mode: str = "sum"
    ) -> "Cube":
        """
        Project the channel range whose `velocity_axis` falls within ``[v_low, v_high]``.

        Parameters
        ----------
        v_low : astropy.units.Quantity
            Lower bound of the velocity range (a velocity-convertible
            quantity, e.g. km/s).
        v_high : astropy.units.Quantity
            Upper bound of the velocity range.
        mode : {"sum", "mean"}, optional
            Reduction to apply, passed through to `project`. Default is
            ``"sum"``.

        Returns
        -------
        Cube
            The result of calling `project` on the channel indices whose
            velocity falls in ``[v_low, v_high]``.

        Raises
        ------
        ValueError
            If ``v_low >= v_high``, or if the requested range does not
            overlap the cube's velocity coverage at all.

        Warns
        -----
        UserWarning
            If the requested range only partially overlaps the cube's
            velocity coverage, naming the clipped range actually used.

        Notes
        -----
        Selection uses a boolean mask over `velocity_axis`, rather than
        `numpy.searchsorted`, because the velocity axis may be descending
        (negative ``CDELT``) as well as ascending, and a mask is correct in
        both cases without needing to know the sort direction up front.
        """
        if v_low >= v_high:
            raise ValueError(f"v_low ({v_low}) must be less than v_high ({v_high}).")

        velocities = u.Quantity(self.velocity_axis()).to(u.km / u.s)
        v_low = v_low.to(u.km / u.s)
        v_high = v_high.to(u.km / u.s)
        coverage_low, coverage_high = velocities.min(), velocities.max()

        mask = (velocities >= v_low) & (velocities <= v_high)
        if not np.any(mask):
            raise ValueError(
                f"Requested velocity range [{v_low}, {v_high}] does not overlap the "
                f"cube's coverage [{coverage_low}, {coverage_high}]."
            )

        indices = np.nonzero(mask)[0]
        low_index, high_index = int(indices.min()), int(indices.max()) + 1

        if v_low < coverage_low or v_high > coverage_high:
            warnings.warn(
                f"Requested velocity range [{v_low}, {v_high}] partially overlaps the "
                f"cube's coverage [{coverage_low}, {coverage_high}]; clipped to "
                f"[{velocities[low_index]}, {velocities[high_index - 1]}].",
                stacklevel=2,
            )

        return self.project(low_index, high_index, mode=mode)
