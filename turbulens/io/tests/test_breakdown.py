import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from turbulens.io.breakdown import CubeBreakdown, describe_cube
from turbulens.io.cube import Cube


def _spectral_header(ctype3, cunit3, crval3, cdelt3, naxis3):
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 3
    header["NAXIS2"] = 3
    header["NAXIS3"] = naxis3
    header["CTYPE1"] = "RA---SIN"
    header["CRVAL1"], header["CRPIX1"], header["CDELT1"], header["CUNIT1"] = 0.0, 1.0, -1.0, "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL2"], header["CRPIX2"], header["CDELT2"], header["CUNIT2"] = 0.0, 1.0, 1.0, "deg"
    header["CTYPE3"] = ctype3
    header["CRVAL3"], header["CRPIX3"], header["CDELT3"], header["CUNIT3"] = crval3, 1.0, cdelt3, cunit3
    return header


def test_describe_cube_with_spectral_axis():
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, 500.0, naxis3=5)
    data = np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3)
    cube = Cube(data=data, wcs=WCS(header), unit=u.K, meta=header)

    breakdown = describe_cube(cube, source_path="fake.fits")

    assert isinstance(breakdown, CubeBreakdown)
    assert breakdown.shape == (5, 3, 3)
    assert breakdown.n_channels == 5
    assert breakdown.has_spectral_axis is True
    assert breakdown.is_projected_image is False
    assert breakdown.unit == "K"
    # In this environment's astropy/wcslib version, WCS(header) applies its own
    # "spcfix" normalization during construction, rewriting the AIPS-legacy
    # 8-char CTYPE3='VELO-LSR' to the WCS-Paper-III-conformant 'VOPT' (with
    # SPECSYS='LSRK') before describe_cube ever sees the WCS -- so wcs.wcs.ctype
    # already reads 'VOPT' here, not the header's original 'VELO-LSR' string.
    assert breakdown.spectral_ctype == "VOPT"
    lo, hi = breakdown.velocity_range
    assert lo == pytest.approx(-1.0, abs=1e-6)
    assert hi == pytest.approx(1.0, abs=1e-6)
    assert breakdown.source_path == "fake.fits"


def test_describe_cube_without_spectral_axis_is_projected_image():
    cube = Cube.from_array(np.zeros((4, 4), dtype=np.float32))

    breakdown = describe_cube(cube, source_path="image.png")

    assert breakdown.has_spectral_axis is False
    assert breakdown.is_projected_image is True
    assert breakdown.velocity_range is None
    assert breakdown.n_channels == 1
    # All-zero image has no power at any wavenumber.
    assert breakdown.spectral_estimate_band_edge == {"k_min": None, "k_max": None}
    assert breakdown.spectral_estimate_power_law == {
        "alpha": None, "r_squared": None, "k_min": None, "k_max": None,
    }


def test_describe_cube_spectral_estimate_band_edge_recovers_band_limited_image():
    n = 64
    kax = np.fft.fftshift(np.fft.fftfreq(n)) * n
    kxg, kyg = np.meshgrid(kax, kax)
    kr = np.sqrt(kxg**2 + kyg**2)
    kmin, kmax = 5, 20
    amp = np.where((kr >= kmin) & (kr <= kmax), 1.0, 0.0)
    rng = np.random.default_rng(0)
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    image = np.fft.ifft2(np.fft.ifftshift(amp * phase)).real.astype(np.float32)
    cube = Cube.from_array(image)

    breakdown = describe_cube(cube, source_path="band_limited.npy")

    assert abs(breakdown.spectral_estimate_band_edge["k_min"] - kmin) <= 3
    assert abs(breakdown.spectral_estimate_band_edge["k_max"] - kmax) <= 3


def test_describe_cube_spectral_estimate_power_law_recovers_alpha():
    n = 256
    kax = np.fft.fftshift(np.fft.fftfreq(n)) * n
    kxg, kyg = np.meshgrid(kax, kax)
    kr = np.sqrt(kxg**2 + kyg**2)
    amp = np.clip(kr, 1.0, None) ** (-3.67 / 2.0)
    rng = np.random.default_rng(0)
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    image = np.fft.ifft2(np.fft.ifftshift(amp * phase)).real.astype(np.float32)
    cube = Cube.from_array(image)

    breakdown = describe_cube(cube, source_path="power_law.npy")

    assert breakdown.spectral_estimate_power_law["alpha"] == pytest.approx(-3.67, abs=0.3)


def test_describe_cube_spectral_estimates_are_none_for_spectral_cube():
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, 500.0, naxis3=5)
    data = np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3)
    cube = Cube(data=data, wcs=WCS(header), unit=u.K, meta=header)

    breakdown = describe_cube(cube, source_path="fake.fits")

    assert breakdown.spectral_estimate_band_edge is None
    assert breakdown.spectral_estimate_power_law is None


def test_describe_cube_spectral_estimates_are_none_for_multichannel_cube_without_wcs():
    # A real telescope (x, y, velocity) cube with no recognized spectral axis
    # (e.g. missing/malformed WCS) must not have its spectral estimates silently
    # computed from a single arbitrary channel; it needs an explicit projection
    # (as `turbulens infer --integrate`/`--velocity` does) first.
    data = np.random.default_rng(0).random((5, 32, 32)).astype(np.float32)
    cube = Cube(data=data, wcs=None, unit=None, meta={})

    breakdown = describe_cube(cube, source_path="raw_multichannel.fits")

    assert breakdown.n_channels == 5
    assert breakdown.spectral_estimate_band_edge is None
    assert breakdown.spectral_estimate_power_law is None


def test_describe_cube_single_channel_with_real_spectral_axis_is_not_projected_image():
    header = _spectral_header("VELO-LSR", "m/s", 100.0, 500.0, naxis3=1)
    data = np.zeros((1, 3, 3), dtype=np.float32)
    cube = Cube(data=data, wcs=WCS(header), unit=None, meta=header)

    breakdown = describe_cube(cube, source_path="single_channel.fits")

    assert breakdown.n_channels == 1
    assert breakdown.has_spectral_axis is True
    assert breakdown.is_projected_image is False
    lo, hi = breakdown.velocity_range
    assert lo == pytest.approx(hi, abs=1e-9)


def test_describe_cube_pixel_stats_ignore_nan():
    cube = Cube.from_array(np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32))

    breakdown = describe_cube(cube, source_path="stats.npy")

    stats = breakdown.pixel_stats
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)
    assert stats["mean"] == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)
    assert stats["nan_fraction"] == pytest.approx(0.25)


def test_describe_cube_frequency_axis_records_rest_frequency():
    header = _spectral_header("FREQ", "Hz", 1.420405751e9, -1e5, naxis3=3)
    data = np.zeros((3, 3, 3), dtype=np.float32)
    cube = Cube(data=data, wcs=WCS(header), unit=None, meta=header)

    with pytest.warns(UserWarning):
        breakdown = describe_cube(cube, source_path="freq.fits")

    assert breakdown.rest_frequency_ghz == pytest.approx(1.420405751, abs=1e-6)
