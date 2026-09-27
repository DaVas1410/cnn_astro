from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from turbulens.io.cube import Cube

_GASS_FIXTURE = Path(__file__).parent / "data" / "gass_0_0_1774393774.fits.gz"


def _write_fits(path, data, header=None):
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path)


def test_read_3d_cube_keeps_channel_axis_and_parses_unit(tmp_path):
    path = tmp_path / "cube.fits"
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    header = fits.Header({"BUNIT": "K"})
    _write_fits(path, data, header)

    cube = Cube.read(path)

    assert cube.data.shape == (2, 4, 4)
    assert cube.n_channels == 2
    np.testing.assert_array_equal(cube.data, data)
    assert str(cube.unit) == "K"
    assert cube.wcs is not None


def test_read_2d_image_gets_promoted_to_single_channel(tmp_path):
    path = tmp_path / "image.fits"
    data = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    _write_fits(path, data)

    cube = Cube.read(path)

    assert cube.data.shape == (1, 4, 4)
    assert cube.n_channels == 1


def test_read_squeezes_degenerate_fourth_axis(tmp_path):
    path = tmp_path / "stokes_cube.fits"
    data = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    _write_fits(path, data)

    cube = Cube.read(path)

    assert cube.data.shape == (2, 4, 4)
    np.testing.assert_array_equal(cube.data, data[0])


def test_read_rejects_fourth_axis_with_no_degenerate_dimension(tmp_path):
    path = tmp_path / "bad.fits"
    data = np.zeros((2, 2, 4, 4), dtype=np.float32)
    _write_fits(path, data)

    with pytest.raises(ValueError):
        Cube.read(path)


def test_read_warns_on_unparseable_bunit(tmp_path):
    path = tmp_path / "bad_unit.fits"
    data = np.zeros((2, 4, 4), dtype=np.float32)
    header = fits.Header({"BUNIT": "not a unit!!"})
    _write_fits(path, data, header)

    with pytest.warns(UserWarning):
        cube = Cube.read(path)
    assert cube.unit is None


def _cube_with_data(data):
    return Cube(data=data.astype(np.float32), wcs=None, unit=None, meta=None)


def test_slice_channel_returns_single_channel_cube():
    data = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 2.0), np.full((3, 3), 3.0)])
    cube = _cube_with_data(data)

    result = cube.slice_channel(1)

    assert result.data.shape == (1, 3, 3)
    assert result.n_channels == 1
    np.testing.assert_array_equal(result.data, np.full((1, 3, 3), 2.0))


def test_slice_channel_rejects_out_of_range_index():
    cube = _cube_with_data(np.zeros((2, 3, 3)))
    with pytest.raises(IndexError):
        cube.slice_channel(5)


def test_project_sum_adds_selected_channels():
    data = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0), np.full((2, 2), 3.0)])
    cube = _cube_with_data(data)

    result = cube.project(0, 2, mode="sum")

    assert result.data.shape == (1, 2, 2)
    assert result.n_channels == 1
    np.testing.assert_array_equal(result.data, np.full((1, 2, 2), 3.0))


def test_project_mean_averages_selected_channels():
    data = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 3.0)])
    cube = _cube_with_data(data)

    result = cube.project(0, 2, mode="mean")

    assert result.data.shape == (1, 2, 2)
    np.testing.assert_array_equal(result.data, np.full((1, 2, 2), 2.0))


def test_project_rejects_invalid_range():
    cube = _cube_with_data(np.zeros((3, 2, 2)))
    with pytest.raises(ValueError):
        cube.project(2, 1)


def test_project_rejects_unsupported_mode():
    cube = _cube_with_data(np.zeros((3, 2, 2)))
    with pytest.raises(ValueError):
        cube.project(0, 2, mode="bogus")


def test_slice_channel_meta_is_not_aliased_to_parent(tmp_path):
    path = tmp_path / "cube.fits"
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    _write_fits(path, data)
    cube = Cube.read(path)

    result = cube.slice_channel(0)
    result.meta["MYKEY"] = "changed"

    assert "MYKEY" not in cube.meta


def test_project_meta_is_not_aliased_to_parent(tmp_path):
    path = tmp_path / "cube.fits"
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    _write_fits(path, data)
    cube = Cube.read(path)

    result = cube.project(0, 2)
    result.meta["MYKEY"] = "changed"

    assert "MYKEY" not in cube.meta


def test_slice_channel_drops_wcs_axis(tmp_path):
    path = tmp_path / "cube.fits"
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    header = fits.Header(
        {
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRVAL3": 1.0,
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRPIX3": 1.0,
            "CDELT1": -1.0,
            "CDELT2": 1.0,
            "CDELT3": 1.0,
        }
    )
    _write_fits(path, data, header)
    cube = Cube.read(path)

    result = cube.slice_channel(0)

    assert result.wcs.naxis == cube.wcs.naxis - 1
    assert list(result.wcs.wcs.ctype) == ["RA---SIN", "DEC--SIN"]


def test_slice_channel_on_2d_promoted_cube_leaves_wcs_untouched(tmp_path):
    path = tmp_path / "image.fits"
    data = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    header = fits.Header(
        {
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CDELT1": -1.0,
            "CDELT2": 1.0,
        }
    )
    _write_fits(path, data, header)
    cube = Cube.read(path)
    assert cube.data.shape == (1, 4, 4)

    result = cube.slice_channel(0)

    assert result.wcs.naxis == 2
    assert list(result.wcs.wcs.ctype) == ["RA---SIN", "DEC--SIN"]


def test_from_array_promotes_2d_input():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)

    cube = Cube.from_array(data)

    assert cube.data.shape == (1, 4, 4)
    assert cube.n_channels == 1
    np.testing.assert_array_equal(cube.data[0], data)


def test_from_array_passes_through_3d_input():
    data = np.zeros((3, 4, 4), dtype=np.float32)

    cube = Cube.from_array(data)

    assert cube.data.shape == (3, 4, 4)


def test_from_array_carries_unit_wcs_meta():
    data = np.zeros((4, 4), dtype=np.float32)

    cube = Cube.from_array(data, unit=u.K, wcs=None, meta={"SOURCE": "test"})

    assert cube.unit == u.K
    assert cube.meta == {"SOURCE": "test"}


def test_from_array_rejects_bad_ndim():
    with pytest.raises(ValueError):
        Cube.from_array(np.zeros((2, 2, 2, 2), dtype=np.float32))


def _spectral_header(ctype3, cunit3, crval3, cdelt3, crpix3=1.0, naxis3=5, restfrq=None):
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["NAXIS3"] = naxis3
    header["CTYPE1"] = "RA---SIN"
    header["CRVAL1"] = 0.0
    header["CRPIX1"] = 1.0
    header["CDELT1"] = -1.0
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL2"] = 0.0
    header["CRPIX2"] = 1.0
    header["CDELT2"] = 1.0
    header["CUNIT2"] = "deg"
    header["CTYPE3"] = ctype3
    header["CRVAL3"] = crval3
    header["CRPIX3"] = crpix3
    header["CDELT3"] = cdelt3
    header["CUNIT3"] = cunit3
    if restfrq is not None:
        header["RESTFRQ"] = restfrq
    return header


def _cube_with_header(header, naxis3):
    data = np.zeros((naxis3, 4, 4), dtype=np.float32)
    return Cube(data=data, wcs=WCS(header), unit=None, meta=header)


def test_has_spectral_axis_true_for_velocity_ctype():
    header = _spectral_header("VELO-LSR", "m/s", -494700.0, 824.5, naxis3=5)
    cube = _cube_with_header(header, 5)

    assert cube.has_spectral_axis is True


def test_has_spectral_axis_false_without_spectral_ctype():
    header = _spectral_header("LINEAR", "", 0.0, 1.0, naxis3=5)
    cube = _cube_with_header(header, 5)

    assert cube.has_spectral_axis is False


def test_has_spectral_axis_raises_on_multiple_spectral_axes():
    # The second spectral axis is appended as a new trailing WCS axis rather
    # than overwriting CTYPE1: overwriting CTYPE1 (RA) would leave CTYPE2
    # (DEC) as an unpaired celestial axis, which this environment's wcslib
    # rejects outright at WCS-construction time ("Unmatched celestial axes"),
    # before turbulens' own multiple-spectral-axis check ever runs.
    header = _spectral_header("VELO-LSR", "m/s", -494700.0, 824.5, naxis3=5)
    header["NAXIS"] = 4
    header["NAXIS4"] = 3
    header["CTYPE4"] = "FREQ"
    header["CRVAL4"] = 1.4e9
    header["CRPIX4"] = 1.0
    header["CDELT4"] = 1e6
    header["CUNIT4"] = "Hz"
    data = np.zeros((3, 5, 4, 4), dtype=np.float32)
    cube = Cube(data=data, wcs=WCS(header), unit=None, meta=header)

    with pytest.raises(ValueError):
        cube.has_spectral_axis


def test_has_spectral_axis_raises_when_not_trailing_axis():
    # Both celestial axes (RA and DEC) are kept, with FREQ placed first, so
    # the WCS remains constructible (an unpaired celestial axis is rejected
    # by this environment's wcslib at construction time) while still
    # exercising a spectral axis that is not the trailing WCS axis.
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 5
    header["NAXIS2"] = 4
    header["NAXIS3"] = 4
    header["CTYPE1"] = "FREQ"
    header["CRVAL1"] = 1.4e9
    header["CRPIX1"] = 1.0
    header["CDELT1"] = 1e6
    header["CUNIT1"] = "Hz"
    header["CTYPE2"] = "RA---SIN"
    header["CRVAL2"] = 0.0
    header["CRPIX2"] = 1.0
    header["CDELT2"] = 1.0
    header["CUNIT2"] = "deg"
    header["CTYPE3"] = "DEC--SIN"
    header["CRVAL3"] = 0.0
    header["CRPIX3"] = 1.0
    header["CDELT3"] = 1.0
    header["CUNIT3"] = "deg"
    data = np.zeros((4, 4, 5), dtype=np.float32)
    cube = Cube(data=data, wcs=WCS(header), unit=None, meta=header)

    with pytest.raises(ValueError):
        cube.has_spectral_axis


def test_velocity_axis_for_velocity_typed_ctype():
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, 500.0, naxis3=5)
    cube = _cube_with_header(header, 5)

    velocities = cube.velocity_axis()

    expected_m_s = -1000.0 + np.arange(5) * 500.0
    np.testing.assert_allclose(velocities.to_value(u.m / u.s), expected_m_s, rtol=1e-6)


def test_velocity_axis_honours_cd_matrix_not_just_cdelt():
    # Some survey FITS headers encode the pixel-to-world scale via a CDi_j
    # matrix instead of CDELTi; a naive `crval + (pixel - crpix) * cdelt`
    # read ignores the CD matrix entirely (astropy leaves wcs.wcs.cdelt at
    # its default 1.0 per axis when CD is present), silently returning a
    # spacing wrong by whatever factor CD actually encodes.
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, cdelt3=1.0, naxis3=5)
    del header["CDELT3"]
    header["CD3_3"] = 500.0
    cube = _cube_with_header(header, 5)

    velocities = cube.velocity_axis()

    expected_m_s = -1000.0 + np.arange(5) * 500.0
    np.testing.assert_allclose(velocities.to_value(u.m / u.s), expected_m_s, rtol=1e-6)


def test_velocity_axis_for_frequency_typed_ctype_uses_header_restfrq():
    rest_hz = 1.42e9
    freq0 = rest_hz  # cube centered on the rest frequency at channel 0
    header = _spectral_header("FREQ", "Hz", freq0, -1e5, naxis3=5, restfrq=rest_hz)
    cube = _cube_with_header(header, 5)

    with pytest.warns(UserWarning, match="RESTFRQ"):
        velocities = cube.velocity_axis()

    assert velocities.to_value(u.km / u.s)[0] == pytest.approx(0.0, abs=1e-3)


def test_velocity_axis_frequency_without_restfrq_uses_hi_default():
    header = _spectral_header("FREQ", "Hz", 1.420405751e9, -1e5, naxis3=3)
    cube = _cube_with_header(header, 3)

    with pytest.warns(UserWarning, match="21cm"):
        cube.velocity_axis()


def test_velocity_axis_raises_without_spectral_axis():
    header = _spectral_header("LINEAR", "", 0.0, 1.0, naxis3=3)
    cube = _cube_with_header(header, 3)

    with pytest.raises(ValueError):
        cube.velocity_axis()


def test_velocity_axis_against_real_gass_file():
    cube = Cube.read(str(_GASS_FIXTURE))

    velocities = cube.velocity_axis().to_value(u.km / u.s)

    assert velocities[0] == pytest.approx(-494.7, abs=0.1)
    assert velocities[-1] == pytest.approx(494.7, abs=0.1)


def test_project_velocity_selects_matching_channels():
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, 500.0, naxis3=5)
    # channel velocities (m/s): -1000, -500, 0, 500, 1000
    data = np.stack([np.full((2, 2), float(i)) for i in range(5)]).astype(np.float32)
    cube = Cube(data=data, wcs=WCS(header), unit=None, meta=header)

    result = cube.project_velocity(-0.5 * u.km / u.s, 0.5 * u.km / u.s, mode="sum")

    # -500, 0, 500 m/s == -0.5, 0.0, 0.5 km/s -> channels 1,2,3 -> values 1+2+3=6
    np.testing.assert_array_equal(result.data, np.full((1, 2, 2), 6.0))


def test_project_velocity_handles_descending_axis():
    header = _spectral_header("VELO-LSR", "m/s", 1000.0, -500.0, naxis3=5)
    # channel velocities (m/s): 1000, 500, 0, -500, -1000 (descending)
    data = np.stack([np.full((2, 2), float(i)) for i in range(5)]).astype(np.float32)
    cube = Cube(data=data, wcs=WCS(header), unit=None, meta=header)

    result = cube.project_velocity(-0.5 * u.km / u.s, 0.5 * u.km / u.s, mode="sum")

    # -500, 0, 500 m/s are channels 3,2,1 -> values 1+2+3=6
    np.testing.assert_array_equal(result.data, np.full((1, 2, 2), 6.0))


def test_project_velocity_raises_when_range_does_not_overlap():
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, 500.0, naxis3=5)
    cube = _cube_with_header(header, 5)

    with pytest.raises(ValueError):
        cube.project_velocity(100 * u.km / u.s, 200 * u.km / u.s)


def test_project_velocity_warns_and_clips_on_partial_overlap():
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, 500.0, naxis3=5)
    # channel velocities (m/s): -1000, -500, 0, 500, 1000
    data = np.stack([np.full((2, 2), float(i)) for i in range(5)]).astype(np.float32)
    cube = Cube(data=data, wcs=WCS(header), unit=None, meta=header)

    with pytest.warns(UserWarning, match="partially overlaps"):
        result = cube.project_velocity(-5 * u.km / u.s, 0.5 * u.km / u.s, mode="sum")

    # requested goes far below coverage; clips to the full available range up to 0.5 km/s:
    # channels 0,1,2,3 (velocities -1000..500 m/s) -> 0+1+2+3=6
    np.testing.assert_array_equal(result.data, np.full((1, 2, 2), 6.0))


def test_project_velocity_rejects_inverted_bounds():
    header = _spectral_header("VELO-LSR", "m/s", -1000.0, 500.0, naxis3=5)
    cube = _cube_with_header(header, 5)

    with pytest.raises(ValueError):
        cube.project_velocity(1 * u.km / u.s, -1 * u.km / u.s)
