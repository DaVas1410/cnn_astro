import numpy as np
import pytest
from PIL import Image

from turbulens.io.image_loaders import load_input, load_numpy, load_png_or_tiff


def test_load_png_single_band_preserves_values(tmp_path):
    path = tmp_path / "image.png"
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    Image.fromarray(data, mode="L").save(path)

    cube = load_png_or_tiff(path)

    assert cube.data.shape == (1, 4, 4)
    np.testing.assert_array_equal(cube.data[0], data.astype(np.float32))
    assert cube.unit is None
    assert cube.wcs is None


def test_load_tiff_32bit_float_preserves_dynamic_range(tmp_path):
    path = tmp_path / "image.tiff"
    data = np.linspace(0, 70000, 16, dtype=np.float32).reshape(4, 4)
    Image.fromarray(data, mode="F").save(path)

    cube = load_png_or_tiff(path)

    np.testing.assert_allclose(cube.data[0], data, rtol=1e-4)


def test_load_rgb_png_converts_without_8bit_l_mode(tmp_path):
    path = tmp_path / "rgb.png"
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    data[..., 0] = 100
    Image.fromarray(data, mode="RGB").save(path)

    cube = load_png_or_tiff(path)

    assert cube.data.shape == (1, 4, 4)


def test_load_numpy_round_trips(tmp_path):
    path = tmp_path / "array.npy"
    data = np.arange(9, dtype=np.float64).reshape(3, 3)
    np.save(path, data)

    cube = load_numpy(path)

    np.testing.assert_array_equal(cube.data[0], data.astype(np.float32))


def test_load_numpy_rejects_non_2d_array(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((2, 3, 3)))

    with pytest.raises(ValueError):
        load_numpy(path)


def test_load_input_rejects_npz(tmp_path):
    path = tmp_path / "archive.npz"
    np.savez(path, a=np.zeros((3, 3)))

    with pytest.raises(ValueError):
        load_input(path)


def test_load_input_rejects_unrecognized_extension(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("not an image")

    with pytest.raises(ValueError):
        load_input(path)


def test_load_input_rejects_hdu_for_non_fits(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(path)

    with pytest.raises(ValueError):
        load_input(path, hdu=1)


def test_load_input_dispatches_fits(tmp_path):
    from astropy.io import fits

    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(data=np.zeros((2, 4, 4), dtype=np.float32)).writeto(path)

    cube = load_input(path)

    assert cube.data.shape == (2, 4, 4)
