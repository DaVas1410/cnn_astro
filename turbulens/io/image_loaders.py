"""Loaders for non-FITS image formats (PNG, TIFF, .npy), plus format dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from turbulens.io.cube import Cube

_IMAGE_SUFFIXES = (".png", ".tif", ".tiff")


def load_png_or_tiff(path: str | Path) -> Cube:
    """Load a PNG or TIFF file as a single-channel `Cube`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.png``, ``.tif``, or ``.tiff`` file.

    Returns
    -------
    Cube
        A cube of shape ``(1, H, W)``, with no unit, WCS, or metadata,
        since plain image formats carry none.

    Raises
    ------
    ImportError
        If Pillow is not installed. Install with ``pip install
        turbulens[image]``.

    Notes
    -----
    Grayscale and single-precision-float modes (``"L"``, ``"F"``,
    ``"I"``, ``"I;16"``) are read directly as float32. Any other mode
    (e.g. RGB, palette) is converted to Pillow's ``"F"`` (32-bit float
    grayscale) mode first, which is a lossy, information-discarding
    conversion for color images: `turbulens` models expect a single
    physical intensity channel, not RGB.

    References
    ----------
    `PIL.Image.Image.convert`
        https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Reading PNG/TIFF images requires Pillow. Install with: pip install turbulens[image]"
        ) from exc

    with Image.open(path) as image:
        if image.mode in ("L", "F", "I", "I;16"):
            array = np.asarray(image, dtype=np.float32)
        else:
            array = np.asarray(image.convert("F"), dtype=np.float32)
    return Cube.from_array(array)


def load_numpy(path: str | Path) -> Cube:
    """Load a ``.npy`` file as a single-channel `Cube`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.npy`` file containing a 2D array.

    Returns
    -------
    Cube
        A cube of shape ``(1, H, W)``.

    Raises
    ------
    ValueError
        If the loaded array is not 2D.

    Notes
    -----
    Loaded with ``allow_pickle=False``, since a `turbulens` input is
    always a plain numeric array; allowing pickled objects would open an
    arbitrary-code-execution path for a malicious ``.npy`` file.
    """
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2D array, got shape {array.shape}.")
    return Cube.from_array(array.astype(np.float32))


def load_input(path: str | Path, hdu: int | str = 0) -> Cube:
    """Load any supported input format into a `Cube`, dispatching on file extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a FITS file (``.fits``, ``.fits.gz``), a PNG/TIFF image,
        or a ``.npy`` array file.
    hdu : int or str, optional
        FITS HDU index or name. Only meaningful for FITS input; passing
        a non-default value for a non-FITS input raises. Default is
        ``0``.

    Returns
    -------
    Cube
        The loaded cube, via `turbulens.io.cube.Cube.read` for FITS
        input, or `load_png_or_tiff` / `load_numpy` for other formats.

    Raises
    ------
    ValueError
        If ``hdu`` is given for non-FITS input, if the extension is
        ``.npz``, or if the extension is not recognized.

    Notes
    -----
    ``.npz`` (multi-array archives) are explicitly rejected rather than
    silently picking one array, since there is no reliable convention
    for which array in an ``.npz`` archive is the intended image;
    callers should extract the single array they want to a ``.npy``
    file first.
    """
    path = Path(path)
    name = path.name.lower()

    if name.endswith(".fits") or name.endswith(".fits.gz"):
        return Cube.read(path, hdu=hdu)

    if hdu != 0:
        raise ValueError(f"{path}: --hdu is only meaningful for FITS input.")

    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return load_png_or_tiff(path)
    if suffix == ".npy":
        return load_numpy(path)
    if suffix == ".npz":
        raise ValueError(f"{path}: .npz multi-array archives are not supported; provide a single .npy array.")
    raise ValueError(f"{path}: unrecognized input format {suffix!r}.")
