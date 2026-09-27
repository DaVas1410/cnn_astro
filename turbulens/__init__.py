from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("turbulens")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from turbulens.io.cube import Cube
from turbulens.inference import Inferencer

__all__ = ["Cube", "Inferencer", "__version__"]
