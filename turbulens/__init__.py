from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("turbulens")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["Cube", "Inferencer", "__version__"]


def __getattr__(name: str):
    """Import `Cube` and `Inferencer` on first use, so ``import turbulens`` does not load PyTorch."""
    if name == "Inferencer":
        from turbulens.inference import Inferencer

        return Inferencer
    if name == "Cube":
        from turbulens.io.cube import Cube

        return Cube
    raise AttributeError(f"module 'turbulens' has no attribute {name!r}")
