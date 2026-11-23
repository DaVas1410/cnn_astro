"""
The CLI imports each heavy library only for the commands that need it.

PyTorch, astropy and scipy take seconds to import, so ``--version``,
``--help`` and usage errors must not load any of them, and ``inspect`` and
the classical methods must not load PyTorch. Each case runs in a fresh
interpreter, since this test process has long since imported everything.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import turbulens

_PROBE = """
import sys
sys.argv[0] = "turbulens"
from turbulens.cli import main
try:
    main(sys.argv[1:])
except SystemExit:
    pass
print("LOADED:" + ",".join(name for name in ("torch", "astropy", "scipy") if name in sys.modules), file=sys.__stderr__)
"""


def _env() -> dict[str, str]:
    """The environment, with the `turbulens` under test first on the path whatever the working directory."""
    root = str(Path(turbulens.__file__).resolve().parents[1])
    return dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))


def _loaded(*args, cwd=None) -> set[str]:
    done = subprocess.run([sys.executable, "-c", _PROBE, *args], capture_output=True, text=True, cwd=cwd,
                          encoding="utf-8", env=_env(), check=False)
    line = next(line for line in done.stderr.splitlines() if line.startswith("LOADED:"))
    return set(filter(None, line[len("LOADED:"):].split(",")))


@pytest.mark.parametrize("args", [["--version"], ["--help"], ["infer", "--help"], ["inspect", "--help"],
                                  ["download", "--help"], ["infer"], ["infer", "--input", "x.npy"]])
def test_quick_commands_load_no_heavy_library(args):
    assert _loaded(*args) == set()


@pytest.mark.parametrize("args", [["inspect", "--input", "image.npy"],
                                  ["infer", "--input", "image.npy", "--method", "fft-band"]])
def test_commands_without_a_model_do_not_load_torch(tmp_path, args):
    np.save(tmp_path / "image.npy", np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32))

    loaded = _loaded(*args, cwd=tmp_path)

    assert "torch" not in loaded
    assert "astropy" in loaded  # the probe really ran the command


def test_package_attributes_are_imported_on_first_use():
    code = ("import sys, turbulens; assert 'torch' not in sys.modules; "
            "from turbulens import Cube, Inferencer; assert Inferencer.__name__ == 'Inferencer'; "
            "assert Cube.__name__ == 'Cube'; assert 'torch' in sys.modules")
    subprocess.run([sys.executable, "-c", code], env=_env(), check=True)
    with pytest.raises(AttributeError):
        turbulens.not_an_attribute  # noqa: B018
