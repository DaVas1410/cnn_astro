"""
Snapshot tests of the CLI's console output.

Each case renders one output with a recording console at a fixed width
(``Console(record=True, width=100, force_terminal=True)``) and compares
the text with ``snapshots/<case>.txt``. After an intended change to the
output, regenerate the files with::

    TURBULENS_UPDATE_SNAPSHOTS=1 pytest turbulens/io/tests/test_snapshots.py

and review the diff.
"""

import io
import os
import warnings
from pathlib import Path

import pytest
from astropy.table import MaskedColumn, QTable
from astropy.wcs import FITSFixedWarning
from rich.console import Console

from turbulens._style import THEME
from turbulens.io.breakdown import CubeBreakdown
from turbulens.io.display import key_value_table, message_panels, results_table
from turbulens.io.report import breakdown_renderables

SNAPSHOTS = Path(__file__).parent / "snapshots"


def _predictions(rows, clipped=None):
    table = QTable()
    table["image_index"] = [0] * len(rows)
    table["target"] = [target for target, _, _ in rows]
    table["value"] = [value for _, value, _ in rows]
    stds = [std for _, _, std in rows]
    table["epistemic_std"] = MaskedColumn([0.0 if s is None else s for s in stds], mask=[s is None for s in stds])
    if clipped is not None:
        table.meta["clipped_pixel_fraction_max"] = clipped
    return table


def _cnn(clipped):
    return _predictions([("k_min", 1.3792, 0.5257), ("k_max", 63.9098, 0.1580),
                         ("sigma", 4.1419, 0.1171), ("beta", -2.5287, 0.1232)], clipped=clipped)


def _panels():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(".csv does not preserve table metadata; use .ecsv to keep run provenance.", stacklevel=1)
        warnings.warn("'unitfix' made the change 'DEG' -> 'deg'", FITSFixedWarning, stacklevel=1)
        warnings.warn("'spcfix' made the change 'VELO-LSR' -> 'VOPT'", FITSFixedWarning, stacklevel=1)
    return message_panels(caught)


def _inspect_image():
    return breakdown_renderables(CubeBreakdown(
        shape=(1, 128, 128), n_channels=1, has_spectral_axis=False, is_projected_image=True, unit=None,
        wcs_summary={}, velocity_range=None, rest_frequency_ghz=None, spectral_ctype=None,
        pixel_stats={"min": -1.658, "max": 1.354, "mean": -0.1891, "std": 0.406, "nan_fraction": 0.0},
        spectral_estimate_band_edge={"k_min": 7.0, "k_max": 57.0},
        spectral_estimate_power_law={"alpha": 7.409, "r_squared": 0.2619, "k_min": 24.0, "k_max": 28.0},
        source_path="synthetic.npy",
    ))


CASES = {
    "run_table": lambda: [key_value_table("Run", [
        ("Input", "examples/gass_0_0_1774393774.fits.gz"), ("Selection", "channels 0 to 1201, integrated"),
        ("Method", "cnn"), ("Model", "multitask · local_v2"),
        ("Hub revision", "cPhysPlus/turbulens-ensembles@a784ee297427"), ("Members", 5),
        ("Normalization", "real observation"), ("Clipped pixels", "0.1%"),
    ])],
    "predictions_cnn": lambda: [results_table(_cnn(clipped=0.001))],
    "predictions_out_of_range": lambda: [results_table(_cnn(clipped=1.0), suggest_real_data=True)],
    "predictions_powerlaw": lambda: [results_table(_predictions(
        [("k_min", 2.0, None), ("k_max", 61.0, None), ("alpha", -4.0234, None), ("r_squared", 0.9959, None)]))],
    "predictions_ascii": lambda: [results_table(_cnn(clipped=0.001), unicode=False)],
    "panels": _panels,
    "inspect_image": _inspect_image,
}


def _render(renderables) -> str:
    # legacy_windows=False: on a Windows console Rich would swap rounded boxes for square ones,
    # so the same snapshot holds on every platform.
    console = Console(record=True, width=100, force_terminal=True, legacy_windows=False, theme=THEME,
                      file=io.StringIO())
    console.print(*renderables)
    return console.export_text()


@pytest.mark.parametrize("case", sorted(CASES))
def test_console_output_matches_snapshot(case):
    text = _render(CASES[case]())
    path = SNAPSHOTS / f"{case}.txt"
    if os.environ.get("TURBULENS_UPDATE_SNAPSHOTS"):
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert path.exists(), f"missing snapshot {path.name}; run with TURBULENS_UPDATE_SNAPSHOTS=1"
    assert text == path.read_text(encoding="utf-8")


def test_console_output_carries_theme_styles():
    console = Console(record=True, width=100, force_terminal=True, color_system="truecolor", legacy_windows=False,
                      theme=THEME, file=io.StringIO())
    console.print(results_table(_cnn(clipped=1.0), suggest_real_data=True))
    styled = console.export_text(styles=True)

    assert "\x1b[" in styled  # colored when the console is a terminal
    assert "\x1b[1;31m" in styled  # the bad style (bold red) on the out-of-range caption and flags
