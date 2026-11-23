import math
import warnings

import numpy as np
import pytest
from astropy.table import MaskedColumn, QTable

from turbulens.io.display import (
    MISSING,
    format_measurement,
    format_number,
    message_panels,
    render_text,
    results_table,
    results_written_text,
)


@pytest.mark.parametrize("value", [None, np.ma.masked, math.nan, math.inf, -math.inf])
def test_format_number_shows_dash_for_values_not_computed(value):
    assert format_number(value) == MISSING


@pytest.mark.parametrize(("value", "expected"), [
    (48.0, "48"),
    (0.0, "0"),
    (-11.915000915527344, "-11.92"),
    (148.39999389648438, "148.4"),
    (0.8871838915269291, "0.8872"),
    (0.08, "0.08"),
    (-1.1396834262892155, "-1.14"),
    (-494700.0, "-494700"),
    (2.4771e-4, "2.477e-04"),
])
def test_format_number_rounds_to_four_significant_figures(value, expected):
    assert format_number(value) == expected


@pytest.mark.parametrize(("value", "std", "expected"), [
    (18.930530548095703, 5.142858982086182, ("18.9", "5.1")),
    (-1.7842986583709717, 0.674095869064331, ("-1.78", "0.67")),
    (41.09661865234375, 15.4, ("41", "15")),
    (1.0, math.nan, ("1", MISSING)),
    (math.nan, 0.5, (MISSING, MISSING)),
])
def test_format_measurement_rounds_value_to_uncertainty_precision(value, std, expected):
    assert format_measurement(value, std) == expected


def _cells(text, first):
    """The cells of the table row whose first cell is ``first``."""
    rows = ([cell.strip() for cell in line.replace("│", "|").split("|") if cell.strip()] for line in text.splitlines())
    return next(cells for cells in rows if cells[:1] == [first])


def _predictions(targets, values, stds=None, clipped=None):
    table = QTable()
    table["image_index"] = [0] * len(targets)
    table["target"] = targets
    table["value"] = values
    table["epistemic_std"] = (
        MaskedColumn([0.0] * len(targets), mask=[True] * len(targets)) if stds is None else stds
    )
    if clipped is not None:
        table.meta["clipped_pixel_fraction_max"] = clipped
    return table


def test_results_table_shows_value_alone_without_uncertainty_and_hides_single_image_index():
    text = render_text(results_table(_predictions(["k_min", "k_max"], [1.0, 48.0])))

    assert "nan" not in text.lower()
    assert "Image" not in text
    assert _cells(text, "k_max")[:3] == ["k_max", "48", "✓"]


def test_results_table_merges_value_and_spread_with_spread_precision():
    text = render_text(results_table(_predictions(["k_min", "sigma"], [1.3792, 4.1419], [0.5257, 0.1171])))

    assert _cells(text, "k_min")[:3] == ["k_min", "1.38 ± 0.53", "✓"]
    assert _cells(text, "σ")[:2] == ["σ", "4.14 ± 0.12"]
    assert "Value ± 1σ" in text


def test_results_table_falls_back_to_ascii():
    text = render_text(results_table(_predictions(["sigma", "r_squared"], [4.1419, 0.99], [0.1171, math.nan]),
                                     unicode=False))

    assert all(ch.isascii() or "─" <= ch <= "╿" for ch in text)  # box drawing aside
    assert _cells(text, "sigma")[:3] == ["sigma", "4.14 +/- 0.12", "ok"]
    assert _cells(text, "R^2")[:3] == ["R^2", "0.99", "ok good fit"]


@pytest.mark.parametrize(("r_squared", "flag"), [(0.2619, "⚠ poor fit"), (0.9959, "✓ good fit")])
def test_results_table_flags_poor_power_law_fit(r_squared, flag):
    text = render_text(results_table(_predictions(["k_min", "alpha", "r_squared"], [24.0, 7.409, r_squared])))

    assert _cells(text, "R²")[2] == flag
    assert _cells(text, "α")[2] == ("⚠ poor fit" if r_squared < 0.9 else "✓")


def test_results_table_flags_spread_larger_than_value():
    text = render_text(results_table(_predictions(["sigma"], [0.023], [0.051])))

    assert _cells(text, "σ")[2] == "⚠ spread ≥ value"


@pytest.mark.parametrize(("clipped", "suggest"), [(1.0, True), (1.0, False), (0.6, False)])
def test_results_table_marks_out_of_distribution_runs(clipped, suggest):
    table = _predictions(["k_min"], [12.5], [3.1], clipped=clipped)

    text = render_text(results_table(table, suggest_real_data=suggest), width=120)

    assert _cells(text, "k_min")[2] == "✗ out of range"
    flat = " ".join(text.split())
    assert f"{clipped:.0%} of pixels clipped: input is outside the training range." in flat
    assert ("Did you mean --real-data?" in flat) is suggest


def test_results_table_is_not_out_of_distribution_below_half_clipped():
    text = render_text(results_table(_predictions(["k_min"], [1.38], [0.53], clipped=0.5)))

    assert "out of range" not in text and "clipped" not in text


def _record(*messages):
    """Capture ``(message, category)`` pairs as real warning records."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for message, category in messages:
            warnings.warn(message, category, stacklevel=1)
    return caught


def _fits_fixes():
    from astropy.wcs import FITSFixedWarning

    return [
        ("'unitfix' made the change 'DEG' -> 'deg'", FITSFixedWarning),
        ("'spcfix' made the change 'VELO-LSR' -> 'VOPT'", FITSFixedWarning),
    ]


def _panel_text(panels):
    return render_text(*panels, width=200)


def test_message_panels_collapse_and_deduplicate_messages():
    panels = message_panels(_record(("first line\n   second line", UserWarning), ("first line\n   second line", UserWarning)))

    assert _panel_text(panels).count("UserWarning: first line second line") == 1
    assert message_panels([]) == []


def test_message_panels_route_routine_fixes_to_notes_and_the_rest_to_warnings():
    panels = message_panels(_record(*_fits_fixes(), ("12% of pixels hit the clip boundary", UserWarning)))

    assert [str(panel.title) for panel in panels] == ["[warn]Warnings[/warn]", "[label]Notes[/label]"]
    warnings_text, notes_text = (render_text(panel, width=200) for panel in panels)
    assert "UserWarning: 12% of pixels hit the clip boundary" in warnings_text
    assert "routine" not in warnings_text and "unitfix" not in warnings_text
    assert "2 routine FITS header fixes applied (--verbose to list)" in notes_text


@pytest.mark.parametrize("verbose", [False, True])
def test_message_panels_with_only_routine_fixes_show_only_notes(verbose):
    text = _panel_text(message_panels(_record(*_fits_fixes()), verbose=verbose))

    assert "Notes" in text
    assert "Warnings" not in text


def test_message_panels_verbose_lists_routine_fixes_in_notes():
    text = _panel_text(message_panels(_record(*_fits_fixes()), verbose=True))

    assert "FITSFixedWarning: 'unitfix' made the change 'DEG' -> 'deg'" in text
    assert "FITSFixedWarning: 'spcfix' made the change 'VELO-LSR' -> 'VOPT'" in text
    assert "routine" not in text


def test_message_panels_skip_categories_already_shown():
    from turbulens.inference import ClippedPixelsWarning

    panels = message_panels(_record(("100% of pixels clipped", ClippedPixelsWarning)), skip=(ClippedPixelsWarning,))

    assert panels == []


def test_results_written_text_links_the_path_and_summarizes(tmp_path):
    table = _predictions(["k_min", "k_max"], [1.0, 2.0])
    table.meta["n_members"] = 5
    out = tmp_path / "cnn.ecsv"

    text = results_written_text(out, table)

    assert text.plain == f"Results written to {out}  · 2 parameters · 5 members"
    links = [span.style.link for span in text.spans if getattr(span.style, "link", None)]
    assert links == [out.resolve().as_uri()]
