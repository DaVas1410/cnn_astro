import json

from turbulens.io.breakdown import CubeBreakdown
from turbulens.io.report import format_breakdown_cli, write_breakdown_html, write_breakdown_json


def _breakdown(source_path="<script>alert(1)</script>.fits"):
    return CubeBreakdown(
        shape=(5, 3, 3),
        n_channels=5,
        has_spectral_axis=True,
        is_projected_image=False,
        unit="K",
        wcs_summary={"ctype": ["RA---SIN", "DEC--SIN", "VELO-LSR"], "crval": [0.0, 0.0, -1000.0]},
        velocity_range=(-1.0, 1.0),
        rest_frequency_ghz=None,
        spectral_ctype="VELO-LSR",
        pixel_stats={"min": 0.0, "max": 44.0, "mean": 22.0, "std": 10.0, "nan_fraction": 0.0},
        spectral_estimate_band_edge=None,
        spectral_estimate_power_law=None,
        source_path=source_path,
    )


def test_format_breakdown_cli_includes_key_fields():
    text = format_breakdown_cli(_breakdown(source_path="cube.fits"))

    assert "cube.fits" in text
    assert "5 x 3 x 3" in text
    assert "-1.00 to 1.00 km/s" in text
    assert "VELO-LSR" in text


def test_format_breakdown_cli_shows_dash_for_values_not_computed():
    breakdown = _breakdown(source_path="cube.fits")
    breakdown.unit = None
    breakdown.pixel_stats = {"min": None, "max": None, "mean": None, "std": None, "nan_fraction": 1.0}

    text = format_breakdown_cli(breakdown)

    assert "None" not in text
    unit_row = next(line for line in text.splitlines() if "Unit" in line)
    assert unit_row.split() == ["Unit", "-"]
    stats_row = text.splitlines()[next(i for i, line in enumerate(text.splitlines()) if "NaN fraction" in line) + 2]
    cells = [cell.strip() for cell in stats_row.replace("│", "|").split("|") if cell.strip()]
    assert cells == ["-", "-", "-", "-", "100.00%"]


def test_format_breakdown_cli_shows_wcs_axis_units():
    breakdown = _breakdown(source_path="cube.fits")
    breakdown.wcs_summary = {**breakdown.wcs_summary, "cunit": ["deg", "deg", "m/s"], "pixel_scales": [0.08, 0.08, 824.5]}

    text = format_breakdown_cli(breakdown)

    rows = ([cell.strip() for cell in line.replace("│", "|").split("|") if cell.strip()] for line in text.splitlines())
    assert next(cells for cells in rows if cells[:1] == ["3"]) == ["3", "VELO-LSR", "-1000", "824.5", "m/s"]


def test_format_breakdown_cli_shows_dash_for_missing_wcs_units():
    text = format_breakdown_cli(_breakdown(source_path="cube.fits"))

    row = next(line for line in text.splitlines() if "RA---SIN" in line)
    cells = [cell.strip() for cell in row.replace("│", "|").split("|") if cell.strip()]
    assert cells == ["1", "RA---SIN", "0", "-", "-"]


def test_format_breakdown_cli_for_projected_image_has_no_velocity_line():
    breakdown = _breakdown(source_path="image.png")
    breakdown.has_spectral_axis = False
    breakdown.is_projected_image = True
    breakdown.velocity_range = None
    breakdown.spectral_ctype = None

    text = format_breakdown_cli(breakdown)

    assert "already-projected image" in text.lower()


def test_format_breakdown_cli_includes_band_edge_estimate_when_present():
    breakdown = _breakdown(source_path="image.png")
    breakdown.spectral_estimate_band_edge = {"k_min": 10.0, "k_max": 60.0}

    text = format_breakdown_cli(breakdown)

    row = next(line for line in text.splitlines() if "Band edge" in line)
    cells = [cell.strip() for cell in row.replace("│", "|").split("|") if cell.strip()]
    assert cells[:5] == ["Band edge", "10", "60", "-", "-"]


def test_format_breakdown_cli_includes_power_law_estimate_when_present():
    breakdown = _breakdown(source_path="image.png")
    breakdown.spectral_estimate_power_law = {"alpha": -3.67, "r_squared": 0.99, "k_min": 2.0, "k_max": 61.0}

    text = format_breakdown_cli(breakdown)

    row = next(line for line in text.splitlines() if "Power law" in line)
    cells = [cell.strip() for cell in row.replace("│", "|").split("|") if cell.strip()]
    assert cells[:5] == ["Power law", "2", "61", "-3.67", "0.99"]


def test_format_breakdown_cli_marks_a_poor_power_law_fit_as_not_meaningful():
    breakdown = _breakdown(source_path="image.png")
    breakdown.spectral_estimate_power_law = {"alpha": 7.409, "r_squared": 0.2619, "k_min": 24.0, "k_max": 28.0}

    text = format_breakdown_cli(breakdown)

    row = next(line for line in text.splitlines() if "Power law" in line)
    cells = [cell.strip() for cell in row.replace("│", "|").split("|") if cell.strip()]
    assert cells == ["Power law", "24", "28", "7.409", "0.2619", "⚠ poor fit, not meaningful"]
    assert "R²" in text and "α" in text


def test_format_breakdown_cli_omits_spectral_estimate_lines_when_none():
    text = format_breakdown_cli(_breakdown())

    assert "Classical FFT estimates" not in text


def test_write_breakdown_json_round_trips(tmp_path):
    path = tmp_path / "breakdown.json"
    write_breakdown_json(_breakdown(source_path="cube.fits"), path)

    payload = json.loads(path.read_text())
    assert payload["shape"] == [5, 3, 3]
    assert payload["velocity_range"] == [-1.0, 1.0]
    assert payload["source_path"] == "cube.fits"


def test_write_breakdown_html_has_real_tables_with_terminal_rounding(tmp_path):
    breakdown = _breakdown(source_path="cube.fits")
    breakdown.pixel_stats = {"min": -11.915000915527344, "max": 148.39999389648438, "mean": 3.13, "std": 12.99,
                             "nan_fraction": 0.0}
    breakdown.spectral_estimate_power_law = {"alpha": 7.409, "r_squared": 0.2619, "k_min": 24.0, "k_max": 28.0}
    path = tmp_path / "breakdown.html"

    write_breakdown_html(breakdown, path)

    content = path.read_text(encoding="utf-8")
    assert content.count("<table>") == 3  # WCS, pixel statistics, classical estimates
    assert "<dt>Unit</dt><dd>K</dd>" in content
    assert '<td class="num">-11.92</td>' in content and '<td class="num">148.4</td>' in content
    assert "148.39999" not in content
    assert "<tr class=flag>" in content and "poor fit, not meaningful" in content
    assert "prefers-color-scheme: dark" in content
    assert "<script" not in content and "http" not in content


def test_write_breakdown_html_escapes_source_path(tmp_path):
    path = tmp_path / "breakdown.html"
    write_breakdown_html(_breakdown(), path)

    content = path.read_text()
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content
