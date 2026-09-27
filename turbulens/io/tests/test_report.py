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
    assert "(5, 3, 3)" in text
    assert "-1.00" in text and "1.00" in text
    assert "VELO-LSR" in text


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

    assert "band edge, synthetic data" in text
    assert "k_min=10.0, k_max=60.0" in text


def test_format_breakdown_cli_includes_power_law_estimate_when_present():
    breakdown = _breakdown(source_path="image.png")
    breakdown.spectral_estimate_power_law = {"alpha": -3.67, "r_squared": 0.99, "k_min": 2.0, "k_max": 61.0}

    text = format_breakdown_cli(breakdown)

    assert "power-law inertial range, real clouds" in text
    assert "alpha=-3.67, r_squared=0.99, k_min=2.0, k_max=61.0" in text


def test_format_breakdown_cli_omits_spectral_estimate_lines_when_none():
    text = format_breakdown_cli(_breakdown())

    assert "Classical FFT estimate" not in text


def test_write_breakdown_json_round_trips(tmp_path):
    path = tmp_path / "breakdown.json"
    write_breakdown_json(_breakdown(source_path="cube.fits"), path)

    payload = json.loads(path.read_text())
    assert payload["shape"] == [5, 3, 3]
    assert payload["velocity_range"] == [-1.0, 1.0]
    assert payload["source_path"] == "cube.fits"


def test_write_breakdown_html_escapes_source_path(tmp_path):
    path = tmp_path / "breakdown.html"
    write_breakdown_html(_breakdown(), path)

    content = path.read_text()
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content
