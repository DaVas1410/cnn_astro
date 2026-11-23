#!/usr/bin/env python3
"""Build a plain PDF report of the two-database spectral verification.

Reads the verify JSON summaries and the PNG figures from outputs/spectra_check/
and lays them out as a simple, minimally-styled document.

Run:
    .venv_py311/Scripts/python.exe scripts/build_report_pdf.py
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "spectra_check")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=17, spaceAfter=4, textColor=colors.HexColor("#1a1a1a"))
SUBT = ParagraphStyle("Subt", parent=styles["Normal"], fontSize=9.5,
                      textColor=colors.HexColor("#555555"), spaceAfter=12)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=12, spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1a1a1a"))
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13,
                      textColor=colors.HexColor("#222222"))
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=8.5,
                      textColor=colors.HexColor("#555555"), spaceBefore=6)
CAP = ParagraphStyle("Cap", parent=NOTE, spaceBefore=3, spaceAfter=12)


def _load(stem):
    with open(os.path.join(OUT_DIR, f"{stem}_verify.json")) as fh:
        return json.load(fh)


def _table(data, col_widths):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#999999")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("ALIGN", (1, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def _fig(fname, max_w):
    path = os.path.join(OUT_DIR, fname)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = max_w / w
    return Image(path, width=max_w, height=h * scale)


def main():
    d1, d2 = _load("db1_kmax128_256x256_1000"), _load("db2_kmax96_256x256_1000")
    pct = lambda x: f"{x*100:.0f}%"
    story = []

    story.append(Paragraph("Fractal-Cloud Databases: Spectral Verification", H1))
    story.append(Paragraph(
        "Two synthetic databases verified against their 1D (azimuthal) Fourier spectra &nbsp;|&nbsp; "
        "256&times;256 slices, 1000 images each &nbsp;|&nbsp; 2026-07-10", SUBT))

    story.append(Paragraph(
        "Both databases are identical except for the upper spectral cutoff k_max, the single "
        "variable under test. Per-image k_min is sampled uniformly on [1, 32]; sigma is fixed "
        "at 2.0 and beta at &minus;5/3. Both databases pass the spectral sanity gate: the "
        "measured cutoff matches the target k_max within &plusmn;3 wavenumbers and k_min is "
        "recovered with R&sup2; = 0.998 on a 200-image sample per database.", BODY))

    story.append(Paragraph("1. Configuration", H2))
    story.append(_table([
        ["Parameter", "Database 1", "Database 2"],
        ["Resolution", "256 x 256 x 1", "256 x 256 x 1"],
        ["Nyquist limit", "128", "128"],
        ["k_min", "U(1, 32) per image", "U(1, 32) per image"],
        ["k_max", "128 (= Nyquist)", "96"],
        ["sigma", "2.0 (fixed)", "2.0 (fixed)"],
        ["beta", "-5/3", "-5/3"],
        ["Images", "1000", "1000"],
        ["File", "db1_kmax128_256x256_1000.h5", "db2_kmax96_256x256_1000.h5"],
    ], [4.2 * cm, 5.6 * cm, 5.6 * cm]))

    story.append(Paragraph("2. Verification (200-image sample per database)", H2))
    story.append(_table([
        ["Metric", "DB1 (k_max=128)", "DB2 (k_max=96)"],
        ["k_min - R2", f"{d1['kmin_r2']:.3f}", f"{d2['kmin_r2']:.3f}"],
        ["k_min - MAE", f"{d1['kmin_mae']:.2f}", f"{d2['kmin_mae']:.2f}"],
        ["k_min - within +/-2", pct(d1['kmin_within2_frac']), pct(d2['kmin_within2_frac'])],
        ["k_max - measured median", f"{d1['kmax_median']:.1f}", f"{d2['kmax_median']:.1f}"],
        ["k_max - target", str(d1['target_kmax']), str(d2['target_kmax'])],
        ["k_max - MAE", f"{d1['kmax_mae']:.2f}", f"{d2['kmax_mae']:.2f}"],
        ["k_max - within +/-2", pct(d1['kmax_within2_frac']), pct(d2['kmax_within2_frac'])],
        ["Sanity gate", "PASS" if d1['sanity_pass'] else "FAIL",
                        "PASS" if d2['sanity_pass'] else "FAIL"],
    ], [5.4 * cm, 5.0 * cm, 5.0 * cm]))
    story.append(Paragraph(
        "Note: k_max R&sup2; is undefined (nan) because k_max is constant within each database "
        "(zero label variance); MAE and the measured-median check are the meaningful statistics. "
        "DB1's median of 127 vs 128 reflects the single-bin resolution at the Nyquist edge.", NOTE))

    story.append(PageBreak())
    story.append(Paragraph("3. Parameter distributions", H2))
    story.append(_fig("param_distributions.png", 16 * cm))
    story.append(Paragraph(
        "k_min is uniform on [1, 32] and overlaps between the two databases; k_max is a single "
        "spike per database (128 vs 96); sigma is a delta at 2.0.", CAP))
    story.append(_fig("measured_vs_label.png", 12 * cm))
    story.append(Paragraph(
        "Measured spectral band edge vs stored label. k_min tracks the identity line; k_max "
        "collapses onto its two fixed target values.", CAP))

    story.append(PageBreak())
    story.append(Paragraph("4. Example power spectra", H2))
    story.append(Paragraph(
        "Power is numerical noise (~1e-16) below k_min, turns on sharply at the green line, "
        "follows the power law, and is band-limited at k_max (red).", BODY))
    story.append(Spacer(1, 6))
    story.append(_fig("db1_kmax128_256x256_1000_spectra_examples.png", 16 * cm))
    story.append(Paragraph("DB1 (k_max = 128): power extends to the Nyquist edge.", CAP))
    story.append(_fig("db2_kmax96_256x256_1000_spectra_examples.png", 16 * cm))
    story.append(Paragraph("DB2 (k_max = 96): band-limited below Nyquist at k_max = 96.", CAP))

    out = os.path.join(OUT_DIR, "spectral_verification_report.pdf")
    SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Fractal-Cloud Databases: Spectral Verification",
    ).build(story)
    print("wrote", out, round(os.path.getsize(out) / 1024), "KB")


if __name__ == "__main__":
    main()
