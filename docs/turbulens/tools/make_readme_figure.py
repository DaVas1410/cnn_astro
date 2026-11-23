"""
Draw the README example figure: an integrated GASS HI map and its power spectrum.

The left panel is the velocity-integrated map ``turbulens infer --integrate``
feeds the model. The right panel is its azimuthally averaged power spectrum,
with the classical power-law inertial range (``--method fft-powerlaw``) shaded
and the CNN ensemble's ``k_min``/``k_max`` (``--method cnn --model auto``)
marked, so the two estimates can be compared at a glance.

Usage::

    python docs/tools/make_readme_figure.py --out docs/_static/gass_example.png

The CNN part needs the pretrained ensemble (``--model auto``: Hugging Face,
cached after the first run) or ``--output-root`` for a local training output.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from turbulens.inference import Inferencer
from turbulens.io.image_loaders import load_input
from turbulens.models.registry import EnsembleRegistry
from turbulens.spectral import fit_inertial_range, radial_power_spectrum

ROOT = Path(__file__).resolve().parents[1]
CUBE = ROOT.parent / "examples" / "gass_0_0_1774393774.fits.gz"
if not CUBE.exists():  # research-repo layout: docs/turbulens/tools/
    CUBE = ROOT.parents[1] / "examples" / "gass_0_0_1774393774.fits.gz"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
BAND_FILL = "#cde2fb"
MAP_CMAP = LinearSegmentedColormap.from_list(
    "turbulens_blue", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
)


def _predict_cnn(image: np.ndarray, output_root: str | None) -> dict[str, tuple[float, float]]:
    members = EnsembleRegistry.resolve(output_root) if output_root else EnsembleRegistry.fetch().members_dir
    table = Inferencer(members).predict(image, real_data=True)
    return {str(row["target"]): (float(row["value"]), float(row["epistemic_std"])) for row in table}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, required=True, help="Output image path (.png or .svg).")
    parser.add_argument("--output-root", default=None, help="Local training output root instead of the Hub ensemble.")
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = load_input(CUBE)
        image = cube.project(0, cube.n_channels)
        velocities = cube.velocity_axis().to_value("km/s")
    data = np.asarray(image.data[0], dtype=float)  # a projected Cube is (1, H, W)
    channel_width_kms = abs(float(velocities[1] - velocities[0]))
    column = data * channel_width_kms  # K km/s

    k, power = radial_power_spectrum(data)
    fit = fit_inertial_range(data)
    cnn = _predict_cnn(image.data, args.output_root)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY,
    })
    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)

    lo, hi = np.percentile(column, [1, 99.5])
    shown = ax_map.imshow(column, origin="lower", cmap=MAP_CMAP, vmin=lo, vmax=hi, interpolation="nearest")
    ax_map.set_title("GASS HI, (l, b) = (0°, 0°), integrated", loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax_map.set_xlabel("pixel")
    ax_map.set_ylabel("pixel")
    bar = fig.colorbar(shown, ax=ax_map, fraction=0.046, pad=0.03)
    bar.set_label("integrated intensity (K km/s)", color=TEXT_SECONDARY)
    bar.outline.set_edgecolor(GRID)

    ax_spec.set_facecolor(SURFACE)
    ax_spec.axvspan(fit["k_min"], fit["k_max"], color=BAND_FILL, lw=0, zorder=0,
                    label=f"power-law inertial range (slope {fit['alpha']:.2f})")
    ax_spec.loglog(k, power, color=SERIES_BLUE, lw=2, zorder=2, label="power spectrum P(k)")
    for name in ("k_min", "k_max"):
        value, std = cnn[name]
        ax_spec.axvline(value, color=SERIES_ORANGE, lw=2, ls=(0, (4, 2)), zorder=3,
                        label="CNN k_min, k_max" if name == "k_min" else None)
        ax_spec.annotate(f"CNN {name}\n{value:.1f} ± {std:.1f}", xy=(value, 1), xycoords=("data", "axes fraction"),
                         xytext=(6 if name == "k_min" else -6, -8), textcoords="offset points",
                         ha="left" if name == "k_min" else "right", va="top", fontsize=9, color=TEXT_PRIMARY)
    ax_spec.set_title("Power spectrum, with both estimates", loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax_spec.set_xlabel("k (cycles per image side)")
    ax_spec.set_ylabel("P(k)")
    ax_spec.grid(True, which="major", color=GRID, lw=0.8)
    ax_spec.tick_params(which="both", length=0)
    for spine in ax_spec.spines.values():
        spine.set_visible(False)
    ax_spec.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)

    sigma, beta = cnn["sigma"], cnn["beta"]
    fig.text(0.995, 0.015, f"CNN ensemble: sigma = {sigma[0]:.2f} ± {sigma[1]:.2f},  beta = {beta[0]:.2f} ± {beta[1]:.2f}",
             ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
