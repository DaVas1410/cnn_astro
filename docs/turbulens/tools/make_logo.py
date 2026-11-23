"""
Build the turbulens logo files from one shared geometry.

The mark is a convolutional aperture: three rings of filter blocks, each
block skewed and each ring offset along a logarithmic spiral like the blades
of an iris diaphragm, around an open core showing a log-normal fractal density
field (the kind of 2D slice turbulens is trained on).

Usage::

    python docs/turbulens/tools/make_logo.py

writes, into ``docs/turbulens/_static/``:

- ``logo.svg`` / ``logo-dark.svg``: the full mark for light / dark backgrounds;
- ``favicon.svg``: a simplified mark (fewer, thicker blocks, a 4x4 core) that
  stays legible at 16 px and switches colors with the browser's color scheme;
- ``header.svg`` / ``header-dark.svg``: the README header card, the mark beside
  the name and tagline (text outlined, so no font is needed to render it);
- ``logo.png``: the full mark at 512 px on a transparent background;
- ``social-preview.png``: a 1280x640 card for link previews.

Everything is deterministic (the core field uses a fixed seed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STATIC = Path(__file__).resolve().parents[1] / "_static"
CENTER = 32.0  # the mark is drawn on a 64 x 64 canvas
CORE_RADIUS = 11.4

BLUE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
RING_COLORS = {"light": ["#184f95", "#256abf", "#3987e5"], "dark": ["#3987e5", "#6da7ec", "#9ec5f4"]}
RIM_COLOR = {"light": "#0d366b", "dark": "#cde2fb"}

# (inner radius, outer radius, blocks, angular gap, skew) per ring, outermost first.
FULL_RINGS = [(24.5, 29.5, 24, 0.11, 0.16), (18.5, 23.0, 16, 0.14, 0.20), (13.0, 17.0, 10, 0.20, 0.26)]
SMALL_RINGS = [(21.0, 29.5, 12, 0.20, 0.22), (13.0, 19.0, 8, 0.26, 0.28)]


def _arc(radius: float, a0: float, a1: float, n: int = 10) -> np.ndarray:
    angles = np.linspace(a0, a1, n)
    return np.column_stack([CENTER + radius * np.cos(angles), CENTER + radius * np.sin(angles)])


def ring_blocks(r_in: float, r_out: float, count: int, gap: float, skew: float) -> list[np.ndarray]:
    """Polygons for one ring: ``count`` blocks whose outer edge is rotated by ``skew`` radians."""
    offset = 1.1 * np.log(r_out)  # each ring starts further along a logarithmic spiral
    blocks = []
    for k in range(count):
        a0 = offset + 2 * np.pi * k / count + gap / 2
        a1 = offset + 2 * np.pi * (k + 1) / count - gap / 2
        outer = _arc(r_out, a0 + skew, a1 + skew)
        inner = _arc(r_in, a1, a0)
        blocks.append(np.vstack([outer, inner]))
    return blocks


def core_cells(cells_per_side: int) -> list[tuple[float, float, float, str]]:
    """Square cells ``(x, y, size, color)`` of a log-normal fractal field, quantized to the blue ramp."""
    rng = np.random.default_rng(11)
    n = 128
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    gaussian = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * k**-1.9))
    density = np.exp(1.3 * (gaussian - gaussian.mean()) / gaussian.std())
    block = n // cells_per_side
    coarse = density.reshape(cells_per_side, block, cells_per_side, block).mean(axis=(1, 3))

    size = 2 * CORE_RADIUS / cells_per_side
    origin = CENTER - CORE_RADIUS
    inside = [
        (i, j) for i in range(cells_per_side) for j in range(cells_per_side)
        if (origin + (j + 0.5) * size - CENTER) ** 2 + (origin + (i + 0.5) * size - CENTER) ** 2
        < (CORE_RADIUS + 0.7 * size) ** 2
    ]
    values = np.array([coarse[i, j] for i, j in inside])
    edges = np.quantile(values, np.linspace(0, 1, len(BLUE_RAMP) + 1)[1:-1])
    return [
        (origin + j * size, origin + i * size, size, BLUE_RAMP[int(np.searchsorted(edges, v))])
        for (i, j), v in zip(inside, values, strict=True)
    ]


def _svg_path(polygons: list[np.ndarray]) -> str:
    return "".join("M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in poly) + "Z" for poly in polygons)


def _svg_core(cells: list[tuple[float, float, float, str]]) -> str:
    by_color: dict[str, list[str]] = {}
    for x, y, size, color in cells:
        s = size + 0.04  # a hair of overlap hides anti-aliasing seams between cells
        by_color.setdefault(color, []).append(f"M{x:.2f} {y:.2f}h{s:.2f}v{s:.2f}h-{s:.2f}Z")
    return "".join(f'<path fill="{color}" d="{"".join(d)}"/>' for color, d in by_color.items())


def _mark_parts(rings, cells_per_side: int, ring_fills: list[str] | None, rim: str | None,
                ring_classes: list[str] | None = None) -> str:
    """The mark's SVG elements on its 64x64 canvas; they clip the core with ``#core``."""
    parts = []
    for index, spec in enumerate(rings):
        attr = f'class="{ring_classes[index]}"' if ring_classes else f'fill="{ring_fills[index]}"'
        parts.append(f"<path {attr} d=\"{_svg_path(ring_blocks(*spec))}\"/>")
    parts.append(f'<g clip-path="url(#core)">{_svg_core(core_cells(cells_per_side))}</g>')
    if rim:
        parts.append(f'<circle cx="32" cy="32" r="{CORE_RADIUS}" fill="none" stroke="{rim}" stroke-width="1.2"/>')
    return "".join(parts)


_CORE_CLIP = f'<defs><clipPath id="core"><circle cx="32" cy="32" r="{CORE_RADIUS}"/></clipPath></defs>'


def mark_svg(rings, cells_per_side: int, ring_fills: list[str] | None, rim: str | None,
             style: str = "", ring_classes: list[str] | None = None) -> str:
    """The mark as a standalone 64x64 SVG document."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">'
        f"<title>turbulens</title>{style}{_CORE_CLIP}"
        + _mark_parts(rings, cells_per_side, ring_fills, rim, ring_classes) + "</svg>\n"
    )


def _text_path(text: str, x: float, baseline: float, size: float, weight: str) -> str:
    """``text`` as an SVG path ``d`` (DejaVu Sans outlines), so it renders without the font installed."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MplPath
    from matplotlib.textpath import TextPath

    path = TextPath((0, 0), text, size=size, prop=FontProperties(family="DejaVu Sans", weight=weight))
    commands = []
    for vertices, code in path.iter_segments(simplify=False, curves=True):
        points = " ".join(f"{x + vx:.2f} {baseline - vy:.2f}" for vx, vy in vertices.reshape(-1, 2))
        letter = {MplPath.MOVETO: "M", MplPath.LINETO: "L", MplPath.CURVE3: "Q", MplPath.CURVE4: "C"}.get(code)
        commands.append("Z" if code == MplPath.CLOSEPOLY else f"{letter}{points}")
    return "".join(commands)


HEADER_THEME = {
    "light": {"card": "#ffffff", "border": "#d0d7de", "name": "#0d366b", "tagline": "#52514e"},
    "dark": {"card": "#161b22", "border": "#30363d", "name": "#e6edf3", "tagline": "#9198a1"},
}


def header_card_svg(theme: str) -> str:
    """The README header: the mark beside the name and tagline, on a rounded card."""
    colors = HEADER_THEME[theme]
    width, height, mark = 760, 200, 136
    offset, scale = (height - mark) / 2, mark / 64
    name = _text_path("turbulens", 200, 108, 76, "bold")
    tagline = _text_path("Turbulence parameters from a single 2D map", 203, 148, 22, "normal")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        f"<title>turbulens: turbulence parameters from a single 2D map</title>{_CORE_CLIP}"
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="20" fill="{colors["card"]}" '
        f'stroke="{colors["border"]}" stroke-width="2"/>'
        f'<g transform="translate({offset} {offset}) scale({scale})">'
        + _mark_parts(FULL_RINGS, 16, RING_COLORS[theme], RIM_COLOR[theme])
        + f'</g><path fill="{colors["name"]}" d="{name}"/><path fill="{colors["tagline"]}" d="{tagline}"/></svg>\n'
    )


def favicon_svg() -> str:
    """The simplified mark, recolored by the browser's color scheme."""
    light, dark = RING_COLORS["light"], RING_COLORS["dark"]
    style = (
        f"<style>.r0{{fill:{light[0]}}}.r1{{fill:{light[2]}}}"
        f"@media (prefers-color-scheme: dark){{.r0{{fill:{dark[0]}}}.r1{{fill:{dark[2]}}}}}</style>"
    )
    return mark_svg(SMALL_RINGS, 4, None, None, style=style, ring_classes=["r0", "r1"])


def draw_mark(ax, theme: str = "light", x0: float = 0.0, y0: float = 0.0, scale: float = 1.0) -> None:
    """Draw the full mark with matplotlib, its 64x64 box mapped to ``(x0, y0)`` at ``scale`` units/px."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    def place(points):
        return np.column_stack([x0 + points[:, 0] * scale, y0 + (64 - points[:, 1]) * scale])

    for spec, color in zip(FULL_RINGS, RING_COLORS[theme], strict=True):
        for block in ring_blocks(*spec):
            ax.add_patch(Polygon(place(block), closed=True, facecolor=color, edgecolor="none"))
    clip = Circle((x0 + CENTER * scale, y0 + CENTER * scale), CORE_RADIUS * scale, transform=ax.transData)
    for x, y, size, color in core_cells(16):
        s = size + 0.04
        cell = Rectangle((x0 + x * scale, y0 + (64 - y - s) * scale), s * scale, s * scale,
                         facecolor=color, edgecolor="none")
        ax.add_patch(cell)
        cell.set_clip_path(clip)
    ax.add_patch(Circle((x0 + CENTER * scale, y0 + CENTER * scale), CORE_RADIUS * scale, fill=False,
                        edgecolor=RIM_COLOR[theme], linewidth=1.2 * scale * 72 / 100))


def write_pngs(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_mark(ax, "light")
    fig.savefig(out_dir / "logo.png", transparent=True)
    plt.close(fig)

    surface = "#0d1117"
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(surface)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 640)
    ax.axis("off")
    draw_mark(ax, "dark", x0=120, y0=160, scale=5.0)
    ax.text(520, 350, "turbulens", color="#e6edf3", fontsize=80, fontweight="bold", va="center",
            family="DejaVu Sans")
    ax.text(524, 262, "Turbulence parameters from a single 2D map", color="#c3c2b7", fontsize=21,
            va="center", family="DejaVu Sans")
    fig.savefig(out_dir / "social-preview.png", facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", type=Path, default=STATIC, help="Output folder (default: docs/turbulens/_static).")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for theme, name in (("light", "logo.svg"), ("dark", "logo-dark.svg")):
        (args.out_dir / name).write_text(mark_svg(FULL_RINGS, 16, RING_COLORS[theme], RIM_COLOR[theme]))
    (args.out_dir / "favicon.svg").write_text(favicon_svg())
    for theme, name in (("light", "header.svg"), ("dark", "header-dark.svg")):
        (args.out_dir / name).write_text(header_card_svg(theme))
    write_pngs(args.out_dir)
    print(f"Wrote logo files to {args.out_dir}")


if __name__ == "__main__":
    main()
