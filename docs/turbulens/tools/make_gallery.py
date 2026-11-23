"""
Regenerate the terminal screenshots used in the README and the docs.

Each screenshot is a real ``turbulens`` run on the bundled GASS cube,
captured with color and re-rendered by `rich` into an SVG
(``Console.save_svg``). Run from the repository root::

    python docs/turbulens/tools/make_gallery.py

It writes ``docs/turbulens/_static/cli/<name>.svg``. The CNN runs use ``--model
auto``, so the pretrained ensemble must be downloadable or already cached
(``turbulens download``). Your home folder is masked as ``<you>`` in the
output, keeping the line widths unchanged. The Fira Code fonts are
downloaded once from cdnjs and embedded in each SVG.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "turbulens" / "_static" / "cli"
CUBE = "examples/gass_0_0_1774393774.fits.gz"

RUNS = {
    "inspect": ["inspect", "--input", CUBE],
    "infer_real_data": ["infer", "--input", CUBE, "--integrate", "0:1201", "--real-data", "--model", "auto",
                        "--out", "predictions.ecsv"],
    "infer_no_real_data": ["infer", "--input", CUBE, "--integrate", "0:1201", "--model", "auto"],
    "infer_powerlaw": ["infer", "--input", CUBE, "--integrate", "0:1201", "--method", "fft-powerlaw"],
}
"""dict: Screenshot name -> ``turbulens`` arguments."""

# GitHub's dark palette, so the screenshots sit well on github.com in either theme.
THEME = TerminalTheme(
    (13, 17, 23), (230, 237, 243),
    [(72, 79, 88), (255, 123, 114), (63, 185, 80), (210, 153, 34),
     (88, 166, 255), (188, 140, 255), (57, 197, 207), (177, 186, 196)],
    [(110, 118, 129), (255, 161, 152), (86, 211, 100), (227, 179, 65),
     (121, 192, 255), (210, 168, 255), (86, 212, 221), (255, 255, 255)],
)


FONTS = {
    "Regular": "https://cdnjs.cloudflare.com/ajax/libs/firacode/6.2.0/woff2/FiraCode-Regular.woff2",
    "Bold": "https://cdnjs.cloudflare.com/ajax/libs/firacode/6.2.0/woff2/FiraCode-Bold.woff2",
}
"""dict: The Fira Code faces Rich's SVGs ask for (SIL Open Font License), embedded so they render anywhere."""


def _font_data_uri(face: str, cache_dir: Path) -> str:
    """The Fira Code ``face`` as a ``data:`` URI, downloaded once into ``cache_dir``."""
    path = cache_dir / f"FiraCode-{face}.woff2"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FONTS[face], timeout=30) as response:  # noqa: S310 -- fixed https URL
            path.write_bytes(response.read())
    return "data:font/woff2;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def embed_fonts(svg: str, cache_dir: Path) -> str:
    """
    Replace the SVG's web-font URLs with the fonts themselves.

    Browsers do not load external fonts for an SVG shown as an image (as
    GitHub shows README images), and the fallback monospace font's
    different glyph widths break the table borders apart.
    """
    for face in FONTS:
        pattern = re.compile(rf'src: local\("FiraCode-{face}"\),.*?format\("woff"\);', re.DOTALL)
        svg, count = pattern.subn(f'src: url("{_font_data_uri(face, cache_dir)}") format("woff2");', svg)
        if count != 1:
            raise RuntimeError(f"could not find the Fira Code {face} font declaration in Rich's SVG")
    return svg


# Runs the real CLI with one change: its consoles write ANSI color to the pipe as a terminal would.
# (Writing to a pipe, Rich turns color off, and on Windows it also falls back to legacy rendering.)
_RUN_CLI = """
import sys
from rich.console import Console
import turbulens.cli as cli
import turbulens.io.display as display
from turbulens._style import THEME

def make_console(stderr=False):
    return Console(file=sys.stderr if stderr else sys.stdout, width=display._NON_TTY_WIDTH, highlight=False,
                   theme=THEME, force_terminal=True, color_system="truecolor", legacy_windows=False)

display.make_console = make_console  # the commands import it from here when they run
sys.argv[0] = "turbulens"
cli.main()
"""


def capture(args: list[str]) -> str:
    """Run ``turbulens <args>`` and return its colored stdout and stderr, home folder masked."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", _RUN_CLI, *args], capture_output=True, env=env, cwd=ROOT,
                          check=False)
    if "--out" in args:  # the run only writes it to show the "Results written to" line
        (ROOT / args[args.index("--out") + 1]).unlink(missing_ok=True)
    home = str(Path.home())
    masked = home[: -len(Path.home().name)] + "<you>"
    output = (done.stdout.decode("utf-8") + done.stderr.decode("utf-8")).replace("\r\n", "\n")
    for form in (home, home.replace("\\", "/")):  # paths, and file:// links (always forward slashes)
        output = output.replace(form, masked.replace("\\", "/") if "/" in form else masked)
    return output.rstrip("\n")


# Color codes, and the hidden URL of a terminal hyperlink (OSC 8): neither takes up a column.
_ESCAPES = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;[^\x1b\a]*(?:\x1b\\|\a)")


def save(name: str, args: list[str], out_dir: Path, font_cache: Path) -> Path:
    """Capture one run and save it as ``<out_dir>/<name>.svg``, fonts embedded."""
    output = capture(args)
    plain = _ESCAPES.sub("", output)
    command = "turbulens " + " ".join(args)
    width = max([len(line.rstrip()) for line in plain.splitlines()] + [len(command) + 2])  # as wide as printed
    console = Console(record=True, width=width, force_terminal=True, legacy_windows=False, file=io.StringIO())
    console.print(Text("$ ", style="bold green") + Text(command, style="bold"))
    console.print(Text.from_ansi(output))
    path = out_dir / f"{name}.svg"
    path.write_text(embed_fonts(console.export_svg(title="turbulens", theme=THEME), font_cache), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output folder (default: docs/turbulens/_static/cli).")
    parser.add_argument("--font-cache", type=Path, default=Path(tempfile.gettempdir()) / "turbulens-gallery-fonts",
                        help="Where to keep the downloaded Fira Code fonts.")
    parser.add_argument("names", nargs="*", help=f"Screenshots to regenerate: {', '.join(RUNS)} (default: all).")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.names) - set(RUNS))
    if unknown:
        parser.error(f"unknown screenshot {', '.join(unknown)}; choose from {', '.join(RUNS)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names or RUNS:
        print(f"Wrote {save(name, RUNS[name], args.out_dir, args.font_cache).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
