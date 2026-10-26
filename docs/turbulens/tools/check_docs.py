"""Content checks for the turbulens docs that a Sphinx build cannot catch.

The docs explain the method and leave every results number to the paper, never link
into private or untracked repository paths, and cite only references that carry a
verifiable identifier. Run from anywhere:

    python docs/turbulens/tools/check_docs.py [--root docs/turbulens] [--package turbulens]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

_NUMBER = r"\d+(?:\.\d+)?"
_METRIC = r"(?:R\^2|R²|R\^\{2\}|\bR2\b|\bRMSE\b|\bMSE\b|\bMAE\b|\bCRPS\b|coverage|covers|accuracy)"
_VALUE = r"(?:\d*\.\d+|\d+(?:\.\d+)?\s*%)"
_RESULTS_PATTERNS = [
    # a metric followed closely by a decimal or a percentage, in any phrasing:
    # "R^2 = 0.93", "RMSE: 1.2", "R^2 was 0.93", "MAE 0.4", "coverage of 63%"
    re.compile(_METRIC + r"[^\n]{0,25}?" + _VALUE),
    # throughput / timing
    re.compile(_NUMBER + r"\s*(?:img/s|images?/s|images? per second|ms per image|ms/image)"),
]
# Lines that define a metric (e.g. R^2 = 1 - SS_res/SS_tot) are exempt; they need real LaTeX markers.
_DEFINITION_MARKERS = (r"\mathrm{SS}", r"\frac", r"\sum")
_FORBIDDEN_PHRASES = ("aleatoric head", "predicted variance", "variance head")
_PRIVATE_PATH = re.compile(r"(?<![\w/.-])(?:archive/|outputs/|data/raw|docs/superpowers|src/|notebooks/)")
_BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)", re.S)
_BIB_IDENTIFIER = re.compile(r"^\s*(?:doi|eprint|url)\s*=", re.M | re.I)


def find_forbidden(text: str) -> list[str]:
    """Return results-number and forbidden-phrase hits in `text`."""
    hits = []
    if not any(marker in text for marker in _DEFINITION_MARKERS):
        for pattern in _RESULTS_PATTERNS:
            hits += [f"results number: {m.group(0)!r}" for m in pattern.finditer(text)]
    lowered = text.lower()
    hits += [f"forbidden phrase: {p!r}" for p in _FORBIDDEN_PHRASES if p in lowered]
    return hits


def find_private_paths(text: str) -> list[str]:
    """Return references to untracked or local-only repository paths in `text`."""
    return [f"private path: {m.group(0)!r}" for m in _PRIVATE_PATH.finditer(text)]


def find_unverifiable_bib_entries(bib_text: str) -> list[str]:
    """Return the keys of BibTeX entries with no ``doi``, ``eprint`` or ``url`` field."""
    return [
        key for key, body in _BIB_ENTRY.findall(bib_text)
        if not _BIB_IDENTIFIER.search(body.replace(",", ",\n"))
    ]


def docstrings(source: str) -> list[tuple[int, str]]:
    """Return ``(first line number, text)`` for every module, class and function docstring."""
    tree = ast.parse(source)
    found = []
    for node in [tree, *ast.walk(tree)]:
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.append((first.lineno, first.value.value))
    return sorted(set(found))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--exclude", nargs="*", default=[], help="file basenames to skip")
    parser.add_argument("--package", default=None,
                        help="also scan this package's docstrings (autodoc renders them), "
                        "skipping tests/ and training/")
    args = parser.parse_args(argv)
    root = Path(args.root)

    problems = []
    if args.package:
        package = Path(args.package)
        for path in sorted(package.rglob("*.py")):
            if {"tests", "training", "__pycache__"} & set(path.relative_to(package).parts):
                continue
            for start, text in docstrings(path.read_text(encoding="utf-8")):
                for offset, line in enumerate(text.splitlines()):
                    for hit in find_forbidden(line) + find_private_paths(line):
                        problems.append(f"{path.relative_to(package.parent)}:{start + offset}: {hit}")
    for path in sorted(root.rglob("*.rst")):
        if "_build" in path.parts or path.name in args.exclude:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for hit in find_forbidden(line) + find_private_paths(line):
                problems.append(f"{path.relative_to(root)}:{lineno}: {hit}")
    bib = root / "references.bib"
    if bib.exists():
        for key in find_unverifiable_bib_entries(bib.read_text(encoding="utf-8")):
            problems.append(f"references.bib: entry {key!r} has no doi, eprint or url")

    for problem in problems:
        print(problem)
    if problems:
        return 1
    print("check_docs: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
