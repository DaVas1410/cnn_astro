"""The `turbulens` command-line interface: `infer` and `inspect` subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from astropy import units as u

from turbulens import __version__
from turbulens.inference import Inferencer
from turbulens.io.breakdown import describe_cube
from turbulens.io.cube import Cube
from turbulens.io.image_loaders import load_input
from turbulens.io.report import format_breakdown_cli, write_breakdown_html, write_breakdown_json
from turbulens.io.results import write_results
from turbulens.models.registry import EnsembleRegistry
from turbulens.spectral import SpectralEstimator


def build_parser() -> argparse.ArgumentParser:
    """
    Build the `turbulens` command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        A parser with two subcommands: ``infer`` (predict physical
        targets from a FITS cube or image) and ``inspect`` (summarize a
        cube's shape, WCS, and pixel statistics without running a
        model). See each subcommand's own ``--help`` for its arguments;
        the full rendered reference is in :doc:`/cli`.
    """
    parser = argparse.ArgumentParser(prog="turbulens")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Predict fractal turbulence parameters from a FITS slice or image.")
    infer.add_argument(
        "--config", default=None,
        help="Path to a JSON file of infer options (keys: input, hdu, channel, integrate, velocity, "
        "method, model, target, real_data, output_root, out -- same names as the flags below, with "
        "dashes as underscores). Fills in any option not explicitly passed as a flag; an explicit "
        "flag always overrides the matching JSON key.",
    )
    infer.add_argument("--input", default=None, help="Path to a FITS cube, or a PNG/TIFF/.npy image.")
    infer.add_argument("--hdu", type=int, default=None, help="FITS HDU index (FITS input only). Default 0.")
    channel_group = infer.add_mutually_exclusive_group(required=False)
    channel_group.add_argument("--channel", type=int, default=None, help="0-indexed channel to select.")
    channel_group.add_argument(
        "--integrate", default=None,
        help="0-indexed half-open channel range LOW:HIGH to sum over, e.g. --integrate 10:20 sums "
        "channels 10-19. Run 'turbulens inspect' on the same --input first to see its channel count "
        "and velocity range before picking bounds.",
    )
    channel_group.add_argument(
        "--velocity", nargs=2, type=float, default=None,
        metavar=("LOW", "HIGH"), help="Velocity range in km/s to project over, e.g. --velocity -20 20.",
    )
    infer.add_argument(
        "--method", choices=["cnn", "fft-band", "fft-powerlaw"], default="cnn",
        help="Prediction method: 'cnn' (default; requires --model) predicts via a trained ensemble. "
        "'fft-band' estimates k_min/k_max from a sharp spectral band edge; only appropriate for synthetic, "
        "band-limited pyFC images. 'fft-powerlaw' fits a power-law inertial range instead and reads off the "
        "injection/dissipation scales; appropriate for real observational clouds, which have a continuous "
        "spectrum rather than a sharp cutoff. Neither fft-* method needs --model.",
    )
    infer.add_argument(
        "--model", default=None,
        help="Path to a '.../ensemble/members' directory (not the 'ensemble' or version directory "
        "above it), or 'auto' to resolve one automatically from --output-root (newest version, by "
        "modification time, of the multitask ensemble unless --target is also given).",
    )
    infer.add_argument(
        "--target", default=None, choices=["k_min", "k_max", "sigma", "beta"],
        help="--model auto only. Restrict to one target's dedicated single-target ensemble instead of "
        "the multitask ensemble. Omit for multitask (predicts all four targets in one pass).",
    )
    infer.add_argument(
        "--real-data", action="store_true",
        help="--method cnn only. Treat --input as a real (non-synthetic) observation: log10-transform, "
        "then recenter onto the training set's own fixed-width normalization range (not the image's own "
        "percentiles), since real intensities sit on a completely different linear scale than the "
        "log-density training images. Strongly recommended for any real FITS/telescope input -- omitting "
        "it on real data silently produces confident-looking but meaningless predictions. See "
        "turbulens.models.checkpoint.real_observation_normalization.",
    )
    infer.add_argument("--output-root", default=None, help="Pipeline output root; required when --model auto.")
    infer.add_argument("--out", default=None, help="Output path (.ecsv recommended). Prints to stdout if omitted.")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Show a breakdown of a FITS cube or already-projected image."
    )
    inspect_parser.add_argument("--input", default=None, help="Path to a FITS cube, or a PNG/TIFF/.npy image.")
    inspect_parser.add_argument(
        "--hdu", type=int, default=None, help="FITS HDU index (FITS input only). Default 0."
    )
    inspect_parser.add_argument("--json", default=None, help="Write the breakdown as JSON to this path.")
    inspect_parser.add_argument(
        "--html", default=None, help="Write the breakdown as a minimal HTML report to this path."
    )
    return parser


_INFER_CONFIG_DEFAULTS = {
    "input": None, "hdu": None, "channel": None, "integrate": None, "velocity": None,
    "method": "cnn", "model": None, "target": None, "real_data": False,
    "output_root": None, "out": None,
}
"""dict: `infer` option names loadable from `--config`, mapped to their argparse defaults.

An option is only filled in from the JSON config if it is still at this
default, i.e. was not explicitly passed as a flag; see `_apply_config_file`.
"""


def _apply_config_file(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill in `infer` options from `--config`'s JSON file, without overriding explicit flags."""
    if args.config is None:
        return
    try:
        payload = json.loads(Path(args.config).read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"could not read --config {args.config!r}: {exc}")
    if not isinstance(payload, dict):
        parser.error(f"--config {args.config!r} must contain a JSON object, got {type(payload).__name__}.")
    for key, value in payload.items():
        if key not in _INFER_CONFIG_DEFAULTS:
            parser.error(f"--config {args.config!r}: unknown option {key!r}.")
        if getattr(args, key) == _INFER_CONFIG_DEFAULTS[key]:
            setattr(args, key, value)


def _resolve_members_dir(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    """Resolve `--model` to a concrete ensemble members directory, handling `--model auto`."""
    if args.model != "auto":
        if args.target is not None:
            parser.error("--target requires --model auto; it has no effect with an explicit --model path.")
        return Path(args.model)
    if not args.output_root:
        parser.error("--output-root is required when --model auto.")
    return EnsembleRegistry.resolve(args.output_root, target=args.target)


def _validate_input_selection(args: argparse.Namespace, parser: argparse.ArgumentParser, cube: Cube) -> None:
    """Validate that exactly one channel-selection flag is given, or none, for a single-channel input."""
    selected = [
        name
        for name, value in (("--channel", args.channel), ("--integrate", args.integrate), ("--velocity", args.velocity))
        if value is not None
    ]
    if cube.n_channels == 1:
        if selected:
            parser.error(f"{', '.join(selected)} given, but the input has only one channel; nothing to select.")
        return
    if not selected:
        parser.error("One of --channel, --integrate, or --velocity is required for multi-channel input.")
    if len(selected) > 1:
        parser.error(f"--channel, --integrate, and --velocity are mutually exclusive; got {', '.join(selected)}.")


def run_infer(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Run the ``infer`` subcommand: load input, select a channel, predict, write or print results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from the ``infer`` subparser built by
        `build_parser`.
    parser : argparse.ArgumentParser
        The top-level parser, used to call ``parser.error(...)`` for
        user-facing validation failures (which exits with a usage
        message rather than raising a traceback).

    Notes
    -----
    ``--method fft-band``/``fft-powerlaw`` estimate k_min/k_max
    classically from the image's Fourier power spectrum
    (`turbulens.spectral.SpectralEstimator`) and do not need
    ``--model``; ``--method cnn`` (the default) predicts via a trained
    `turbulens.inference.Inferencer` ensemble and requires ``--model``.

    Exactly one of ``--channel``, ``--integrate``, or ``--velocity`` is
    required when the input has more than one channel, and none of them
    is allowed when the input already has exactly one channel (i.e. is
    already a projected image); see `_validate_input_selection`.

    ``--config`` (see `_apply_config_file`) is applied first, filling in
    any option not explicitly given as a flag, so every check below sees
    the same merged `args` regardless of source.
    """
    _apply_config_file(args, parser)
    if args.input is None:
        parser.error("--input is required.")
    if args.method in ("fft-band", "fft-powerlaw"):
        ignored = [
            name for name, value in (
                ("--model", args.model), ("--target", args.target), ("--real-data", args.real_data or None),
            )
            if value is not None
        ]
        if ignored:
            parser.error(
                f"{', '.join(ignored)} has no effect with --method {args.method} (no model is used); "
                "drop it or switch to --method cnn."
            )
    hdu = args.hdu if args.hdu is not None else 0
    try:
        cube = load_input(args.input, hdu=hdu)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    _validate_input_selection(args, parser, cube)

    if cube.n_channels == 1:
        image_cube = cube
    elif args.channel is not None:
        image_cube = cube.slice_channel(args.channel)
    elif args.integrate is not None:
        parts = args.integrate.split(":")
        if len(parts) != 2:
            parser.error(f"--integrate must be LOW:HIGH, got {args.integrate!r}.")
        low_str, high_str = parts
        try:
            low, high = int(low_str), int(high_str)
        except ValueError:
            parser.error(f"--integrate must be LOW:HIGH with integer bounds, got {args.integrate!r}.")
        try:
            image_cube = cube.project(low, high)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        v_low, v_high = args.velocity
        try:
            image_cube = cube.project_velocity(v_low * u.km / u.s, v_high * u.km / u.s)
        except ValueError as exc:
            parser.error(str(exc))

    if args.method in ("fft-band", "fft-powerlaw"):
        spectral_method = "band_edge" if args.method == "fft-band" else "power_law"
        table = SpectralEstimator(method=spectral_method).predict(image_cube.data)
        table.meta.update({
            "input_path": args.input,
            "method": args.method,
            "channel": args.channel,
            "integrate": args.integrate,
            "velocity": list(args.velocity) if args.velocity is not None else None,
            "turbulens_version": __version__,
        })
    else:
        if args.model is None:
            parser.error("--model is required.")
        try:
            members_dir = _resolve_members_dir(args, parser)
            table = Inferencer(members_dir).predict(image_cube.data, real_data=args.real_data)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        table.meta.update({
            "input_path": args.input,
            "method": args.method,
            "model": str(members_dir),
            "real_data": args.real_data,
            "channel": args.channel,
            "integrate": args.integrate,
            "velocity": list(args.velocity) if args.velocity is not None else None,
            "turbulens_version": __version__,
        })

    if args.out:
        write_results(table, args.out)
    else:
        table.pprint(max_width=-1)


def run_inspect(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Run the ``inspect`` subcommand: load input, describe it, print and optionally write reports.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from the ``inspect`` subparser built by
        `build_parser`.
    parser : argparse.ArgumentParser
        The top-level parser, used for `parser.error` on validation
        failures.
    """
    if args.input is None:
        parser.error("--input is required.")
    hdu = args.hdu if args.hdu is not None else 0

    try:
        cube = load_input(args.input, hdu=hdu)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    breakdown = describe_cube(cube, source_path=str(args.input))
    print(format_breakdown_cli(breakdown))

    if args.json:
        write_breakdown_json(breakdown, args.json)
    if args.html:
        write_breakdown_html(breakdown, args.html)


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the `turbulens` console script.

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse, in place of `sys.argv`. Primarily for
        testing; default is `None`, which makes `argparse` read
        `sys.argv`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "infer":
        run_infer(args, parser)
    elif args.command == "inspect":
        run_inspect(args, parser)


if __name__ == "__main__":
    main()
