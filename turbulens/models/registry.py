"""Resolving ensemble-member directories from the training pipeline's on-disk layout."""

from __future__ import annotations

from pathlib import Path


class EnsembleRegistry:
    """Locate ensemble member directories produced by the training pipeline.

    The training pipeline (`turbulens/training`) writes each
    run under ``<output_root>/<multitask|single_<target>>/<version>/ensemble/members/``,
    with one subdirectory per completed member. This class is the only
    place `turbulens` encodes that on-disk layout.
    """

    @staticmethod
    def discover_members(members_dir: Path | str) -> list[Path]:
        """
        List every completed ensemble member directory under `members_dir`.

        Parameters
        ----------
        members_dir : pathlib.Path or str
            An ``ensemble/members`` directory, as returned by `resolve`
            or passed explicitly via the CLI's ``--model`` argument.

        Returns
        -------
        list of pathlib.Path
            Sorted paths to every member subdirectory containing a
            ``COMPLETED.json`` marker file.

        Raises
        ------
        FileNotFoundError
            If `members_dir` does not exist, or contains no completed
            members.

        Notes
        -----
        A member is only considered complete if it has a
        ``COMPLETED.json`` file, which the training pipeline writes
        atomically as its last step; this avoids picking up a member
        directory that a still-running or crashed training job left in
        a partially written state.
        """
        members_dir = Path(members_dir)
        if not members_dir.is_dir():
            raise FileNotFoundError(f"Ensemble members directory not found: {members_dir}")
        members = [
            path
            for path in sorted(members_dir.iterdir())
            if path.is_dir() and (path / "COMPLETED.json").exists()
        ]
        if not members:
            raise FileNotFoundError(
                f"No completed ensemble members found under {members_dir}. "
                "--model must point at an '.../ensemble/members' directory containing "
                "'member_XX_seed.../COMPLETED.json' subdirectories, not the 'ensemble', "
                "version, or run-mode directory above it."
            )
        return members

    @staticmethod
    def resolve(output_root: Path | str, target: str | None = None, version: str = "latest") -> Path:
        """
        Resolve the ``ensemble/members`` directory for a mode and version.

        Parameters
        ----------
        output_root : pathlib.Path or str
            The training pipeline's output root directory.
        target : str, optional
            If given, resolves a single-target run
            (``single_<target>``); if `None` (the default), resolves
            the joint multitask run (``multitask``).
        version : str, optional
            A specific version directory name, or ``"latest"`` (the
            default) to pick the most recently modified version
            directory.

        Returns
        -------
        pathlib.Path
            The resolved ``<version>/ensemble/members`` directory.

        Raises
        ------
        FileNotFoundError
            If the mode directory, any version directory, or the final
            ``ensemble/members`` directory does not exist.

        Notes
        -----
        "Latest" is determined by filesystem modification time
        (`pathlib.Path.stat().st_mtime`), not by parsing version
        directory names, since the training pipeline's version naming
        is not guaranteed to sort lexicographically or numerically in
        chronological order.
        """
        output_root = Path(output_root)
        mode_dir = output_root / ("multitask" if target is None else f"single_{target}")
        if not mode_dir.is_dir():
            raise FileNotFoundError(f"No ensembles found under {mode_dir}")

        if version == "latest":
            candidates = sorted(
                (path for path in mode_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates:
                raise FileNotFoundError(f"No versions found under {mode_dir}")
            version_dir = candidates[-1]
        else:
            version_dir = mode_dir / version
            if not version_dir.is_dir():
                raise FileNotFoundError(f"Version {version!r} not found under {mode_dir}")

        members_dir = version_dir / "ensemble" / "members"
        if not members_dir.is_dir():
            raise FileNotFoundError(f"No ensemble/members directory under {version_dir}")
        return members_dir
