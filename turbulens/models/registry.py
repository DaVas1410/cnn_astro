"""Resolving ensemble-member directories from the training pipeline's on-disk layout."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

DEFAULT_HUB_REPO = "Davas1410/turbulens-ensembles"
"""str: Hugging Face model repository holding the published pretrained ensembles."""


class FetchedEnsemble(NamedTuple):
    """An ensemble downloaded from the Hugging Face Hub by `EnsembleRegistry.fetch`.

    Attributes
    ----------
    members_dir : pathlib.Path
        The ``ensemble/members`` directory inside the local cache snapshot.
    repo_id : str
        The Hugging Face repository the ensemble came from.
    revision : str
        The exact repository commit hash that was downloaded, for provenance.
    """

    members_dir: Path
    repo_id: str
    revision: str


class EnsembleRegistry:
    """Locate ensemble member directories produced by the training pipeline.

    The training pipeline (`turbulens/training`) writes each
    run under ``<output_root>/<multitask|single_<target>>/<version>/ensemble/members/``,
    with one subdirectory per completed member. This class is the only
    place `turbulens` encodes that on-disk layout. The published
    Hugging Face model repository mirrors the same layout, so `fetch`
    treats a downloaded snapshot as just another output root.
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

    @staticmethod
    def fetch(
        target: str | None = None,
        version: str = "latest",
        repo_id: str = DEFAULT_HUB_REPO,
        revision: str | None = None,
    ) -> FetchedEnsemble:
        """
        Download one pretrained ensemble from the Hugging Face Hub, on demand.

        Parameters
        ----------
        target : str, optional
            If given, fetches that target's single-target ensemble
            (``single_<target>``); if `None` (the default), fetches the
            joint multitask ensemble.
        version : str, optional
            A specific version directory name, or ``"latest"`` (the
            default); see `resolve`.
        repo_id : str, optional
            Hugging Face model repository to download from. Default
            `DEFAULT_HUB_REPO`.
        revision : str, optional
            Branch, tag, or commit hash to download. Default `None`, the
            repository's ``main`` branch.

        Returns
        -------
        FetchedEnsemble
            The local ``ensemble/members`` directory plus the repository
            and exact commit hash it came from.

        Raises
        ------
        FileNotFoundError
            If the repository, revision, or requested ensemble cannot be
            found, or the machine is offline and the ensemble is not
            already cached.

        Notes
        -----
        Only the requested ensemble's files are downloaded (about 450 MB
        each), not the whole repository. Files land in the standard
        Hugging Face cache (``HF_HOME``, by default
        ``~/.cache/huggingface``) and are reused on later calls; with no
        network, or with ``HF_HUB_OFFLINE=1``, an already-cached
        ensemble is served from the cache.
        """
        mode = "multitask" if target is None else f"single_{target}"
        pattern = f"{mode}/*" if version == "latest" else f"{mode}/{version}/*"
        try:
            snapshot = Path(
                snapshot_download(repo_id, repo_type="model", revision=revision, allow_patterns=[pattern])
            )
        except LocalEntryNotFoundError as exc:
            raise FileNotFoundError(
                f"Ensemble {mode!r} from {repo_id} is not in the local cache and could not be "
                "downloaded (offline?). Run 'turbulens download' once on a machine with network "
                f"access to cache it. ({exc})"
            ) from exc
        except RepositoryNotFoundError as exc:
            raise FileNotFoundError(
                f"Hugging Face repository {repo_id!r} not found. If it is private, log in with "
                f"'hf auth login' first. ({exc})"
            ) from exc
        except HfHubHTTPError as exc:
            raise FileNotFoundError(f"Could not download ensemble {mode!r} from {repo_id}: {exc}") from exc
        members_dir = EnsembleRegistry.resolve(snapshot, target=target, version=version)
        return FetchedEnsemble(members_dir=members_dir, repo_id=repo_id, revision=snapshot.name)
