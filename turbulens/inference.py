"""Ensemble inference: running every member, denormalizing, and computing epistemic spread."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from astropy.table import MaskedColumn, QTable

from turbulens._warnings import ClippedPixelsWarning
from turbulens.models.checkpoint import (
    denormalize_targets,
    load_member,
    preprocess_image,
    real_observation_normalization,
)
from turbulens.models.registry import EnsembleRegistry

CLIPPED_FRACTION_WARNING_THRESHOLD = 0.05
"""float: Fraction of clipped pixels (see `turbulens.models.checkpoint.preprocess_image`) above which `Inferencer.predict` warns that the input may be out of distribution."""


class Inferencer:
    """
    Load an ensemble and predict physical targets from images.

    Parameters
    ----------
    members_dir : pathlib.Path or str
        Directory of ensemble members, as returned by
        `turbulens.models.registry.EnsembleRegistry.resolve` or
        `turbulens.models.registry.EnsembleRegistry.discover_members`.
    device : str, optional
        The `torch.device` string to run inference on. Default is
        ``"cpu"``.

    Raises
    ------
    ValueError
        If fewer than two members are found (at least two are required
        to compute an epistemic spread), or if members disagree on
        their target names or order.

    Notes
    -----
    Requiring at least two members is a deliberate design constraint,
    not an incidental one: `predict`'s whole purpose is to report both
    a mean prediction and an epistemic (member-to-member) standard
    deviation, which is undefined for a single model.
    """

    def __init__(self, members_dir: Path | str, device: str = "cpu") -> None:
        member_paths = EnsembleRegistry.discover_members(members_dir)
        self._members = [load_member(path) for path in member_paths]
        if len(self._members) < 2:
            raise ValueError("At least two ensemble members are required for epistemic spread.")

        first_targets = self._members[0].model.targets
        for member in self._members[1:]:
            if member.model.targets != first_targets:
                raise ValueError(
                    "Ensemble members disagree on target names/order: "
                    f"{first_targets} vs {member.model.targets}."
                )
        self._targets = first_targets

        self.device = torch.device(device)
        for member in self._members:
            member.model.to(self.device)

    def clipped_pixels(self, image: np.ndarray, real_data: bool = False) -> np.ndarray:
        """
        Mark the pixels of one image that hit the normalization clip boundary.

        Parameters
        ----------
        image : numpy.ndarray
            A single 2D image of shape ``(H, W)``.
        real_data : bool, optional
            As in `predict`. Default `False`.

        Returns
        -------
        numpy.ndarray
            Boolean ``(H, W)`` mask, `True` where any member's scaled
            pixel lands on or outside ``[0, 1]``, by the same rule as
            `turbulens.models.checkpoint.preprocess_image`. Its mean is at
            least ``meta["clipped_pixel_fraction_max"]`` of `predict`.
        """
        mask = np.zeros(image.shape, dtype=bool)
        for member in self._members:
            pixels, normalization = image, member.normalization
            if real_data:
                pixels, normalization = real_observation_normalization(image, member.normalization)
            cleaned = np.where(np.isfinite(pixels), pixels, normalization.lower_value)
            scaled = (cleaned - normalization.lower_value) / normalization.value_range
            mask |= (scaled <= 0.0) | (scaled >= 1.0)
        return mask

    def predict(self, images: np.ndarray, real_data: bool = False) -> QTable:
        """
        Predict physical targets for one image or a batch of images.

        Every ensemble member is run over the whole batch in one forward
        pass; predictions are denormalized to physical units per member,
        then averaged across members to give the ensemble mean and the
        epistemic standard deviation (member-to-member spread).

        Parameters
        ----------
        images : numpy.ndarray
            A single 2D image of shape ``(H, W)``, or a batch of shape
            ``(N, H, W)``.
        real_data : bool, optional
            Set for a real (non-synthetic) observation, e.g. a GASS HI
            brightness-temperature map. Training images are
            ``log10(density)``, percentile-normalized from the whole
            training set; a real image is on a completely different linear
            scale, so reusing those training-time stats clips essentially
            every pixel. When `True`, each image is instead divided by its
            own mean and log10-transformed, matching the unit-mean training
            fields, and the member's fixed training-time scale is reused
            rather than deriving a new one from the image itself (see
            `turbulens.models.checkpoint.real_observation_normalization`).
            This does not address the deeper mismatch between synthetic
            2D log-normal training fields and real projected observations
            (see the "Interpreting Results" page of the documentation), so
            predictions on real data should still be treated as
            exploratory. Default `False` (use each member's training-time
            normalization, as for synthetic pyFC images).

        Returns
        -------
        astropy.table.QTable
            One row per ``(image, target)`` pair, with columns
            ``image_index``, ``target``, ``value`` (ensemble mean, in
            physical units), and ``epistemic_std`` (ensemble standard
            deviation, ``ddof=1``), both masked (never ``nan``) where not
            finite. ``table.meta`` holds
            ``clipped_pixel_fraction_max`` and ``n_members``.

        Raises
        ------
        ValueError
            If ``images`` is not 2D or 3D.

        Warns
        -----
        UserWarning
            If more than `CLIPPED_FRACTION_WARNING_THRESHOLD` (5%) of pixels,
            across any member, hit the normalization clip boundary, which
            suggests the input is out of the training distribution.

        Notes
        -----
        Denormalization happens per member, in physical units, before
        averaging: each member's raw ``[0, 1]``-range output is denormalized
        against its own `turbulens.models.checkpoint.NormalizationStats`
        before the mean and standard deviation are computed across members.
        Averaging in physical units (rather than averaging raw outputs and
        denormalizing once) matches how the training pipeline's own
        ``evaluate_ensemble.py`` computes ensemble statistics, and is
        required for the mean and epistemic spread to be numerically
        correct when members have different normalization statistics.

        The whole batch is forwarded through each member in one pass, so
        memory scales with ``N``; this is fine for a first vertical slice
        of use cases, but very large ``N`` may need chunking in a future
        version.
        """
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[np.newaxis, :, :]
        if images.ndim != 3:
            raise ValueError(f"Expected a 2D image or a (N,H,W) batch, got shape {images.shape}.")

        n_images = images.shape[0]
        targets = self._targets
        per_member_physical = []
        clipped_fractions = []

        for member in self._members:
            processed = []
            for image in images:
                if real_data:
                    normalized_image, normalization = real_observation_normalization(
                        image, member.normalization,
                    )
                    stacked, clipped_fraction = preprocess_image(
                        normalized_image, normalization, member.model.in_channels, member.input_standardization,
                    )
                else:
                    stacked, clipped_fraction = preprocess_image(
                        image, member.normalization, member.model.in_channels, member.input_standardization,
                    )
                processed.append(stacked)
                clipped_fractions.append(clipped_fraction)
            tensor = torch.from_numpy(np.stack(processed)).to(self.device)
            with torch.no_grad():
                raw_output = member.model(tensor).cpu().numpy()
            physical = denormalize_targets(raw_output, member.model.targets, member.target_ranges, clamp=True)
            per_member_physical.append(physical)

        stacked_predictions = np.stack(per_member_physical, axis=0)
        means = stacked_predictions.mean(axis=0)
        stds = stacked_predictions.std(axis=0, ddof=1)

        image_indices = []
        target_names = []
        values = []
        epistemic_stds = []
        for image_index in range(n_images):
            for target_index, target in enumerate(targets):
                image_indices.append(image_index)
                target_names.append(target)
                values.append(float(means[image_index, target_index]))
                epistemic_stds.append(float(stds[image_index, target_index]))

        table = QTable()
        table["image_index"] = image_indices
        table["target"] = target_names
        # Never store nan: a non-finite member output (not expected after the clamped
        # denormalization, but possible from a degenerate input) is masked instead.
        values, epistemic_stds = np.array(values), np.array(epistemic_stds)
        table["value"] = MaskedColumn(values, mask=~np.isfinite(values))
        table["epistemic_std"] = MaskedColumn(epistemic_stds, mask=~np.isfinite(values) | ~np.isfinite(epistemic_stds))

        clipped_fraction_max = float(max(clipped_fractions)) if clipped_fractions else 0.0
        table.meta["clipped_pixel_fraction_max"] = clipped_fraction_max
        table.meta["n_members"] = len(self._members)
        if clipped_fraction_max > CLIPPED_FRACTION_WARNING_THRESHOLD:
            warnings.warn(
                f"{clipped_fraction_max:.1%} of pixels hit the normalization clip boundary; "
                "input may be out of the training distribution.",
                ClippedPixelsWarning,
                stacklevel=2,
            )
        return table
