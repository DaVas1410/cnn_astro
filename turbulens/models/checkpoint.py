"""Loading trained ensemble-member checkpoints and the pixel/target normalization contract.

An ensemble member on disk is a directory with a ``configuration.json``
(architecture, task, and normalization config) and a
``checkpoints/best_checkpoint.pt`` (model weights plus the exact
normalization statistics computed at training time). `load_member` reads
both into a `Member`, and `preprocess_image` / `denormalize_targets`
replicate the training pipeline's input preprocessing and output
denormalization exactly, since a mismatch here would silently produce
wrong predictions rather than an error.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from turbulens.models.architecture import ResNetPhysicalRegressor, build_model_from_config

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
"""numpy.ndarray: Per-channel ImageNet pixel mean (RGB order), used by ``input_standardization="imagenet"``."""

IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
"""numpy.ndarray: Per-channel ImageNet pixel standard deviation (RGB order)."""


@dataclass(frozen=True)
class NormalizationStats:
    """
    Per-member pixel normalization statistics, computed once at training time.

    These define an affine map from raw pixel intensity to the ``[0,
    1]`` range the model was trained on: ``(pixel - lower_value) /
    value_range``, clipped to ``[0, 1]``. Values outside
    ``[lower_percentile, upper_percentile]`` of the training
    distribution get clipped, by design; `preprocess_image` reports
    what fraction of a new image's pixels hit that clip boundary, as a
    distribution-shift signal.

    Parameters
    ----------
    lower_percentile : float
        Lower percentile (e.g. ``1.0``) used to compute `lower_value`
        from the training set's pixel distribution.
    upper_percentile : float
        Upper percentile used to compute `upper_value`.
    lower_value : float
        Pixel intensity at `lower_percentile`; maps to normalized ``0``.
    upper_value : float
        Pixel intensity at `upper_percentile`; maps to normalized ``1``.
    value_range : float
        ``upper_value - lower_value``.
    finite_pixels_examined : int
        Number of finite pixels sampled to compute these statistics.
    images_examined : int
        Number of training images sampled.
    sampling_seed : int
        Random seed used for sampling, for reproducibility.
    source : str
        Free-text description of which dataset/split these statistics
        were computed from.
    """

    lower_percentile: float
    upper_percentile: float
    lower_value: float
    upper_value: float
    value_range: float
    finite_pixels_examined: int
    images_examined: int
    sampling_seed: int
    source: str


@dataclass
class Member:
    """
    One loaded ensemble member: its model, and everything needed to run and interpret it.

    Parameters
    ----------
    model : ResNetPhysicalRegressor
        The loaded, weight-populated, eval-mode model.
    normalization : NormalizationStats
        This member's pixel normalization statistics.
    target_ranges : dict of str to tuple of float
        ``(low, high)`` physical-unit bounds per target, used by
        `denormalize_targets`.
    input_standardization : str
        Which input standardization `preprocess_image` should apply
        (``"imagenet"`` or ``"none"``).
    """

    model: ResNetPhysicalRegressor
    normalization: NormalizationStats
    target_ranges: dict[str, tuple[float, float]]
    input_standardization: str


def load_member(member_dir: Path | str) -> Member:
    """
    Load one ensemble member's model, weights, and normalization contract from disk.

    Parameters
    ----------
    member_dir : pathlib.Path or str
        Directory containing ``configuration.json`` and
        ``checkpoints/best_checkpoint.pt``, as produced by the training
        pipeline (see `turbulens.models.registry.EnsembleRegistry`).

    Returns
    -------
    Member
        The loaded member, with ``model.eval()`` already called.

    Notes
    -----
    The model is built with ``pretrained=False`` regardless of what
    ``configuration.json`` says, since the immediately following
    ``load_state_dict(strict=True)`` overwrites every weight anyway;
    building with ``pretrained=True`` here would only trigger a
    pointless ImageNet weight download.

    Checkpoints are loaded with ``weights_only=False``, since the
    checkpoint dict contains plain Python objects (the normalization
    statistics), not only tensors.

    References
    ----------
    `torch.load`
        https://pytorch.org/docs/stable/generated/torch.load.html
    """
    member_dir = Path(member_dir)
    config = json.loads((member_dir / "configuration.json").read_text())
    checkpoint = torch.load(
        member_dir / "checkpoints" / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    # Build with pretrained=False: load_state_dict(strict=True) below immediately
    # overwrites every weight regardless of the initial pretrained value, so building
    # with pretrained=True here would only trigger a pointless ImageNet weight download.
    build_config = {**config, "model": {**config["model"], "pretrained": False}}
    model = build_model_from_config(build_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    normalization = NormalizationStats(**checkpoint["normalization_stats"])
    target_ranges = {
        target: (float(bounds[0]), float(bounds[1]))
        for target, bounds in config["task"]["target_ranges"].items()
    }
    input_standardization = str(config["model"].get("input_standardization", "imagenet"))
    return Member(
        model=model,
        normalization=normalization,
        target_ranges=target_ranges,
        input_standardization=input_standardization,
    )


def preprocess_image(
    image: np.ndarray,
    normalization: NormalizationStats,
    in_channels: int,
    input_standardization: str = "imagenet",
) -> tuple[np.ndarray, float]:
    """
    Replicate the training pipeline's input preprocessing for one image.

    Parameters
    ----------
    image : numpy.ndarray
        A single 2D ``(H, W)`` image.
    normalization : NormalizationStats
        The member's pixel normalization statistics to apply.
    in_channels : {1, 3}
        Number of channels the model expects; the single input channel
        is replicated to 3 if needed.
    input_standardization : {"imagenet", "none"}, optional
        Whether to apply ImageNet mean/std standardization after
        scaling and clipping. Default is ``"imagenet"``.

    Returns
    -------
    stacked : numpy.ndarray
        The preprocessed image, of shape ``(in_channels, H, W)``, ready
        to stack into a model input batch.
    clipped_fraction : float
        Fraction of pixels that hit the ``[0, 1]`` clip boundary after
        scaling, as a distribution-shift signal (see
        `turbulens.inference.Inferencer.predict`).

    Raises
    ------
    ValueError
        If ``image`` is not 2D, if ``in_channels`` is not ``1`` or
        ``3``, or if ``input_standardization`` is not ``"imagenet"`` or
        ``"none"``.

    Notes
    -----
    Non-finite pixels are replaced with `NormalizationStats.lower_value`
    before scaling, so they map to the normalized value ``0.0`` rather
    than propagating `nan` into the model.

    Channel replication happens after scaling and clipping. This
    ordering is only equivalent to the training pipeline's
    stored-image-average-then-scale order because this function's input
    is always a single-channel 2D image, never a pre-stored
    multi-channel image; the two orders are not interchangeable in
    general, since averaging commutes with an affine scale/clip only
    when every channel is identical to begin with (which is exactly the
    case here, since the replication produces identical channels).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    finite = np.isfinite(image)
    cleaned = np.where(finite, image, normalization.lower_value).astype(np.float32)
    scaled = (cleaned - normalization.lower_value) / normalization.value_range
    clipped_fraction = float(np.mean((scaled <= 0.0) | (scaled >= 1.0)))
    clipped = np.clip(scaled, 0.0, 1.0)

    # Channel replication happens after scale/clip; this is only equivalent to
    # pipeline_lib's stored-image-average-then-scale order because this
    # function's input is always a single-channel 2D image, never a
    # pre-stored multi-channel image.
    if in_channels == 1:
        stacked = clipped[np.newaxis, :, :]
    elif in_channels == 3:
        stacked = np.repeat(clipped[np.newaxis, :, :], 3, axis=0)
    else:
        raise ValueError(f"Unsupported in_channels: {in_channels}.")

    if input_standardization == "imagenet":
        if in_channels == 3:
            mean, std = IMAGENET_MEAN[:, None, None], IMAGENET_STD[:, None, None]
        else:
            mean = np.asarray([IMAGENET_MEAN.mean()], dtype=np.float32)[:, None, None]
            std = np.asarray([IMAGENET_STD.mean()], dtype=np.float32)[:, None, None]
        stacked = (stacked - mean) / std
    elif input_standardization != "none":
        raise ValueError(f"Unknown input standardization: {input_standardization!r}")

    return stacked.astype(np.float32), clipped_fraction


def real_observation_normalization(
    image: np.ndarray,
    member_normalization: NormalizationStats,
) -> tuple[np.ndarray, NormalizationStats]:
    """
    Normalize a real observation to unit mean and log10-transform it, on a member's training scale.

    Training images are ``log10(density)`` of synthetic 2D log-normal
    fractal fields generated with pyFC, then percentile-normalized
    using stats computed once across the whole training set. A real
    observation (e.g. a GASS HI brightness-temperature map) is neither
    log-transformed nor on that same absolute scale (different surveys and
    instruments have arbitrary, mutually incompatible calibrations), so
    reusing `member_normalization` on raw real pixels clips essentially every
    pixel (see `turbulens.inference.Inferencer.predict`'s out-of-distribution
    warning).

    The generator fixes the mean density of every training field at 1, so
    each training image is ``log10(rho)`` with ``<rho> = 1``. This function
    puts a real map on the same footing: it divides the image by its own
    mean and takes ``log10``. That removes the map's arbitrary units and
    calibration while keeping both properties of a training image the model
    reads `sigma` from: the spread of the log values (the field's contrast)
    and their absolute level (for a log-normal field with unit mean,
    ``<ln rho> = -sigma_s**2 / 2``, so the level itself depends on `sigma`).
    The image's spread is not rescaled, and `member_normalization`'s fixed
    ``lower_value``/``upper_value`` are reused unchanged.

    Parameters
    ----------
    image : numpy.ndarray
        A single 2D ``(H, W)`` real-observation image, in its native
        (linear, positive-valued) physical units.
    member_normalization : NormalizationStats
        The ensemble member's own training-time normalization stats
        (`turbulens.models.checkpoint.Member.normalization`). Its
        ``lower_value``/``upper_value``/``value_range`` are reused
        unchanged.

    Returns
    -------
    normalized_image : numpy.ndarray
        ``log10(image / mean(image))``, ready to pass to `preprocess_image`
        together with `stats`.
    stats : NormalizationStats
        `member_normalization` with `source` and `finite_pixels_examined`
        updated to describe this normalization; `lower_value`/`upper_value`/
        `value_range` are identical to `member_normalization`'s.

    Notes
    -----
    Non-positive pixels (instrumental noise dipping below zero, absorption,
    or an all-zero border) have no finite log10 and are floored to the
    image's own smallest positive value before the mean and the log10 are
    taken, rather than to an arbitrary global constant, since real-image
    intensity scales vary by orders of magnitude across surveys and targets.

    The result is invariant to the map's units: multiplying the input by
    any positive constant leaves it unchanged.
    """
    image = np.asarray(image, dtype=np.float64)
    positive = image[image > 0]
    floor = float(positive.min()) if positive.size else float(np.finfo(np.float32).tiny)
    clean = np.where(image > 0, image, floor)
    normalized_image = np.log10(clean / clean.mean()).astype(np.float32)

    stats = NormalizationStats(
        lower_percentile=member_normalization.lower_percentile,
        upper_percentile=member_normalization.upper_percentile,
        lower_value=member_normalization.lower_value,
        upper_value=member_normalization.upper_value,
        value_range=member_normalization.value_range,
        finite_pixels_examined=int(np.isfinite(normalized_image).sum()),
        images_examined=1,
        sampling_seed=-1,
        source="real_observation_unit_mean_fixed_training_scale",
    )
    return normalized_image, stats


def denormalize_targets(
    values: np.ndarray,
    targets: Sequence[str],
    target_ranges: Mapping[str, tuple[float, float]],
    *,
    clamp: bool = True,
) -> np.ndarray:
    """
    Convert raw ``[0, 1]``-range model outputs to physical units.

    Parameters
    ----------
    values : numpy.ndarray
        Raw model outputs of shape ``(N, len(targets))``.
    targets : sequence of str
        Target names, in the same column order as `values`.
    target_ranges : mapping of str to tuple of float
        ``(low, high)`` physical-unit bounds per target.
    clamp : bool, optional
        Whether to clip `values` to ``[0, 1]`` before the affine map.
        Default is `True`, since the model's raw output is not
        guaranteed to stay within ``[0, 1]`` (no output activation
        enforces it), and an unclamped value would extrapolate outside
        the training-time target range.

    Returns
    -------
    numpy.ndarray
        Denormalized values of the same shape as `values`, in physical
        units per target.

    Raises
    ------
    ValueError
        If ``values`` is not 2D with `len(targets)` columns.

    Notes
    -----
    This must be applied per ensemble member, before averaging across
    members: see `turbulens.inference.Inferencer.predict` for why
    averaging in physical units (rather than averaging raw outputs and
    denormalizing once) is required when members can have different
    `target_ranges`.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(targets):
        raise ValueError(f"Expected predictions shape (N,{len(targets)}), got {values.shape}.")
    work = np.clip(values, 0.0, 1.0) if clamp else values
    output = np.empty_like(work, dtype=np.float32)
    for col, target in enumerate(targets):
        low, high = target_ranges[target]
        output[:, col] = work[:, col] * (high - low) + low
    return output
