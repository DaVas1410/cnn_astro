"""The CNN regression architecture: a ResNet backbone with per-target heads.

The network predicts several continuous physical targets (e.g. `k_min`,
`k_max`, `sigma`, `beta` from `archive/src/pyFC_lib`'s log-normal fractal cube
model) from a single 2D image, sharing one backbone and one shared
trunk across all targets, then branching into a small independent head
per target. This joint-multi-head design is what the repository's
`archive/docs/RESEARCH_JOURNEY.md` (phases 5-6) calls "joint regression".

The `local_v2` multitask-vs-single-target comparison
(`outputs/comparison/local_v2/comparison_report.md`) found the opposite
of the earlier working assumption: a dedicated single-target model is a
statistically significant (paired test, Holm-Bonferroni adjusted) but
practically small improvement over the shared model on `k_min`,
`k_max`, and `sigma`, while the shared model remains clearly better on
`beta`. The single-target system also costs ~5x the total training time
and ~4x the parameters of one shared multitask model, since it trains
one full network per target instead of one. Joint regression is kept as
the default here for that cost/accuracy tradeoff and because it is
still the best approach for `beta`, not because it dominates on every
target.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models


def build_resnet_backbone(architecture: str, pretrained: bool) -> nn.Module:
    """
    Construct a torchvision ResNet backbone by name.

    Parameters
    ----------
    architecture : {"resnet18", "resnet34", "resnet50"}
        Which ResNet variant to build.
    pretrained : bool
        Whether to load ImageNet-pretrained weights.

    Returns
    -------
    torch.nn.Module
        The backbone, with its original ``fc`` classification head still
        attached (`ResNetPhysicalRegressor` replaces it).

    Raises
    ------
    ValueError
        If ``architecture`` is not one of the supported names.

    References
    ----------
    `torchvision.models`
        https://pytorch.org/vision/stable/models.html
    """
    if architecture == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if architecture == "resnet34":
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    if architecture == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    raise ValueError(f"Unsupported architecture: {architecture}")


class ResNetPhysicalRegressor(nn.Module):
    """
    A ResNet backbone with a shared trunk and one regression head per target.

    Parameters
    ----------
    architecture : {"resnet18", "resnet34", "resnet50"}
        Backbone variant, passed to `build_resnet_backbone`.
    in_channels : {1, 3}
        Number of input image channels. If ``1``, the backbone's first
        convolution is replaced with a single-channel version (see
        Notes); ``3`` uses the backbone unmodified.
    pretrained : bool
        Whether to initialize the backbone from ImageNet-pretrained
        weights.
    dropout : float
        Base dropout probability for the shared trunk; the second trunk
        layer and each per-target head use scaled-down values derived
        from this (see Notes).
    preserve_resolution : bool
        If `True`, replaces the backbone's ``maxpool`` layer with the
        identity, so spatial resolution is preserved further into the
        network. Useful for smaller input images where ResNet's default
        aggressive early downsampling would discard too much spatial
        information.
    targets : sequence of str
        Names of the physical targets to predict (e.g. ``["k_min",
        "k_max", "sigma"]``), one output head per target. Must be
        non-empty and contain no duplicates.
    shared_dimensions : sequence of int, optional
        Hidden dimensions of the shared trunk. Default is ``(512,
        256)``, which is currently also the only accepted value (see
        Raises); a fixed shared trunk shape is what allows a
        checkpoint's `model_config` to be compared/reloaded reliably.
    head_hidden_dimension : int, optional
        Hidden dimension of each per-target head. Default is ``96``.

    Raises
    ------
    ValueError
        If ``targets`` is empty, contains duplicate names, if
        ``shared_dimensions`` is not exactly ``[512, 256]``, or if
        ``in_channels`` is not ``1`` or ``3``.

    Notes
    -----
    When ``in_channels=1``, the backbone's first convolution (normally 3
    input channels, for RGB) is replaced with a 1-channel convolution.
    If ``pretrained`` is also `True`, the new convolution's weights are
    initialized as the channel-mean of the pretrained 3-channel weights,
    which approximately preserves the filters' learned response to
    intensity rather than discarding the pretrained initialization
    entirely; otherwise the new convolution is Kaiming-initialized.

    Dropout is scaled across the network rather than using one fixed
    rate everywhere: the shared trunk's second layer uses
    ``min(0.60, dropout + 0.10)`` (more dropout deeper in a wider shared
    representation), and each head uses ``min(0.35, dropout * 0.50)``
    (less dropout in the small, target-specific heads, which have fewer
    parameters to overfit with).
    """

    def __init__(
        self,
        *,
        architecture: str,
        in_channels: int,
        pretrained: bool,
        dropout: float,
        preserve_resolution: bool,
        targets: Sequence[str],
        shared_dimensions: Sequence[int] = (512, 256),
        head_hidden_dimension: int = 96,
    ) -> None:
        super().__init__()
        if not targets:
            raise ValueError("At least one target is required.")
        if len(set(targets)) != len(targets):
            raise ValueError("Target names must be unique.")
        if list(shared_dimensions) != [512, 256]:
            raise ValueError(
                "The current compatibility implementation requires shared_dimensions=[512,256]."
            )
        self.architecture = str(architecture)
        self.in_channels = int(in_channels)
        self.pretrained = bool(pretrained)
        self.dropout = float(dropout)
        self.preserve_resolution = bool(preserve_resolution)
        self.targets = tuple(str(target) for target in targets)
        self.n_outputs = len(self.targets)
        self.shared_dimensions = tuple(int(x) for x in shared_dimensions)
        self.head_hidden_dimension = int(head_hidden_dimension)

        backbone = build_resnet_backbone(self.architecture, self.pretrained)
        if self.in_channels == 1:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if self.pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = new_conv
        elif self.in_channels != 3:
            raise ValueError("in_channels must be 1 or 3.")

        if self.preserve_resolution:
            backbone.maxpool = nn.Identity()

        feature_count = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.backbone_feature_count = feature_count

        second_dropout = min(0.60, self.dropout + 0.10)
        head_dropout = min(0.35, self.dropout * 0.50)
        self.shared = nn.Sequential(
            nn.Linear(feature_count, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(second_dropout),
        )

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(256, self.head_hidden_dimension),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(self.head_hidden_dimension, 1),
            )

        self.heads = nn.ModuleDict({target: make_head() for target in self.targets})

    def extract_embedding(self, x: torch.Tensor, layer: str = "shared") -> torch.Tensor:
        """
        Run the backbone (and optionally the shared trunk) and return the embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, in_channels, H, W)``.
        layer : {"backbone", "shared"}, optional
            Which embedding to return: the raw backbone feature vector,
            or the shared trunk's output (what the per-target heads
            consume). Default is ``"shared"``.

        Returns
        -------
        torch.Tensor
            The requested embedding, of shape ``(B,
            backbone_feature_count)`` for ``"backbone"`` or ``(B, 256)``
            for ``"shared"``.

        Raises
        ------
        ValueError
            If ``x`` is not a 4D tensor with `in_channels` channels, or
            if ``layer`` is not ``"backbone"`` or ``"shared"``.

        Notes
        -----
        Exposed as a separate method (rather than only reachable via
        `forward`) so that interpretability analyses (PCA/CCA/UMAP over
        embeddings, per `archive/docs/RESEARCH_JOURNEY.md` phase 12) can extract
        either representation without running the per-target heads.
        """
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input (B,{self.in_channels},H,W), received {tuple(x.shape)}."
            )
        backbone_features = self.backbone(x)
        if layer == "backbone":
            return backbone_features
        shared_features = self.shared(backbone_features)
        if layer == "shared":
            return shared_features
        raise ValueError("embedding layer must be 'backbone' or 'shared'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict all targets for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Raw (not denormalized) predictions of shape ``(B,
            n_outputs)``, in the same order as `targets`. Values are in
            the model's training-time normalized range, typically
            ``[0, 1]``; see
            `turbulens.models.checkpoint.denormalize_targets` for
            converting to physical units.

        Raises
        ------
        RuntimeError
            If the concatenated head outputs do not have the expected
            shape, which would indicate an internal inconsistency
            between `targets` and the constructed `heads`.
        """
        features = self.extract_embedding(x, layer="shared")
        outputs = torch.cat([self.heads[target](features) for target in self.targets], dim=1)
        if outputs.shape != (x.shape[0], self.n_outputs):
            raise RuntimeError(f"Unexpected output shape {tuple(outputs.shape)}.")
        return outputs

    def model_config(self) -> dict[str, object]:
        """
        Return this model's construction arguments and derived sizes as a plain dict.

        Returns
        -------
        dict
            Every constructor argument plus `n_outputs` and
            `backbone_feature_count`. Round-trips through
            `build_model_from_config` (via the ``"model"`` key of a
            run config), and is what a checkpoint's
            ``configuration.json`` records for `turbulens.models.checkpoint.load_member`
            to rebuild the architecture before loading weights.
        """
        return {
            "architecture": self.architecture,
            "in_channels": self.in_channels,
            "pretrained": self.pretrained,
            "dropout": self.dropout,
            "preserve_resolution": self.preserve_resolution,
            "targets": list(self.targets),
            "n_outputs": self.n_outputs,
            "shared_dimensions": list(self.shared_dimensions),
            "head_hidden_dimension": self.head_hidden_dimension,
            "backbone_feature_count": self.backbone_feature_count,
        }


def build_model_from_config(run_config: dict[str, object]) -> ResNetPhysicalRegressor:
    """
    Construct a `ResNetPhysicalRegressor` from a training-pipeline run config.

    Parameters
    ----------
    run_config : dict
        A run configuration with ``"model"`` and ``"task"`` keys, in the
        same shape as `turbulens.models.checkpoint.load_member` reads
        from a member's ``configuration.json``.

    Returns
    -------
    ResNetPhysicalRegressor
        The constructed (randomly initialized, or ImageNet-pretrained if
        ``model["pretrained"]`` is set) model.

    Notes
    -----
    Only ``model["preserve_resolution"]``, ``model["shared_dimensions"]``,
    and ``model["head_hidden_dimension"]`` are optional in `run_config`,
    each falling back to `ResNetPhysicalRegressor`'s own defaults if
    absent; every other key is required.
    """
    model = run_config["model"]
    task = run_config["task"]
    assert isinstance(model, dict) and isinstance(task, dict)
    return ResNetPhysicalRegressor(
        architecture=str(model["architecture"]),
        in_channels=int(model["in_channels"]),
        pretrained=bool(model["pretrained"]),
        dropout=float(model["dropout"]),
        preserve_resolution=bool(model.get("preserve_resolution", True)),
        targets=list(task["targets"]),
        shared_dimensions=list(model.get("shared_dimensions", [512, 256])),
        head_hidden_dimension=int(model.get("head_hidden_dimension", 96)),
    )


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """
    Count a model's total and trainable parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Any PyTorch module.

    Returns
    -------
    dict
        ``{"total": ..., "trainable": ...}``, both `int`.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
