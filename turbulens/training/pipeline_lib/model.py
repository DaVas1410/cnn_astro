from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torchvision import models


def resolve_resnet_weights(architecture: str, pretrained: bool, weight_name: str | None):
    """Resolve an explicitly named torchvision weight enum.

    `weight_name` must be present for pretrained models and absent for scratch
    models. This avoids the version-dependent `Weights.DEFAULT` alias.
    """
    if not pretrained:
        if weight_name not in {None, "", "none"}:
            raise ValueError("pretrained_weight must be null for scratch models.")
        return None
    if not weight_name:
        raise ValueError("A pretrained_weight name is required for pretrained models.")
    enum_by_architecture = {
        "resnet18": models.ResNet18_Weights,
        "resnet34": models.ResNet34_Weights,
        "resnet50": models.ResNet50_Weights,
    }
    if architecture not in enum_by_architecture:
        raise ValueError(f"Unsupported architecture: {architecture}")
    enum_class = enum_by_architecture[architecture]
    try:
        return enum_class[str(weight_name)]
    except KeyError as exc:
        supported = [member.name for member in enum_class]
        raise ValueError(
            f"Unsupported pretrained weight {weight_name!r} for {architecture}; "
            f"supported={supported}."
        ) from exc


def build_resnet_backbone(architecture: str, pretrained: bool, weight_name: str | None) -> nn.Module:
    weights = resolve_resnet_weights(architecture, pretrained, weight_name)
    if architecture == "resnet18":
        return models.resnet18(weights=weights)
    if architecture == "resnet34":
        return models.resnet34(weights=weights)
    if architecture == "resnet50":
        return models.resnet50(weights=weights)
    raise ValueError(f"Unsupported architecture: {architecture}")


class ResNetPhysicalRegressor(nn.Module):
    """ResNet image encoder with one shared latent representation and target-specific heads.

    The shared MLP dimensions and target-head width are read from configuration.
    With the repository defaults ([512, 256] and 96), this is numerically identical
    in architecture to the original multitask model while avoiding hidden fixed widths.
    """

    def __init__(
        self,
        *,
        architecture: str,
        in_channels: int,
        pretrained: bool,
        pretrained_weight: str | None,
        dropout: float,
        preserve_resolution: bool,
        targets: Sequence[str],
        shared_dimensions: Sequence[int],
        head_hidden_dimension: int,
        dropout_policy: Mapping[str, float],
    ) -> None:
        super().__init__()
        if not targets:
            raise ValueError("At least one target is required.")
        if len(set(targets)) != len(targets):
            raise ValueError("Target names must be unique.")
        shared_dimensions = tuple(int(x) for x in shared_dimensions)
        if not shared_dimensions or any(width < 1 for width in shared_dimensions):
            raise ValueError("shared_dimensions must contain one or more positive integers.")
        self.architecture = str(architecture)
        self.in_channels = int(in_channels)
        self.pretrained = bool(pretrained)
        self.pretrained_weight = str(pretrained_weight) if pretrained_weight is not None else None
        self.dropout = float(dropout)
        self.preserve_resolution = bool(preserve_resolution)
        self.targets = tuple(str(target) for target in targets)
        self.n_outputs = len(self.targets)
        self.shared_dimensions = shared_dimensions
        self.head_hidden_dimension = int(head_hidden_dimension)
        self.dropout_policy = {str(key): float(value) for key, value in dropout_policy.items()}
        required_dropout_keys = {
            "shared_increment_per_layer", "shared_max", "head_multiplier", "head_max"
        }
        if set(self.dropout_policy) != required_dropout_keys:
            raise ValueError(
                "dropout_policy keys must be exactly "
                f"{sorted(required_dropout_keys)}; got {sorted(self.dropout_policy)}"
            )

        backbone = build_resnet_backbone(
            self.architecture, self.pretrained, self.pretrained_weight
        )
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

        head_dropout = min(
            self.dropout_policy["head_max"],
            self.dropout * self.dropout_policy["head_multiplier"],
        )
        shared_layers: list[nn.Module] = []
        previous_width = feature_count
        for index, width in enumerate(self.shared_dimensions):
            layer_dropout = min(
                self.dropout_policy["shared_max"],
                self.dropout + self.dropout_policy["shared_increment_per_layer"] * index,
            )
            shared_layers.extend((
                nn.Linear(previous_width, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(layer_dropout),
            ))
            previous_width = width
        self.shared = nn.Sequential(*shared_layers)
        shared_output_dimension = self.shared_dimensions[-1]

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(shared_output_dimension, self.head_hidden_dimension),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(self.head_hidden_dimension, 1),
            )

        self.heads = nn.ModuleDict({target: make_head() for target in self.targets})

    def extract_embedding(self, x: torch.Tensor, layer: str = "shared") -> torch.Tensor:
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
        features = self.extract_embedding(x, layer="shared")
        outputs = torch.cat([self.heads[target](features) for target in self.targets], dim=1)
        if outputs.shape != (x.shape[0], self.n_outputs):
            raise RuntimeError(f"Unexpected output shape {tuple(outputs.shape)}.")
        return outputs

    def model_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "in_channels": self.in_channels,
            "pretrained": self.pretrained,
            "pretrained_weight": self.pretrained_weight,
            "dropout": self.dropout,
            "preserve_resolution": self.preserve_resolution,
            "targets": list(self.targets),
            "n_outputs": self.n_outputs,
            "shared_dimensions": list(self.shared_dimensions),
            "head_hidden_dimension": self.head_hidden_dimension,
            "dropout_policy": dict(self.dropout_policy),
            "backbone_feature_count": self.backbone_feature_count,
        }


def build_model_from_config(run_config: dict[str, object]) -> ResNetPhysicalRegressor:
    model = run_config["model"]
    task = run_config["task"]
    if not isinstance(model, dict) or not isinstance(task, dict):
        raise TypeError("run_config.model and run_config.task must be dictionaries.")
    return ResNetPhysicalRegressor(
        architecture=str(model["architecture"]),
        in_channels=int(model["in_channels"]),
        pretrained=bool(model["pretrained"]),
        pretrained_weight=(
            str(model["pretrained_weight"]) if model.get("pretrained_weight") is not None else None
        ),
        dropout=float(model["dropout"]),
        preserve_resolution=bool(model["preserve_resolution"]),
        targets=list(task["targets"]),
        shared_dimensions=list(model["shared_dimensions"]),
        head_hidden_dimension=int(model["head_hidden_dimension"]),
        dropout_policy=dict(model["dropout_policy"]),
    )


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
