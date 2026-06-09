import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50HeteroJoint(nn.Module):
    """ResNet50 with a heteroscedastic regression head.

    Forward output shape: (B, n_params, 2)
        [..., 0] = μ  — passed through Sigmoid, bounded [0, 1] in normalised space
        [..., 1] = log_σ — unconstrained; apply F.softplus() at inference to get σ > 0

    Parameter order: [k_min, k_max, sigma]
    """

    def __init__(self, in_channels: int = 1, n_params: int = 3):
        super().__init__()
        self.n_params = n_params
        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_params * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                       # (B, 2048)
        raw = self.head(features)                          # (B, n_params * 2)
        raw = raw.view(-1, self.n_params, 2)              # (B, n_params, 2)
        mu = torch.sigmoid(raw[..., 0])                   # (B, n_params) — [0, 1]
        log_sigma = raw[..., 1]                           # (B, n_params) — unconstrained
        return torch.stack([mu, log_sigma], dim=-1)       # (B, n_params, 2)
