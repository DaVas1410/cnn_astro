import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedGaussianNLL(nn.Module):
    """Weighted Gaussian negative log-likelihood loss for heteroscedastic regression.

    Expects pred shape (B, n_params, 2):
        pred[..., 0] = mu        (Sigmoid-bounded [0,1])
        pred[..., 1] = log_sigma (unconstrained; softplus applied internally)

    target shape (B, n_params): normalised targets in [0, 1]

    Loss = sum_i w_i * mean_B [ (y_i - mu_i)^2 / (2*sigma_i^2) + log(sigma_i) ]
    """

    def __init__(self, weights: list):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = pred[..., 0]                                    # (B, n_params)
        log_sigma = pred[..., 1]                             # (B, n_params)
        sigma = F.softplus(log_sigma) + 1e-6                 # (B, n_params), σ > 0
        nll = 0.5 * (target - mu).pow(2) / sigma.pow(2) + torch.log(sigma)
        # Mean over batch, weighted sum over params
        return (self.weights * nll.mean(dim=0)).sum()
