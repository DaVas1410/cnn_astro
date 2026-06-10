# scripts/hpc_ensemble/evaluate_ensemble.py
"""Combine 5 trained ensemble members and evaluate on the test set. Usage:
    python evaluate_ensemble.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from config import load_config, output_dir
from dataset import load_splits, denorm, uncertainty_to_orig
from model import ResNet50HeteroJoint

PARAMS = ['k_min', 'k_max', 'sigma']
COLORS = {'k_min': '#2196F3', 'k_max': '#F44336', 'sigma': '#4CAF50'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    return p.parse_args()


@torch.no_grad()
def predict_member(model: ResNet50HeteroJoint, loader: DataLoader, device) -> np.ndarray:
    """Returns (N, 3, 2): [..., 0]=μ, [..., 1]=log_σ in normalised space."""
    model.eval()
    preds = []
    for X, _ in loader:
        preds.append(model(X.to(device)).cpu())
    return torch.cat(preds).numpy()


def combine_ensemble(member_preds: list) -> dict:
    """Combine list of (N, 3, 2) arrays into ensemble statistics.

    Returns dict with keys: mu, sigma_aleatoric, sigma_epistemic, sigma_total — each (N, 3).
    All values are in normalised [0, 1] space.
    """
    mus        = np.stack([p[..., 0] for p in member_preds])   # (M, N, 3)
    log_sigmas = np.stack([p[..., 1] for p in member_preds])   # (M, N, 3)
    sigmas = np.logaddexp(0, log_sigmas) + 1e-6                 # softplus + floor, matches loss.py (M, N, 3)

    mu_ens          = mus.mean(axis=0)                         # (N, 3)
    sigma_aleatoric = np.sqrt((sigmas ** 2).mean(axis=0))      # (N, 3)
    sigma_epistemic = mus.std(axis=0)                          # (N, 3)
    sigma_total     = np.sqrt(sigma_aleatoric**2 + sigma_epistemic**2)  # (N, 3)

    return {
        'mu':              mu_ens,
        'sigma_aleatoric': sigma_aleatoric,
        'sigma_epistemic': sigma_epistemic,
        'sigma_total':     sigma_total,
    }


def plot_scatter(y_true: dict, y_pred: dict, y_sigma: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, param in zip(axes, PARAMS):
        yt, yp, ys = y_true[param], y_pred[param], y_sigma[param]
        r2  = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        ax.errorbar(yt, yp, yerr=ys, fmt='none', alpha=0.08, color=COLORS[param], elinewidth=0.5)
        ax.scatter(yt, yp, alpha=0.15, s=4, color=COLORS[param], rasterized=True)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Pred {param}')
        ax.set_title(f'{param}\nR²={r2:.4f}  MAE={mae:.3f}')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Predicted vs True — Ensemble (μ ± σ_total)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'scatter.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true: dict, y_pred: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, param in zip(axes, PARAMS):
        res = y_pred[param] - y_true[param]
        ax.hist(res, bins=80, color=COLORS[param], alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(res.mean(), color='red', linestyle='--', linewidth=1,
                   label=f'mean={res.mean():.3f}  std={res.std():.3f}')
        ax.set_xlabel(f'Residual (pred − true {param})')
        ax.set_ylabel('Count')
        ax.set_title(param)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Residual Distributions — Ensemble', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'residuals.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration(y_true: dict, y_pred: dict, y_sigma: dict, out_dir: Path):
    """Reliability diagram: observed coverage vs expected confidence level."""
    confidence_levels = np.linspace(0.05, 0.95, 19)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, param in zip(axes, PARAMS):
        yt, yp, ys = y_true[param], y_pred[param], y_sigma[param]
        observed = []
        for conf in confidence_levels:
            z = scipy.stats.norm.ppf((1.0 + conf) / 2.0)
            observed.append((np.abs(yt - yp) <= z * ys).mean())
        ax.plot(confidence_levels, observed, 'o-', color=COLORS[param], markersize=4, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
        ax.set_xlabel('Expected confidence')
        ax.set_ylabel('Observed coverage')
        ax.set_title(param)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Uncertainty Calibration — Reliability Diagram', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'uncertainty_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_members = cfg['ensemble']['n_members']
    base_dir  = output_dir(cfg)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load test set
    _, _, test_ds = load_splits(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'],
                             shuffle=False, num_workers=0)
    y_true_norm = test_ds.labels  # (N, 3)

    # Collect predictions from all members
    member_preds = []
    for mid in range(n_members):
        ckpt = output_dir(cfg, mid) / 'best_model.pt'
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
        model = ResNet50HeteroJoint().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        member_preds.append(predict_member(model, test_loader, device))
        print(f'  Loaded member {mid}')

    # Combine
    ens = combine_ensemble(member_preds)  # all (N, 3), normalised space

    # Denormalise to original units
    y_true = denorm(y_true_norm, cfg)
    y_pred = denorm(ens['mu'], cfg)
    y_sigma_total     = uncertainty_to_orig(ens['sigma_total'],     ens['mu'], cfg)
    y_sigma_aleatoric = uncertainty_to_orig(ens['sigma_aleatoric'], ens['mu'], cfg)
    y_sigma_epistemic = uncertainty_to_orig(ens['sigma_epistemic'], ens['mu'], cfg)

    # Metrics
    metrics = {}
    for i, param in enumerate(PARAMS):
        metrics[param] = {
            'mae':              float(mean_absolute_error(y_true[param], y_pred[param])),
            'rmse':             float(np.sqrt(mean_squared_error(y_true[param], y_pred[param]))),
            'r2':               float(r2_score(y_true[param], y_pred[param])),
            'mean_sigma_total': float(y_sigma_total[param].mean()),
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Ensemble ({n_members} members) — test set performance")
    print(f"{'='*60}")
    print(f"{'PARAM':<8} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'σ_total':>10}")
    print(f"{'-'*60}")
    for param, m in metrics.items():
        print(f"{param:<8} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.4f} {m['mean_sigma_total']:>10.4f}")
    print(f"{'='*60}")

    # Save JSON results
    results = {
        'metrics': metrics,
        'n_members': n_members,
        'params': PARAMS,
        'y_true':              {p: y_true[p].tolist() for p in PARAMS},
        'y_pred':              {p: y_pred[p].tolist() for p in PARAMS},
        'sigma_total':         {p: y_sigma_total[p].tolist() for p in PARAMS},
        'sigma_aleatoric':     {p: y_sigma_aleatoric[p].tolist() for p in PARAMS},
        'sigma_epistemic':     {p: y_sigma_epistemic[p].tolist() for p in PARAMS},
    }
    with open(base_dir / 'results.json', 'w') as f:
        json.dump(results, f)

    # Write human-readable summary
    with open(base_dir / 'results_summary.txt', 'w') as f:
        f.write(f"Ensemble ({n_members} members) — test set\n")
        f.write(f"{'='*50}\n")
        f.write(f"{'PARAM':<8} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'σ_total':>10}\n")
        for param, m in metrics.items():
            f.write(f"{param:<8} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
                    f"{m['r2']:>8.4f} {m['mean_sigma_total']:>10.4f}\n")

    # Plots
    plot_scatter(y_true, y_pred, y_sigma_total, base_dir)
    plot_residuals(y_true, y_pred, base_dir)
    plot_calibration(y_true, y_pred, y_sigma_total, base_dir)
    print(f'Results saved to {base_dir}')


if __name__ == '__main__':
    main()
