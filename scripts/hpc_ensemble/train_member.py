# scripts/hpc_ensemble/train_member.py
"""Train one ensemble member. Usage:
    python train_member.py --config config.yaml --member-id 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_config, output_dir
from dataset import load_splits
from loss import WeightedGaussianNLL
from model import ResNet50HeteroJoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True, help='Path to config.yaml')
    p.add_argument('--member-id', required=True, type=int, help='Ensemble member index (0-4)')
    return p.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda'):
                loss = criterion(model(X), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total += criterion(model(X), y).item() * len(X)
    return total / len(loader.dataset)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    member_id = args.member_id
    seed = cfg['training']['base_seed'] + member_id
    set_seeds(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[member {member_id}] seed={seed} | device={device}')

    out_dir = output_dir(cfg, member_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_splits(cfg)
    loader_kw = dict(batch_size=cfg['training']['batch_size'], num_workers=0, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    model     = ResNet50HeteroJoint().to(device)
    lw        = cfg['loss_weights']
    criterion = WeightedGaussianNLL([lw['k_min'], lw['k_max'], lw['sigma']]).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['training']['lr'],
                            weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5, min_lr=1e-7)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history        = {'train_loss': [], 'val_loss': []}
    best_val_loss  = float('inf')
    best_ckpt      = out_dir / 'best_model.pt'
    epochs         = cfg['training']['epochs']
    t0             = time.time()

    for epoch in range(epochs):
        tl = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step(vl)

        history['train_loss'].append(float(tl))
        history['val_loss'].append(float(vl))

        lr = optimizer.param_groups[0]['lr']
        print(f'  Ep {epoch+1:3d}/{epochs} | train {tl:.4f} | val {vl:.4f} | lr {lr:.1e}')

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), best_ckpt)

    elapsed = (time.time() - t0) / 60.0
    print(f'[member {member_id}] done in {elapsed:.1f} min — best val {best_val_loss:.4f}')

    with open(out_dir / 'history.json', 'w') as f:
        json.dump({
            'member_id': member_id,
            'seed': seed,
            'best_val_loss': best_val_loss,
            'training_time_min': elapsed,
            'history': history,
        }, f, indent=2)


if __name__ == '__main__':
    main()
