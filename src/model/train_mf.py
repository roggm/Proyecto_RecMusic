import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


RATING_MAP = {"like": 3.0, "play": 2.0, "skip": 1.0}


class InteractionsDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = users.astype(np.int64)
        self.items = items.astype(np.int64)
        self.ratings = ratings.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=32):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, n_factors)
        self.item_emb = torch.nn.Embedding(n_items, n_factors)
        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor([0.0]))

        # Initialization
        torch.nn.init.normal_(self.user_emb.weight, std=0.01)
        torch.nn.init.normal_(self.item_emb.weight, std=0.01)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, u, i):
        pu = self.user_emb(u)
        qi = self.item_emb(i)
        ub = self.user_bias(u).squeeze(-1)
        ib = self.item_bias(i).squeeze(-1)
        dot = (pu * qi).sum(dim=1)
        return dot + ub + ib + self.global_bias


def train_test_per_user(ratings_df, n_val_per_user=1, min_interactions=1, seed=42):
    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []

    for uid, group in ratings_df.groupby('USER_ID'):
        if len(group) <= min_interactions:
            train_parts.append(group)
            continue
        n_val = min(n_val_per_user, len(group))
        idx = group.sample(n=n_val, random_state=rng).index
        val_parts.append(group.loc[idx])
        train_parts.append(group.drop(idx))

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True) if len(val_parts) > 0 else pd.DataFrame(columns=ratings_df.columns)
    return train, val


def load_data(data_dir: Path):
    items_path = data_dir / 'items.csv'
    interactions_path = data_dir / 'interactions.csv'

    items = pd.read_csv(items_path)
    interactions = pd.read_csv(interactions_path)
    interactions = interactions.copy()
    interactions['rating'] = interactions['EVENT_TYPE'].map(RATING_MAP)
    # Keep max rating per (user,item)
    ratings = (
        interactions.groupby(['USER_ID', 'ITEM_ID'])['rating']
        .max()
        .reset_index()
    )
    return items, interactions, ratings


def prepare_indices(ratings):
    users = sorted(ratings['USER_ID'].unique())
    items = sorted(ratings['ITEM_ID'].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {s: i for i, s in enumerate(items)}
    ratings['u_idx'] = ratings['USER_ID'].map(user_to_idx)
    ratings['i_idx'] = ratings['ITEM_ID'].map(item_to_idx)
    return ratings, user_to_idx, item_to_idx


def run_training(
    data_dir,
    n_factors=64,
    epochs=50,
    lr=1e-3,
    batch_size=4096,
    weight_decay=1e-5,
    patience=5,
    val_per_user=1,
    device='cpu',
):
    items, interactions, ratings = load_data(data_dir)
    ratings, user_to_idx, item_to_idx = prepare_indices(ratings)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    train_df, val_df = train_test_per_user(ratings, n_val_per_user=val_per_user)

    train_ds = InteractionsDataset(train_df['u_idx'].to_numpy(), train_df['i_idx'].to_numpy(), train_df['rating'].to_numpy())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if len(val_df) > 0:
        val_ds = InteractionsDataset(val_df['u_idx'].to_numpy(), val_df['i_idx'].to_numpy(), val_df['rating'].to_numpy())
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    model = MatrixFactorization(n_users, n_items, n_factors=n_factors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    no_improvement = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_pct_within_05': [], 'val_pct_within_05': [],
        'train_rmse_norm': [], 'val_rmse_norm': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        for u, i, r in train_loader:
            u = u.to(device)
            i = i.to(device)
            r = r.to(device)
            pred = model(u, i)
            loss = loss_fn(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(r.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_preds_all = np.concatenate(train_preds)
        train_targets_all = np.concatenate(train_targets)
        train_mae = float(np.mean(np.abs(train_preds_all - train_targets_all)))
        train_pct_within_05 = float(np.mean(np.abs(train_preds_all - train_targets_all) <= 0.5) * 100)
        train_rmse = float(np.sqrt(np.mean((train_preds_all - train_targets_all) ** 2)))
        train_rmse_norm = float(train_rmse / 2.0) * 100  # Normalized to [0, 100%]

        val_loss = None
        val_mae = None
        val_pct_within_05 = None
        val_rmse_norm = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for u, i, r in val_loader:
                    u = u.to(device)
                    i = i.to(device)
                    r = r.to(device)
                    pred = model(u, i)
                    loss = loss_fn(pred, r)
                    val_losses.append(loss.item())
                    val_preds.append(pred.detach().cpu().numpy())
                    val_targets.append(r.detach().cpu().numpy())
            val_loss = float(np.mean(val_losses)) if val_losses else None
            val_preds_all = np.concatenate(val_preds)
            val_targets_all = np.concatenate(val_targets)
            val_mae = float(np.mean(np.abs(val_preds_all - val_targets_all)))
            val_pct_within_05 = float(np.mean(np.abs(val_preds_all - val_targets_all) <= 0.5) * 100)
            val_rmse = float(np.sqrt(np.mean((val_preds_all - val_targets_all) ** 2)))
            val_rmse_norm = float(val_rmse / 2.0) * 100

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else None)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae if val_mae is not None else None)
        history['train_pct_within_05'].append(train_pct_within_05)
        history['val_pct_within_05'].append(val_pct_within_05 if val_pct_within_05 is not None else None)
        history['train_rmse_norm'].append(train_rmse_norm)
        history['val_rmse_norm'].append(val_rmse_norm if val_rmse_norm is not None else None)

        val_pct_disp = f'{val_pct_within_05:.1f}%' if val_pct_within_05 is not None else '-'
        print(f'Epoch {epoch:3d} | loss={train_loss:.6f}/{val_loss if val_loss is not None else "-":>7} | pct_±0.5={train_pct_within_05:>6.1f}%/{val_pct_disp:>6}')

        # Early stopping condition
        if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                # Save best model
                best_path = data_dir / 'modelo_mf_best.pt'
                torch.save({'model_state': model.state_dict(), 'user_to_idx': user_to_idx, 'item_to_idx': item_to_idx}, best_path)
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f'Early stopping triggered! (no improvement in {patience} epochs)')
                    break

    # Save final model and history
    out_model = data_dir / 'modelo_mf_final.pt'
    torch.save({'model_state': model.state_dict(), 'user_to_idx': user_to_idx, 'item_to_idx': item_to_idx}, out_model)

    history_path = data_dir / 'mf_history.json'
    with open(history_path, 'w', encoding='utf-8') as fh:
        json.dump(history, fh)

    # Plot learning curves (Loss, % within ±0.5, RMSE norm, Overfitting analysis)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    epochs_range = list(range(1, len(history['train_loss']) + 1))
    
    # Top-left: Loss curve with overfitting gap
    axes[0, 0].plot(epochs_range, history['train_loss'], marker='o', linewidth=2, markersize=4, label='Train Loss', color='#1f77b4')
    if any(v is not None for v in history['val_loss']):
        axes[0, 0].plot(epochs_range, [v if v is not None else np.nan for v in history['val_loss']], marker='s', linewidth=2, markersize=4, label='Val Loss', color='#ff7f0e')
    axes[0, 0].fill_between(epochs_range, history['train_loss'], [v if v is not None else np.nan for v in history['val_loss']], alpha=0.2, color='red', label='Overfitting gap')
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=10)
    axes[0, 0].set_title('Loss Curve (with Overfitting Gap)', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-right: % predictions within ±0.5 stars
    axes[0, 1].plot(epochs_range, history['train_pct_within_05'], marker='o', linewidth=2, markersize=4, label='Train %', color='#2ca02c')
    if any(v is not None for v in history['val_pct_within_05']):
        axes[0, 1].plot(epochs_range, [v if v is not None else np.nan for v in history['val_pct_within_05']], marker='s', linewidth=2, markersize=4, label='Val %', color='#d62728')
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].set_ylabel('% Predictions within ±0.5 stars', fontsize=10)
    axes[0, 1].set_title('Accuracy: Predictions within ±0.5 stars', fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 105])
    
    # Bottom-left: RMSE normalized
    axes[1, 0].plot(epochs_range, history['train_rmse_norm'], marker='o', linewidth=2, markersize=4, label='Train RMSE%', color='#9467bd')
    if any(v is not None for v in history['val_rmse_norm']):
        axes[1, 0].plot(epochs_range, [v if v is not None else np.nan for v in history['val_rmse_norm']], marker='s', linewidth=2, markersize=4, label='Val RMSE%', color='#8c564b')
    axes[1, 0].set_xlabel('Epoch', fontsize=10)
    axes[1, 0].set_ylabel('RMSE Normalized (%)', fontsize=10)
    axes[1, 0].set_title('RMSE Normalized [0-100%]', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: Overfitting analysis (train vs val gap)
    if any(v is not None for v in history['val_loss']):
        overfitting_gap = [v - t if v is not None else np.nan for v, t in zip(history['val_loss'], history['train_loss'])]
        axes[1, 1].plot(epochs_range, overfitting_gap, marker='D', linewidth=2.5, markersize=5, color='#e377c2', label='Val Loss - Train Loss')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='No overfitting')
        axes[1, 1].fill_between(epochs_range, overfitting_gap, 0, alpha=0.3, color='red')
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].set_ylabel('Loss Difference', fontsize=10)
    axes[1, 1].set_title('Overfitting Gap Analysis', fontsize=11, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = data_dir / 'mf_learning_curves.png'
    fig.savefig(curve_path)
    plt.close(fig)

    print(f'Model saved: {out_model}')
    print(f'History saved: {history_path}')
    print(f'Learning curves saved: {curve_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=str(Path(__file__).resolve().parents[2] / 'data' / 'processed'))
    parser.add_argument('--factors', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--val-per-user', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    run_training(
        data_dir,
        n_factors=args.factors,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        patience=args.patience,
        val_per_user=args.val_per_user,
        device=args.device,
    )


if __name__ == '__main__':
    main()
