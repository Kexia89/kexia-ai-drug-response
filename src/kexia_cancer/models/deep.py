from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TwoTowerRegressor(nn.Module):
    """Independent encoders for molecular cell features and compound features."""

    def __init__(self, cell_dim: int, drug_dim: int, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, cell: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cell_encoder(cell), self.drug_encoder(drug)], dim=1)).squeeze(1)


@dataclass
class DeepTrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict]


def train_two_tower(
    model: TwoTowerRegressor,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    output: Path,
    seed: int = 42,
    epochs: int = 300,
    patience: int = 25,
    batch_size: int = 128,
) -> DeepTrainingResult:
    """Reproducible training with weight decay and early stopping.

    Inputs must already be transformed using imputers/scalers fitted exclusively on
    the training partition. That preprocessing contract is deliberate and testable.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    xc, xd, y = (torch.as_tensor(x, dtype=torch.float32) for x in train_arrays)
    vc, vd, vy = (torch.as_tensor(x, dtype=torch.float32) for x in validation_arrays)
    loader = DataLoader(TensorDataset(xc, xd, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for cell, drug, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(cell, drug), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(vc, vd), vy))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, stale = validation_loss, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Deep model produced no valid checkpoint")
    model.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "seed": seed, "history": history}, output)
    return DeepTrainingResult(best_epoch, best_loss, history)

