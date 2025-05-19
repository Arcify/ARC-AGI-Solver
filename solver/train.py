"""Training utilities for ARC-AGI solver."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ARCDataset, collate_grids
from .model import SimpleCNN


def train(dataset_root: str, epochs: int = 5) -> SimpleCNN:
    train_ds = ARCDataset(dataset_root, "training")

    pairs = []
    for task in train_ds:
        pairs.extend(task["train"])

    train_loader = DataLoader(pairs, batch_size=4, shuffle=True, collate_fn=collate_grids)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model
