"""Training utilities for ARC-AGI solver."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ARCDataset
from .model import SimpleCNN


def train(dataset_root: str, epochs: int = 5) -> SimpleCNN:
    train_ds = ARCDataset(dataset_root, "training")
    # Placeholder dataset transformation
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            # TODO: implement proper training using grid pairs
            outputs = model(torch.randn(4, 3, 3, 3))
            targets = torch.randint(0, 10, (4,))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model
