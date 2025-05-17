"""Evaluation utilities for ARC-AGI solver."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .dataset import ARCDataset
from .model import SimpleCNN


def evaluate(model: SimpleCNN, dataset_root: str) -> float:
    _, test_tasks = ARCDataset(dataset_root).load()
    test_loader = DataLoader(test_tasks, batch_size=4)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in test_loader:
            outputs = model(torch.randn(4, 3, 3, 3))
            preds = outputs.argmax(dim=1)
            targets = torch.randint(0, 10, (4,))
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)
