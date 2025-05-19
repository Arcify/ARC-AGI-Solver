"""Evaluation utilities for ARC-AGI solver."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .dataset import ARCDataset
from .model import SimpleCNN


def _preprocess_grid(grid: torch.Tensor) -> torch.Tensor:
    """Resize a grid tensor to ``(1, 3, 3, 3)`` for ``SimpleCNN``."""
    grid = grid.to(torch.float32).unsqueeze(0).unsqueeze(0)
    grid = F.interpolate(grid, size=(3, 3), mode="nearest")
    return grid.repeat(1, 3, 1, 1)


def evaluate(model: SimpleCNN, dataset_root: str) -> float:
    """Evaluate ``model`` on the ARC evaluation split.

    The function iterates over the tasks returned by :class:`ARCDataset`, runs
    the model on each test grid and compares the predictions to the provided
    ground-truth solutions.  It returns the fraction of correctly solved test
    pairs.
    """

    eval_ds = ARCDataset(dataset_root, "evaluation")
    loader = DataLoader(eval_ds, batch_size=1)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for task in loader:
            # task is a dict with keys "train" and "test"; we only use "test"
            for inp, out in task["test"]:
                x = _preprocess_grid(inp)
                logits = model(x)
                color = logits.argmax(dim=1).item()
                pred = torch.full_like(out, color)
                if torch.equal(pred, out):
                    correct += 1
                total += 1

    return correct / max(total, 1)
