"""Basic PyTorch model for ARC-AGI 2 benchmark."""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A very small convolutional network as a starting point."""

    def __init__(self, num_channels: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 3 * 3, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
