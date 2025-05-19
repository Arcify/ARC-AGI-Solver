"""Basic PyTorch model for ARC-AGI 2 benchmark."""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A very small convolutional network as a starting point."""

    def __init__(self, num_channels: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, num_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x.float())
        out = self.head(out)
        return out
