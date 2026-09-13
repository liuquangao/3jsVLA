from __future__ import annotations

import torch
from torch import nn


class DepthwiseBlock(nn.Module):
    def __init__(self, source: int, target: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(source, source, 3, stride=stride, padding=1, groups=source, bias=False),
            nn.BatchNorm2d(source),
            nn.SiLU(),
            nn.Conv2d(source, target, 1, bias=False),
            nn.BatchNorm2d(target),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class TinyVisionEncoder(nn.Module):
    """Small CNN that deliberately preserves a 2D grid of visual tokens."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(32), nn.SiLU()
        )
        self.blocks = nn.Sequential(
            DepthwiseBlock(32, 48, 2),
            DepthwiseBlock(48, 96, 2),
            DepthwiseBlock(96, 160, 2),
            DepthwiseBlock(160, hidden_dim, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 8))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.blocks(self.stem(images)))
        return features.flatten(2).transpose(1, 2)
