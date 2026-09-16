"""Regular encoder-classifier CNN from Zhang et al. (2021), without adversary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ZhangRegularCNNConfig:
    num_classes: int = 5
    attention_reduction: int = 8


class SpatiotemporalAttention(nn.Module):
    """Channel (spatial) and temporal attention for feature maps of height one."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        pooled = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * self.temporal(pooled)


class ResidualAttentionBlock(nn.Module):
    """Two documented 1x3 convolutions, average-pool shortcut, then attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, reduction: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), dilation=(1, 3), padding=(0, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=(1, stride), stride=(1, stride), ceil_mode=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.attention = SpatiotemporalAttention(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.attention(self.activation(x + residual))


class ZhangRegularCNN(nn.Module):
    """Seven-convolution encoder and five-class classifier, excluding adversarial learning."""

    def __init__(self, config: ZhangRegularCNNConfig = ZhangRegularCNNConfig()) -> None:
        super().__init__()
        self.config = config
        self.initial = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualAttentionBlock(16, 16, stride=1, reduction=config.attention_reduction)
        self.block2 = ResidualAttentionBlock(16, 64, stride=2, reduction=config.attention_reduction)
        self.block3 = ResidualAttentionBlock(64, 64, stride=2, reduction=config.attention_reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 64-dimensional raw ECG/RR representation before classification."""
        if x.ndim != 4 or x.shape[1:] != (2, 3, 128):
            raise ValueError(f"Expected input shape (batch, 2, 3, 128), received {tuple(x.shape)}.")
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.pool(x).flatten(1)
