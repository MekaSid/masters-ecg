from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RawECGModelConfig:
    """Configuration for the baseline three-block raw ECG Conv1D classifier."""

    input_length: int
    num_classes: int
    channels: tuple[int, int, int] = (32, 64, 128)
    dropout: float = 0.3


class RawECGConvNet(nn.Module):
    """ECG beat -> Conv1D -> Conv1D -> Conv1D -> pooling -> fully connected."""

    def __init__(self, config: RawECGModelConfig) -> None:
        super().__init__()
        if config.input_length < 8:
            raise ValueError("input_length must be at least 8 for three pooling layers.")
        if config.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        c1, c2, c3 = config.channels
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        pooled_length = config.input_length // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * pooled_length, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape (batch, 1, samples), received {tuple(x.shape)}.")
        return self.classifier(self.features(x))
