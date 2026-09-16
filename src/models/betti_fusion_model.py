"""PH-only and raw-plus-Betti CNNs for controlled Dindin-style experiments."""

from __future__ import annotations

import torch
from torch import nn

from src.models.zhang_regular_cnn import ZhangRegularCNN, ZhangRegularCNNConfig


class BettiCurveEncoder(nn.Module):
    """Small 1D CNN which maps sublevel/upper-level Betti curves to 32 features."""

    def __init__(self, embedding_size: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, embedding_size),
            nn.PReLU(),
        )

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        if curves.ndim != 3 or curves.shape[1] != 2:
            raise ValueError(f"Expected Betti curves shaped (batch, 2, resolution), got {tuple(curves.shape)}.")
        return self.network(curves)


class DindinPHOnlyCNN(nn.Module):
    """Betti curves -> small CNN -> AAMI heartbeat classifier."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.encoder = BettiCurveEncoder()
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(curves))


class ZhangBettiFusionCNN(nn.Module):
    """Zhang raw/RR embedding concatenated with a Dindin-style Betti embedding."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.raw_encoder = ZhangRegularCNN(ZhangRegularCNNConfig(num_classes=num_classes))
        self.ph_encoder = BettiCurveEncoder()
        self.classifier = nn.Linear(64 + 32, num_classes)

    def forward(self, raw_features: torch.Tensor, betti_curves: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((self.raw_encoder.encode(raw_features), self.ph_encoder(betti_curves)), dim=1))
