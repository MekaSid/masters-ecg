from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionModelConfig:
    """Placeholder config for the future raw ECG + TDA fusion milestone."""

    signal_length: int
    image_shape: tuple[int, int]
    num_classes: int
