from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TDAModelConfig:
    """Placeholder config for the future TDA-only model milestone."""

    image_shape: tuple[int, int]
    num_classes: int
