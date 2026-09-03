from __future__ import annotations

import numpy as np


def delay_embed(signal: np.ndarray, dimension: int, delay: int, stride: int = 1) -> np.ndarray:
    """Construct a time-delay embedding from a 1D signal."""
    if dimension < 2:
        raise ValueError("Embedding dimension must be at least 2.")
    if delay < 1:
        raise ValueError("Delay must be at least 1.")
    if stride < 1:
        raise ValueError("Stride must be at least 1.")

    usable = len(signal) - (dimension - 1) * delay
    if usable <= 0:
        raise ValueError(
            f"Signal length {len(signal)} is too short for dimension={dimension}, delay={delay}."
        )

    rows = []
    for start in range(0, usable, stride):
        rows.append(signal[start : start + dimension * delay : delay])
    return np.asarray(rows, dtype=np.float32)
