"""Dindin-style Betti curve representations computed with GUDHI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tda.persistence import compute_sublevel_persistence


@dataclass(frozen=True)
class BettiCurveConfig:
    """Discretization used for sublevel and upper-level H0 Betti curves."""

    resolution: int = 128
    normalize_signal: bool = True


def normalize_unit_interval(signal: np.ndarray) -> np.ndarray:
    """Min-max normalize a finite waveform, returning zeros for a constant signal."""
    if signal.ndim != 1 or len(signal) < 2:
        raise ValueError("Expected a one-dimensional signal with at least two samples.")
    if not np.isfinite(signal).all():
        raise ValueError("Signal must contain only finite values.")
    minimum = float(signal.min())
    maximum = float(signal.max())
    if maximum == minimum:
        return np.zeros_like(signal, dtype=np.float32)
    return ((signal - minimum) / (maximum - minimum)).astype(np.float32)


def betti_curve(diagram: np.ndarray, resolution: int, essential_death: float) -> np.ndarray:
    """Count barcode intervals alive at uniformly sampled filtration values."""
    if resolution < 2:
        raise ValueError("Betti curve resolution must be at least two.")
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError(f"Expected persistence diagram shape (features, 2), got {diagram.shape}.")
    if len(diagram) == 0:
        return np.zeros(resolution, dtype=np.float32)

    intervals = diagram.astype(np.float32, copy=True)
    intervals[~np.isfinite(intervals[:, 1]), 1] = essential_death
    valid = np.isfinite(intervals).all(axis=1) & (intervals[:, 1] >= intervals[:, 0])
    intervals = intervals[valid]
    if len(intervals) == 0:
        return np.zeros(resolution, dtype=np.float32)
    grid = np.linspace(float(intervals[:, 0].min()), float(intervals[:, 1].max()), resolution, dtype=np.float32)
    alive = (intervals[:, 0, None] <= grid) & (grid <= intervals[:, 1, None])
    return alive.sum(axis=0, dtype=np.int32).astype(np.float32)


def dindin_betti_curves(signal: np.ndarray, config: BettiCurveConfig = BettiCurveConfig()) -> np.ndarray:
    """Return sublevel and upper-level H0 Betti curves for one ECG sequence.

    Dindin et al. compute barcodes for both a time series and its opposite.
    GUDHI represents the essential H0 bar with infinite death, which is clipped
    to the maximum filtration value before discretizing the curve.
    """
    prepared = normalize_unit_interval(signal) if config.normalize_signal else signal.astype(np.float32, copy=False)
    sublevel = compute_sublevel_persistence(prepared).diagrams[0]
    upper_signal = -prepared
    upper = compute_sublevel_persistence(upper_signal).diagrams[0]
    return np.stack(
        (
            betti_curve(sublevel, config.resolution, essential_death=float(prepared.max())),
            betti_curve(upper, config.resolution, essential_death=float(upper_signal.max())),
        )
    )
