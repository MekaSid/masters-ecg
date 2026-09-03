from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from persim import bottleneck, wasserstein


@dataclass
class DiagramDistance:
    homology_dimension: int
    bottleneck_distance: float
    wasserstein_distance: float


def compute_diagram_distance(
    clean_diagram: np.ndarray,
    noisy_diagram: np.ndarray,
    homology_dimension: int,
) -> DiagramDistance:
    clean_finite = clean_diagram[np.isfinite(clean_diagram).all(axis=1)]
    noisy_finite = noisy_diagram[np.isfinite(noisy_diagram).all(axis=1)]

    return DiagramDistance(
        homology_dimension=homology_dimension,
        bottleneck_distance=float(bottleneck(clean_finite, noisy_finite)),
        wasserstein_distance=float(wasserstein(clean_finite, noisy_finite)),
    )
