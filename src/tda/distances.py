from __future__ import annotations

from dataclasses import dataclass

import gudhi
import numpy as np
from gudhi import hera

from src.tda.representations import filter_finite_diagram


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
    clean_finite = filter_finite_diagram(clean_diagram).astype(np.float64)
    noisy_finite = filter_finite_diagram(noisy_diagram).astype(np.float64)

    return DiagramDistance(
        homology_dimension=homology_dimension,
        bottleneck_distance=float(gudhi.bottleneck_distance(clean_finite, noisy_finite)),
        wasserstein_distance=float(hera.wasserstein_distance(clean_finite, noisy_finite, order=1.0, internal_p=float("inf"))),
    )
