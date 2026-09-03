from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tda.representations import filter_finite_diagram


@dataclass
class PersistenceStatistics:
    homology_dimension: int
    num_features: int
    total_persistence: float
    max_persistence: float
    mean_persistence: float
    persistent_entropy: float


def compute_persistence_statistics(diagram: np.ndarray, homology_dimension: int) -> PersistenceStatistics:
    finite_diagram = filter_finite_diagram(diagram)
    if finite_diagram.size == 0:
        return PersistenceStatistics(
            homology_dimension=homology_dimension,
            num_features=0,
            total_persistence=0.0,
            max_persistence=0.0,
            mean_persistence=0.0,
            persistent_entropy=0.0,
        )

    persistences = np.maximum(finite_diagram[:, 1] - finite_diagram[:, 0], 0.0).astype(np.float64)
    total = float(np.sum(persistences))
    if total > 0:
        probs = persistences / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        entropy = 0.0

    return PersistenceStatistics(
        homology_dimension=homology_dimension,
        num_features=int(len(persistences)),
        total_persistence=total,
        max_persistence=float(np.max(persistences)),
        mean_persistence=float(np.mean(persistences)),
        persistent_entropy=entropy,
    )
