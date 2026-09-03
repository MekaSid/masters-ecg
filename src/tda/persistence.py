from __future__ import annotations

from dataclasses import dataclass

import gudhi
import numpy as np


@dataclass
class PersistenceResult:
    """Persistent homology output for one ECG-derived object."""

    source_type: str
    diagrams: dict[int, np.ndarray]
    max_homology_dimension: int
    embedding: np.ndarray | None = None


def _collect_diagrams(simplex_tree: gudhi.SimplexTree, max_dimension: int) -> dict[int, np.ndarray]:
    simplex_tree.persistence()
    diagrams: dict[int, np.ndarray] = {}
    for dim in range(max_dimension + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        diagrams[dim] = np.asarray(intervals, dtype=np.float32).reshape(-1, 2)
    return diagrams


def compute_vr_persistence(
    embedding: np.ndarray,
    max_homology_dimension: int = 1,
    max_edge_length: float | None = None,
) -> PersistenceResult:
    """Compute Vietoris-Rips persistence on a delay-embedded point cloud."""
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, received shape {embedding.shape}.")

    rips = gudhi.RipsComplex(
        points=embedding.astype(np.float64),
        max_edge_length=float("inf") if max_edge_length is None else float(max_edge_length),
    )
    simplex_tree = rips.create_simplex_tree(max_dimension=max_homology_dimension + 1)
    diagrams = _collect_diagrams(simplex_tree, max_homology_dimension)
    return PersistenceResult(
        source_type="vietoris_rips",
        diagrams=diagrams,
        max_homology_dimension=max_homology_dimension,
        embedding=embedding.astype(np.float32, copy=False),
    )


def compute_sublevel_persistence(signal: np.ndarray) -> PersistenceResult:
    """Compute 1D sublevel-set persistence directly on the ECG waveform."""
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D waveform, received shape {signal.shape}.")

    cubical = gudhi.CubicalComplex(vertices=signal.astype(np.float64), dimensions=[len(signal)])
    cubical.persistence()
    diagrams = {0: np.asarray(cubical.persistence_intervals_in_dimension(0), dtype=np.float32).reshape(-1, 2)}
    return PersistenceResult(
        source_type="sublevel",
        diagrams=diagrams,
        max_homology_dimension=0,
        embedding=None,
    )
