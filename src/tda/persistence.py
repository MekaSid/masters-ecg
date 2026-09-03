from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ripser import ripser


@dataclass
class PersistenceResult:
    embedding: np.ndarray
    diagrams: list[np.ndarray]
    maxdim: int


def compute_persistence_diagrams(
    embedding: np.ndarray,
    maxdim: int = 1,
    thresh: float | None = None,
) -> PersistenceResult:
    """Compute Vietoris-Rips persistence diagrams from an embedding."""
    kwargs = {"maxdim": maxdim}
    if thresh is not None:
        kwargs["thresh"] = thresh
    result = ripser(embedding, **kwargs)
    diagrams = [np.asarray(diagram, dtype=np.float32) for diagram in result["dgms"]]
    return PersistenceResult(embedding=embedding, diagrams=diagrams, maxdim=maxdim)
