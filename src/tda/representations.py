from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from persim import PersistenceImager


@dataclass
class PersistenceImageResult:
    homology_dimension: int
    image: np.ndarray
    imager: PersistenceImager


def build_persistence_image(
    diagram: np.ndarray,
    homology_dimension: int,
    pixel_size: float = 0.1,
    birth_range: tuple[float, float] = (0.0, 2.0),
    pers_range: tuple[float, float] = (0.0, 2.0),
) -> PersistenceImageResult:
    """Convert a persistence diagram into a persistence image."""
    finite_mask = np.isfinite(diagram).all(axis=1)
    finite_diagram = diagram[finite_mask]
    if finite_diagram.size == 0:
        x_bins = max(1, int((birth_range[1] - birth_range[0]) / pixel_size))
        y_bins = max(1, int((pers_range[1] - pers_range[0]) / pixel_size))
        image = np.zeros((y_bins, x_bins), dtype=np.float32)
        return PersistenceImageResult(homology_dimension=homology_dimension, image=image, imager=PersistenceImager())

    imager = PersistenceImager(
        pixel_size=pixel_size,
        birth_range=birth_range,
        pers_range=pers_range,
    )
    image = imager.transform(finite_diagram)
    return PersistenceImageResult(homology_dimension=homology_dimension, image=np.asarray(image, dtype=np.float32), imager=imager)
