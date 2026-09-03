from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from gudhi.representations import PersistenceImage


@dataclass(frozen=True)
class PersistenceImageConfig:
    resolution: tuple[int, int] = (20, 20)
    birth_range: tuple[float, float] = (0.0, 2.0)
    persistence_range: tuple[float, float] = (0.0, 2.0)
    bandwidth: float = 0.1


@dataclass
class PersistenceImageResult:
    homology_dimension: int
    image: np.ndarray
    transformer: PersistenceImage


def filter_finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    finite_mask = np.isfinite(diagram).all(axis=1)
    return diagram[finite_mask].astype(np.float32, copy=False)


def to_birth_persistence(diagram: np.ndarray) -> np.ndarray:
    finite_diagram = filter_finite_diagram(diagram)
    if finite_diagram.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    births = finite_diagram[:, 0]
    persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
    return np.column_stack([births, persistences]).astype(np.float32)


def fit_persistence_image_transformer(
    diagrams: Iterable[np.ndarray],
    config: PersistenceImageConfig,
) -> PersistenceImage:
    im_range = [
        float(config.birth_range[0]),
        float(config.birth_range[1]),
        float(config.persistence_range[0]),
        float(config.persistence_range[1]),
    ]
    transformer = PersistenceImage(
        bandwidth=float(config.bandwidth),
        resolution=[int(config.resolution[0]), int(config.resolution[1])],
        im_range=im_range,
    )
    birth_persistence_diagrams = [to_birth_persistence(diagram) for diagram in diagrams]
    if birth_persistence_diagrams:
        transformer.fit(birth_persistence_diagrams)
    return transformer


def build_persistence_image(
    diagram: np.ndarray,
    homology_dimension: int,
    config: PersistenceImageConfig,
    transformer: PersistenceImage | None = None,
) -> PersistenceImageResult:
    birth_persistence = to_birth_persistence(diagram)
    if transformer is None:
        transformer = fit_persistence_image_transformer([diagram], config)

    if birth_persistence.size == 0:
        image = np.zeros(config.resolution, dtype=np.float32)
    else:
        image = transformer.transform([birth_persistence])[0].reshape(config.resolution).astype(np.float32)
    return PersistenceImageResult(homology_dimension=homology_dimension, image=image, transformer=transformer)
