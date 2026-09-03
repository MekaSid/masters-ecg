from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tda.embedding import delay_embed
from src.tda.persistence import PersistenceResult, compute_sublevel_persistence, compute_vr_persistence
from src.tda.representations import PersistenceImageConfig, PersistenceImageResult, build_persistence_image
from src.tda.statistics import PersistenceStatistics, compute_persistence_statistics


@dataclass(frozen=True)
class TakensEmbeddingConfig:
    dimension: int = 3
    delay: int = 5
    stride: int = 1


@dataclass(frozen=True)
class VietorisRipsConfig:
    max_homology_dimension: int = 1
    max_edge_length: float | None = None


@dataclass
class TDAOutput:
    method: str
    persistence: PersistenceResult
    persistence_images: dict[int, PersistenceImageResult]
    statistics: dict[int, PersistenceStatistics]

    @property
    def embedding(self) -> np.ndarray | None:
        return self.persistence.embedding

    @property
    def diagrams(self) -> dict[int, np.ndarray]:
        return self.persistence.diagrams


class BeatTDAPipeline:
    """High-level TDA interfaces for ECG beats."""

    def __init__(
        self,
        embedding_config: TakensEmbeddingConfig,
        vr_config: VietorisRipsConfig,
        image_config: PersistenceImageConfig,
    ) -> None:
        self.embedding_config = embedding_config
        self.vr_config = vr_config
        self.image_config = image_config

    def run_vietoris_rips(self, signal: np.ndarray) -> TDAOutput:
        embedding = delay_embed(
            signal,
            dimension=self.embedding_config.dimension,
            delay=self.embedding_config.delay,
            stride=self.embedding_config.stride,
        )
        persistence = compute_vr_persistence(
            embedding,
            max_homology_dimension=self.vr_config.max_homology_dimension,
            max_edge_length=self.vr_config.max_edge_length,
        )
        return self._build_output("takens_vr", persistence)

    def run_sublevel(self, signal: np.ndarray) -> TDAOutput:
        persistence = compute_sublevel_persistence(signal)
        return self._build_output("sublevel", persistence)

    def _build_output(self, method: str, persistence: PersistenceResult) -> TDAOutput:
        images: dict[int, PersistenceImageResult] = {}
        stats: dict[int, PersistenceStatistics] = {}
        for dim, diagram in persistence.diagrams.items():
            images[dim] = build_persistence_image(diagram, homology_dimension=dim, config=self.image_config)
            stats[dim] = compute_persistence_statistics(diagram, homology_dimension=dim)
        return TDAOutput(method=method, persistence=persistence, persistence_images=images, statistics=stats)
