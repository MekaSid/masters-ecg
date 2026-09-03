from __future__ import annotations

import numpy as np

from src.tda.pipeline import BeatTDAPipeline, TakensEmbeddingConfig, VietorisRipsConfig
from src.tda.representations import PersistenceImageConfig


def test_vr_pipeline_returns_images_and_statistics() -> None:
    signal = np.sin(np.linspace(0, 6 * np.pi, 128)).astype(np.float32)
    pipeline = BeatTDAPipeline(
        embedding_config=TakensEmbeddingConfig(dimension=3, delay=3, stride=1),
        vr_config=VietorisRipsConfig(max_homology_dimension=1),
        image_config=PersistenceImageConfig(resolution=(8, 8), birth_range=(0.0, 2.0), persistence_range=(0.0, 2.0), bandwidth=0.2),
    )

    output = pipeline.run_vietoris_rips(signal)

    assert output.method == "takens_vr"
    assert output.embedding is not None
    assert output.persistence_images[0].image.shape == (8, 8)
    assert 0 in output.statistics


def test_sublevel_pipeline_returns_h0_only() -> None:
    signal = np.sin(np.linspace(0, 6 * np.pi, 128)).astype(np.float32)
    pipeline = BeatTDAPipeline(
        embedding_config=TakensEmbeddingConfig(),
        vr_config=VietorisRipsConfig(),
        image_config=PersistenceImageConfig(),
    )

    output = pipeline.run_sublevel(signal)

    assert output.method == "sublevel"
    assert output.embedding is None
    assert set(output.diagrams) == {0}
