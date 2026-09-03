from __future__ import annotations

import numpy as np

from src.tda.embedding import delay_embed
from src.tda.persistence import compute_sublevel_persistence, compute_vr_persistence


def test_vr_persistence_computation_returns_h0_h1() -> None:
    signal = np.sin(np.linspace(0, 8 * np.pi, 128)).astype(np.float32)
    embedding = delay_embed(signal, dimension=3, delay=3, stride=1)
    result = compute_vr_persistence(embedding, max_homology_dimension=1)

    assert set(result.diagrams) == {0, 1}
    assert result.diagrams[0].shape[1] == 2


def test_sublevel_persistence_returns_h0() -> None:
    signal = np.sin(np.linspace(0, 2 * np.pi, 64)).astype(np.float32)
    result = compute_sublevel_persistence(signal)

    assert set(result.diagrams) == {0}
    assert result.diagrams[0].shape[1] == 2
