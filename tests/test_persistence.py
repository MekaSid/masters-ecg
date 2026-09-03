from __future__ import annotations

import numpy as np

from src.tda.embedding import delay_embed
from src.tda.persistence import compute_persistence_diagrams


def test_persistence_computation_returns_h0_h1() -> None:
    signal = np.sin(np.linspace(0, 8 * np.pi, 128)).astype(np.float32)
    embedding = delay_embed(signal, dimension=3, delay=3, stride=1)
    result = compute_persistence_diagrams(embedding, maxdim=1)

    assert len(result.diagrams) == 2
    assert result.diagrams[0].shape[1] == 2
