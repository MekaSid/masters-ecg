from __future__ import annotations

import numpy as np

from src.tda.embedding import delay_embed


def test_delay_embedding_shape() -> None:
    signal = np.arange(20, dtype=np.float32)
    embedded = delay_embed(signal, dimension=3, delay=2, stride=1)

    assert embedded.shape == (16, 3)
    np.testing.assert_array_equal(embedded[0], np.array([0.0, 2.0, 4.0], dtype=np.float32))
