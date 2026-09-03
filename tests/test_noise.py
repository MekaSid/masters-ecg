from __future__ import annotations

import numpy as np

from src.data.noise import NoiseAugmentor


def test_noise_augmentor_hits_target_snr() -> None:
    clean = np.sin(np.linspace(0, 4 * np.pi, 256)).astype(np.float32)
    noise = np.random.default_rng(0).normal(size=4096).astype(np.float32)

    augmentor = NoiseAugmentor(seed=123)
    result = augmentor.add_noise(clean, noise, noise_type="ma", snr_db=12.0)

    assert result.noisy.shape == clean.shape
    assert abs(result.snr_db_achieved - 12.0) < 0.2
