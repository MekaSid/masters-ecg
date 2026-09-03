from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseResult:
    clean: np.ndarray
    noisy: np.ndarray
    noise: np.ndarray
    noise_type: str
    snr_db_target: float
    snr_db_achieved: float
    start_index: int


def compute_signal_power(signal: np.ndarray) -> float:
    return float(np.mean(np.square(signal.astype(np.float64))))


def estimate_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    signal_power = compute_signal_power(clean)
    noise_power = compute_signal_power(noise)
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = compute_signal_power(clean)
    noise_power = compute_signal_power(noise)
    if noise_power == 0:
        raise ValueError("Noise power is zero; cannot scale to requested SNR.")

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return noise * scale


class NoiseAugmentor:
    """Add NSTDB noise to a clean ECG segment at a requested SNR."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_noise_segment(self, noise_signal: np.ndarray, length: int) -> tuple[np.ndarray, int]:
        if length > len(noise_signal):
            raise ValueError(f"Requested noise segment length {length} exceeds noise recording length {len(noise_signal)}.")
        start = int(self.rng.integers(0, len(noise_signal) - length + 1))
        return noise_signal[start : start + length].astype(np.float32, copy=True), start

    def add_noise(
        self,
        clean_signal: np.ndarray,
        noise_signal: np.ndarray,
        noise_type: str,
        snr_db: float,
    ) -> NoiseResult:
        noise_segment, start_index = self.sample_noise_segment(noise_signal, len(clean_signal))
        scaled_noise = scale_noise_to_snr(clean_signal, noise_segment, snr_db).astype(np.float32)
        noisy = clean_signal.astype(np.float32) + scaled_noise
        achieved = estimate_snr_db(clean_signal, noisy)
        return NoiseResult(
            clean=clean_signal.astype(np.float32),
            noisy=noisy.astype(np.float32),
            noise=scaled_noise,
            noise_type=noise_type,
            snr_db_target=float(snr_db),
            snr_db_achieved=float(achieved),
            start_index=start_index,
        )
