from __future__ import annotations

import numpy as np

from src.data.beat_extraction import BeatWindow, extract_beats


def test_extract_beats_skips_boundaries_and_maps_labels() -> None:
    signal = np.arange(100, dtype=np.float32)
    window = BeatWindow(pre_samples=10, post_samples=10)

    beats = extract_beats(
        signal=signal,
        annotation_samples=[5, 25, 95],
        annotation_symbols=["N", "V", "A"],
        record_id="100",
        window=window,
    )

    assert len(beats) == 1
    assert beats[0].original_symbol == "V"
    assert beats[0].mapped_class == "V"
    assert beats[0].waveform.shape == (20,)
