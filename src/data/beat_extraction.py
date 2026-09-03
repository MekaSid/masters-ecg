from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.labels import map_mitdb_symbol


@dataclass
class BeatWindow:
    pre_samples: int
    post_samples: int

    @property
    def length(self) -> int:
        return self.pre_samples + self.post_samples


@dataclass
class BeatSample:
    record_id: str
    beat_index: int
    rpeak_sample: int
    original_symbol: str
    mapped_class: str
    start_sample: int
    end_sample: int
    waveform: np.ndarray

    def metadata(self) -> dict[str, int | str]:
        data = asdict(self)
        data.pop("waveform")
        return data


def extract_beats(
    signal: np.ndarray,
    annotation_samples: Iterable[int],
    annotation_symbols: Iterable[str],
    record_id: str,
    window: BeatWindow,
    unknown_class: str = "Q",
) -> list[BeatSample]:
    beats: list[BeatSample] = []
    for beat_index, (sample, symbol) in enumerate(zip(annotation_samples, annotation_symbols)):
        start = int(sample) - window.pre_samples
        end = int(sample) + window.post_samples
        if start < 0 or end > len(signal):
            continue

        waveform = signal[start:end].astype(np.float32, copy=True)
        label_info = map_mitdb_symbol(symbol, unknown_class=unknown_class)
        beats.append(
            BeatSample(
                record_id=record_id,
                beat_index=beat_index,
                rpeak_sample=int(sample),
                original_symbol=label_info.original_symbol,
                mapped_class=label_info.mapped_class,
                start_sample=start,
                end_sample=end,
                waveform=waveform,
            )
        )
    return beats


def beats_to_dataframe(beats: list[BeatSample]) -> pd.DataFrame:
    return pd.DataFrame([beat.metadata() for beat in beats])


def save_beats(
    beats: list[BeatSample],
    metadata_path: Path,
    waveforms_path: Path,
) -> None:
    if not beats:
        raise ValueError("No beats were extracted; nothing to save.")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    waveforms_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = beats_to_dataframe(beats)
    metadata.to_csv(metadata_path, index=False)

    waveforms = np.stack([beat.waveform for beat in beats], axis=0)
    np.savez_compressed(waveforms_path, waveforms=waveforms)
