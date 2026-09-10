from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.beat_extraction import BeatWindow, extract_beats
from src.data.load_ecg import load_annotations, load_record


@dataclass(frozen=True)
class SplitSummary:
    split_name: str
    record_ids: tuple[str, ...]
    total_annotated_beats: int
    retained_beats: int
    excluded_by_class: int
    class_counts: dict[str, int]


class ECGBeatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory beat dataset with per-beat z-score normalization."""

    def __init__(self, waveforms: np.ndarray, labels: np.ndarray) -> None:
        if len(waveforms) != len(labels):
            raise ValueError("Waveform and label counts must match.")
        if waveforms.ndim != 2:
            raise ValueError(f"Expected waveforms of shape (beats, samples), got {waveforms.shape}.")
        means = waveforms.mean(axis=1, keepdims=True)
        stds = waveforms.std(axis=1, keepdims=True)
        normalized = (waveforms - means) / np.maximum(stds, 1e-6)
        self.waveforms = torch.from_numpy(normalized.astype(np.float32))[:, None, :]
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.waveforms[index], self.labels[index]


def build_beat_split(
    split_name: str,
    record_ids: Iterable[str],
    mitdb_dir: Path,
    window: BeatWindow,
    class_to_index: dict[str, int],
) -> tuple[ECGBeatDataset, SplitSummary]:
    """Load a record-disjoint split and retain only classes supported by the model."""
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    class_counts: Counter[str] = Counter()
    total_annotated_beats = 0
    excluded_by_class = 0
    resolved_record_ids = tuple(str(record_id) for record_id in record_ids)

    for record_id in resolved_record_ids:
        record = load_record(mitdb_dir, record_id)
        annotations = load_annotations(mitdb_dir, record_id)
        beats = extract_beats(record.signal, annotations.samples, annotations.symbols, record_id, window)
        total_annotated_beats += len(beats)
        for beat in beats:
            if beat.mapped_class not in class_to_index:
                excluded_by_class += 1
                continue
            waveforms.append(beat.waveform)
            labels.append(class_to_index[beat.mapped_class])
            class_counts[beat.mapped_class] += 1

    if not waveforms:
        raise ValueError(f"No usable beats found for {split_name}. Check records and configured classes.")
    return (
        ECGBeatDataset(np.stack(waveforms), np.asarray(labels)),
        SplitSummary(
            split_name=split_name,
            record_ids=resolved_record_ids,
            total_annotated_beats=total_annotated_beats,
            retained_beats=len(waveforms),
            excluded_by_class=excluded_by_class,
            class_counts=dict(sorted(class_counts.items())),
        ),
    )
