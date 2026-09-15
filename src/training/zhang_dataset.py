"""Preprocessing used by Zhang et al. (2021) for inter-patient classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import wfdb
from scipy.signal import resample
from torch.utils.data import Dataset


# Only beat annotations belong in the five AAMI heartbeat classes. Rhythm/noise
# annotations such as '+' and '~' are deliberately excluded.
ZHANG_LABEL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


@dataclass(frozen=True)
class ZhangSplitSummary:
    record_ids: tuple[str, ...]
    retained_beats: int
    class_counts: dict[str, int]


class ZhangECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory two-lead ECG/RR feature tensors for the Zhang regular CNN."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 4 or features.shape[1:] != (2, 3, 128):
            raise ValueError(f"Expected features shaped (beats, 2, 3, 128), got {features.shape}.")
        if len(features) != len(labels):
            raise ValueError("Feature and label counts must match.")
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def dynamic_beat_bounds(previous_rpeak: int, current_rpeak: int) -> tuple[int, int]:
    """Return Zhang et al.'s window: previous R peak + 50 through current + 100."""
    return previous_rpeak + 50, current_rpeak + 100


def rr_ratio_features(rpeaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute global and trailing-ten pre-RR ratios for every annotated beat.

    The first beat has no previous interval and is marked NaN. For early beats,
    the local denominator uses all available preceding intervals.
    """
    if rpeaks.ndim != 1 or len(rpeaks) < 2:
        raise ValueError("At least two one-dimensional R-peak positions are required.")
    intervals = np.diff(rpeaks).astype(np.float32)
    if np.any(intervals <= 0):
        raise ValueError("R-peak positions must be strictly increasing.")

    global_mean = float(intervals.mean())
    global_ratio = np.full(len(rpeaks), np.nan, dtype=np.float32)
    local_ratio = np.full(len(rpeaks), np.nan, dtype=np.float32)
    for beat_index in range(1, len(rpeaks)):
        current = intervals[beat_index - 1]
        preceding = intervals[max(0, beat_index - 11) : beat_index - 1]
        # No earlier interval exists for the second beat; use its own interval
        # as the only available local-rate estimate.
        local_mean = float(preceding.mean()) if len(preceding) else float(current)
        global_ratio[beat_index] = current / global_mean
        local_ratio[beat_index] = current / local_mean
    return global_ratio, local_ratio


def build_zhang_feature(beat: np.ndarray, pre_rr_ratio: float, near_pre_rr_ratio: float) -> np.ndarray:
    """Create the paper's two-lead, three-row, 128-sample feature tensor."""
    if beat.shape != (128, 2):
        raise ValueError(f"Expected resampled beat shape (128, 2), got {beat.shape}.")
    if not np.isfinite(pre_rr_ratio) or not np.isfinite(near_pre_rr_ratio):
        raise ValueError("RR ratios must be finite.")
    centered = beat - beat.mean(axis=0, keepdims=True)
    feature = np.empty((2, 3, 128), dtype=np.float32)
    feature[:, 0, :] = centered.T
    feature[:, 1, :] = pre_rr_ratio
    feature[:, 2, :] = near_pre_rr_ratio
    return feature


def build_zhang_split(
    record_ids: Iterable[str],
    mitdb_dir: Path,
    class_to_index: dict[str, int],
) -> tuple[ZhangECGDataset, ZhangSplitSummary]:
    """Build Zhang-style dynamic two-lead features for a collection of records."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    counts: Counter[str] = Counter()
    resolved_ids = tuple(str(record_id) for record_id in record_ids)

    for record_id in resolved_ids:
        record_path = str(mitdb_dir / record_id)
        record = wfdb.rdrecord(record_path)
        if record.p_signal.shape[1] < 2:
            raise ValueError(f"Record {record_id} does not provide two ECG leads.")
        signal = np.asarray(record.p_signal[:, :2], dtype=np.float32)
        annotation = wfdb.rdann(record_path, "atr")
        mapped = [ZHANG_LABEL_MAP.get(symbol) for symbol in annotation.symbol]
        valid_indices = [index for index, label in enumerate(mapped) if label in class_to_index]
        rpeaks = np.asarray(annotation.sample[valid_indices], dtype=np.int64)
        valid_labels = [mapped[index] for index in valid_indices]
        if len(rpeaks) < 2:
            continue
        global_rr, near_rr = rr_ratio_features(rpeaks)

        for index in range(1, len(rpeaks)):
            start, end = dynamic_beat_bounds(int(rpeaks[index - 1]), int(rpeaks[index]))
            if start < 0 or end > len(signal) or end <= start:
                continue
            segment = resample(signal[start:end], 128, axis=0).astype(np.float32)
            label = valid_labels[index]
            features.append(build_zhang_feature(segment, float(global_rr[index]), float(near_rr[index])))
            labels.append(class_to_index[label])
            counts[label] += 1

    if not features:
        raise ValueError("No Zhang-style beats were extracted. Check records and labels.")
    return (
        ZhangECGDataset(np.stack(features), np.asarray(labels)),
        ZhangSplitSummary(record_ids=resolved_ids, retained_beats=len(features), class_counts=dict(sorted(counts.items()))),
    )
