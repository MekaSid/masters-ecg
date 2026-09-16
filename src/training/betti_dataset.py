"""Cached Dindin-style PH features aligned with Zhang raw/RR examples."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import wfdb
from joblib import Parallel, delayed
from scipy.signal import resample
from torch.utils.data import Dataset

from src.tda.betti import BettiCurveConfig, dindin_betti_curves
from src.training.zhang_dataset import ZHANG_LABEL_MAP, build_zhang_feature, dynamic_beat_bounds, rr_ratio_features


@dataclass(frozen=True)
class BettiSplitSummary:
    record_ids: tuple[str, ...]
    retained_beats: int
    class_counts: dict[str, int]


class BettiFusionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Paired Zhang raw/RR tensors and two-channel Dindin Betti curves."""

    def __init__(self, raw_features: np.ndarray, betti_features: np.ndarray, labels: np.ndarray) -> None:
        if raw_features.ndim != 4 or raw_features.shape[1:] != (2, 3, 128):
            raise ValueError(f"Expected raw features shaped (beats, 2, 3, 128), got {raw_features.shape}.")
        if betti_features.ndim != 3 or betti_features.shape[1] != 2:
            raise ValueError(f"Expected Betti features shaped (beats, 2, resolution), got {betti_features.shape}.")
        if not (len(raw_features) == len(betti_features) == len(labels)):
            raise ValueError("Raw features, Betti curves, and labels must have matching lengths.")
        self.raw_features = torch.from_numpy(raw_features.astype(np.float32, copy=False))
        self.betti_features = torch.from_numpy(betti_features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.raw_features[index], self.betti_features[index], self.labels[index]


def _compute_curve(signal: np.ndarray, config: BettiCurveConfig) -> np.ndarray:
    return dindin_betti_curves(signal, config)


def build_betti_fusion_split(
    record_ids: Iterable[str],
    mitdb_dir: Path,
    class_to_index: dict[str, int],
    betti_config: BettiCurveConfig,
    ph_sequence_length: int,
    workers: int = 1,
) -> tuple[BettiFusionDataset, BettiSplitSummary]:
    """Build paired raw/RR and three-beat Betti inputs from official annotations.

    Each PH signal runs from the preceding R peak to the following R peak; its
    central annotated beat supplies the AAMI label and the Zhang raw/RR input.
    """
    if ph_sequence_length < 8:
        raise ValueError("ph_sequence_length must be at least eight.")
    raw_features: list[np.ndarray] = []
    ph_sequences: list[np.ndarray] = []
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
        if len(rpeaks) < 3:
            continue
        global_rr, near_rr = rr_ratio_features(rpeaks)
        for index in range(1, len(rpeaks) - 1):
            raw_start, raw_end = dynamic_beat_bounds(int(rpeaks[index - 1]), int(rpeaks[index]))
            ph_start, ph_end = int(rpeaks[index - 1]), int(rpeaks[index + 1])
            if raw_start < 0 or raw_end > len(signal) or raw_end <= raw_start or ph_end > len(signal):
                continue
            raw_segment = resample(signal[raw_start:raw_end], 128, axis=0).astype(np.float32)
            ph_sequence = resample(signal[ph_start:ph_end, 0], ph_sequence_length).astype(np.float32)
            label = valid_labels[index]
            raw_features.append(build_zhang_feature(raw_segment, float(global_rr[index]), float(near_rr[index])))
            ph_sequences.append(ph_sequence)
            labels.append(class_to_index[label])
            counts[label] += 1

    if not raw_features:
        raise ValueError("No paired raw/Betti examples were extracted.")
    curves = Parallel(n_jobs=workers, prefer="threads")(
        delayed(_compute_curve)(sequence, betti_config) for sequence in ph_sequences
    )
    return (
        BettiFusionDataset(np.stack(raw_features), np.stack(curves), np.asarray(labels)),
        BettiSplitSummary(record_ids=resolved_ids, retained_beats=len(labels), class_counts=dict(sorted(counts.items()))),
    )


def load_or_build_betti_split(
    cache_path: Path,
    record_ids: Iterable[str],
    mitdb_dir: Path,
    class_to_index: dict[str, int],
    betti_config: BettiCurveConfig,
    ph_sequence_length: int,
    workers: int,
) -> tuple[BettiFusionDataset, BettiSplitSummary]:
    """Load a cache when its schema matches, otherwise build and save it."""
    resolved_ids = tuple(str(record_id) for record_id in record_ids)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            tuple(cached["record_ids"].tolist()) == resolved_ids
            and int(cached["resolution"]) == betti_config.resolution
            and int(cached["sequence_length"]) == ph_sequence_length
        ):
            summary = BettiSplitSummary(
                record_ids=resolved_ids,
                retained_beats=len(cached["labels"]),
                class_counts={name: int(count) for name, count in zip(cached["class_names"].tolist(), cached["class_counts"].tolist())},
            )
            return BettiFusionDataset(cached["raw_features"], cached["betti_features"], cached["labels"]), summary

    dataset, summary = build_betti_fusion_split(
        resolved_ids, mitdb_dir, class_to_index, betti_config, ph_sequence_length, workers
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    names = np.asarray(list(summary.class_counts), dtype=str)
    np.savez_compressed(
        cache_path,
        raw_features=dataset.raw_features.numpy(),
        betti_features=dataset.betti_features.numpy(),
        labels=dataset.labels.numpy(),
        record_ids=np.asarray(resolved_ids, dtype=str),
        class_names=names,
        class_counts=np.asarray([summary.class_counts[name] for name in names], dtype=np.int64),
        resolution=np.asarray(betti_config.resolution),
        sequence_length=np.asarray(ph_sequence_length),
    )
    return dataset, summary
