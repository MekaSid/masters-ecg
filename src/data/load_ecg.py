from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


@dataclass
class ECGRecord:
    record_id: str
    signal: np.ndarray
    fs: int
    sig_name: list[str]


@dataclass
class ECGAnnotations:
    record_id: str
    samples: np.ndarray
    symbols: list[str]


def _record_path(base_dir: Path, record_id: str) -> str:
    return str(base_dir / record_id)


def load_record(base_dir: Path, record_id: str, channel: int = 0) -> ECGRecord:
    record = wfdb.rdrecord(_record_path(base_dir, record_id))
    if channel >= record.p_signal.shape[1]:
        raise ValueError(f"Requested channel {channel} but record {record_id} has {record.p_signal.shape[1]} channels.")

    signal = np.asarray(record.p_signal[:, channel], dtype=np.float32)
    return ECGRecord(record_id=record_id, signal=signal, fs=int(record.fs), sig_name=list(record.sig_name))


def load_annotations(base_dir: Path, record_id: str, extension: str = "atr") -> ECGAnnotations:
    annotation = wfdb.rdann(_record_path(base_dir, record_id), extension)
    return ECGAnnotations(
        record_id=record_id,
        samples=np.asarray(annotation.sample, dtype=np.int64),
        symbols=list(annotation.symbol),
    )


def load_noise_record(base_dir: Path, noise_type: str, channel: int = 0) -> ECGRecord:
    return load_record(base_dir=base_dir, record_id=noise_type, channel=channel)
