"""Generate diverse ECG-to-Dindin-PH comparison figures from MIT-BIH annotations."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tda.betti import BettiCurveConfig, dindin_betti_curves, normalize_unit_interval
from src.tda.persistence import compute_sublevel_persistence
from src.training.zhang_dataset import ZHANG_LABEL_MAP
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.plotting import save_dindin_ph_comparison


DEFAULT_SYMBOLS = ["A", "V", "F", "a", "J", "E", "e", "j", "Q", "f"]


@dataclass(frozen=True)
class GalleryExample:
    record_id: str
    central_index: int
    original_symbol: str
    mapped_class: str
    previous_rpeak: int
    central_rpeak: int
    following_rpeak: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create diverse ECG and Dindin-PH comparison PNGs.")
    parser.add_argument("--count", type=int, default=10, help="Maximum examples to generate.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Preferred original MIT-BIH symbols, in selection order.")
    parser.add_argument("--sequence-length", type=int, default=256, help="Resampled three-beat length used for PH.")
    parser.add_argument("--betti-resolution", type=int, default=128, help="Betti-curve bins.")
    parser.add_argument("--min-rr-seconds", type=float, default=0.15, help="Minimum adjacent RR interval accepted for a gallery context.")
    parser.add_argument("--max-rr-seconds", type=float, default=3.0, help="Maximum adjacent RR interval accepted for a gallery context.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNGs and selection CSV.")
    return parser.parse_args()


def find_example(
    record_paths: list[Path],
    symbol: str,
    used_records: set[str],
    min_rr_seconds: float,
    max_rr_seconds: float,
) -> GalleryExample | None:
    """Find a valid central beat, preferring a record not already in the gallery."""
    fallback: GalleryExample | None = None
    for record_path in record_paths:
        record_id = record_path.stem
        sampling_rate = float(wfdb.rdheader(str(record_path.with_suffix(""))).fs)
        annotation = wfdb.rdann(str(record_path.with_suffix("")), "atr")
        valid_indices = [index for index, value in enumerate(annotation.symbol) if value in ZHANG_LABEL_MAP]
        for central_index in range(1, len(valid_indices) - 1):
            annotation_index = valid_indices[central_index]
            if annotation.symbol[annotation_index] != symbol:
                continue
            previous_rpeak = int(annotation.sample[valid_indices[central_index - 1]])
            central_rpeak = int(annotation.sample[annotation_index])
            following_rpeak = int(annotation.sample[valid_indices[central_index + 1]])
            rr_intervals = ((central_rpeak - previous_rpeak) / sampling_rate, (following_rpeak - central_rpeak) / sampling_rate)
            if not all(min_rr_seconds <= interval <= max_rr_seconds for interval in rr_intervals):
                continue
            candidate = GalleryExample(
                record_id=record_id,
                central_index=central_index,
                original_symbol=symbol,
                mapped_class=ZHANG_LABEL_MAP[symbol],
                previous_rpeak=previous_rpeak,
                central_rpeak=central_rpeak,
                following_rpeak=following_rpeak,
            )
            if record_id not in used_records:
                return candidate
            fallback = fallback or candidate
    return fallback


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.sequence_length < 8 or args.betti_resolution < 2 or not 0 < args.min_rr_seconds < args.max_rr_seconds:
        raise ValueError("Require positive count, sequence length >= 8, Betti resolution >= 2, and valid RR bounds.")
    mitdb_dir = PROJECT_ROOT / "data/raw/mitdb"
    record_paths = sorted(mitdb_dir.glob("[0-9][0-9][0-9].hea"))
    if not record_paths:
        raise FileNotFoundError(f"No MIT-BIH headers found in {mitdb_dir}.")
    output_dir = args.output_dir or ensure_dir(PROJECT_ROOT / "results" / "ph_gallery")
    used_records: set[str] = set()
    examples: list[GalleryExample] = []
    for symbol in args.symbols:
        if len(examples) >= args.count:
            break
        if symbol not in ZHANG_LABEL_MAP:
            print(f"Skipping unsupported symbol: {symbol}")
            continue
        example = find_example(record_paths, symbol, used_records, args.min_rr_seconds, args.max_rr_seconds)
        if example is None:
            print(f"No valid three-beat example found for symbol: {symbol}")
            continue
        examples.append(example)
        used_records.add(example.record_id)
    if not examples:
        raise RuntimeError("No gallery examples could be selected.")

    metadata: list[dict[str, int | str]] = []
    betti_config = BettiCurveConfig(resolution=args.betti_resolution)
    for order, example in enumerate(examples, start=1):
        record = wfdb.rdrecord(str(mitdb_dir / example.record_id))
        signal = np.asarray(record.p_signal[:, 0], dtype=np.float32)
        sequence = signal[example.previous_rpeak : example.following_rpeak]
        resampled = resample(sequence, args.sequence_length).astype(np.float32)
        normalized = normalize_unit_interval(resampled)
        sublevel = compute_sublevel_persistence(normalized).diagrams[0]
        upper = compute_sublevel_persistence(-normalized).diagrams[0]
        curves = dindin_betti_curves(resampled, betti_config)
        output_path = output_dir / f"{order:02d}_{example.original_symbol}_record_{example.record_id}_ph.png"
        save_dindin_ph_comparison(
            waveform=sequence,
            sampling_rate=int(record.fs),
            central_offset=example.central_rpeak - example.previous_rpeak,
            record_id=example.record_id,
            original_symbol=example.original_symbol,
            mapped_class=example.mapped_class,
            sublevel_diagram=sublevel,
            upper_diagram=upper,
            betti_curves=curves,
            output_path=output_path,
        )
        metadata.append(
            {
                "order": order,
                "record_id": example.record_id,
                "original_symbol": example.original_symbol,
                "mapped_class": example.mapped_class,
                "previous_rpeak": example.previous_rpeak,
                "central_rpeak": example.central_rpeak,
                "following_rpeak": example.following_rpeak,
                "ph_resampled_length": args.sequence_length,
                "betti_resolution": args.betti_resolution,
                "png": output_path.name,
            }
        )
    with (output_dir / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Saved {len(metadata)} ECG-to-PH comparison PNGs and selection.csv to {output_dir}")


if __name__ == "__main__":
    main()
