"""Generate clean-versus-NSTDB Dindin PH figures for the selected ECG gallery."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.noise import NoiseAugmentor
from src.tda.betti import BettiCurveConfig, dindin_betti_curves, normalize_unit_interval
from src.tda.persistence import compute_sublevel_persistence
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.plotting import save_dindin_noise_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean-vs-noisy Dindin PH gallery PNGs.")
    parser.add_argument("--selection-csv", type=Path, default=PROJECT_ROOT / "results/ph_gallery/selection.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/ph_noise_gallery")
    parser.add_argument("--snr-db", type=float, default=12.0, help="Target SNR in dB; 12 dB is a moderate corruption.")
    parser.add_argument("--noise-types", nargs="+", choices=["bw", "ma", "em"], default=["bw", "ma", "em"])
    parser.add_argument("--noise-channel", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N selected examples.")
    return parser.parse_args()


def ph_components(waveform: np.ndarray, sequence_length: int, betti_resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the direct-1D H0 barcodes and fixed-length Betti input for one waveform."""
    prepared = normalize_unit_interval(resample(waveform, sequence_length).astype(np.float32))
    sublevel = compute_sublevel_persistence(prepared).diagrams[0]
    upper = compute_sublevel_persistence(-prepared).diagrams[0]
    curves = dindin_betti_curves(waveform, BettiCurveConfig(resolution=betti_resolution))
    return sublevel, upper, curves


def main() -> None:
    args = parse_args()
    if not args.selection_csv.is_file():
        raise FileNotFoundError(f"Selection CSV not found: {args.selection_csv}. Run generate_ph_gallery.py first.")
    if not np.isfinite(args.snr_db):
        raise ValueError("snr-db must be finite.")
    output_dir = ensure_dir(args.output_dir)
    mitdb_dir = PROJECT_ROOT / "data/raw/mitdb"
    nstdb_dir = PROJECT_ROOT / "data/raw/nstdb"
    noise_signals = {
        noise_type: np.asarray(wfdb.rdrecord(str(nstdb_dir / noise_type)).p_signal[:, args.noise_channel], dtype=np.float32)
        for noise_type in args.noise_types
    }
    if not all((nstdb_dir / f"{noise_type}.hea").is_file() for noise_type in args.noise_types):
        raise FileNotFoundError(f"NSTDB bw/ma/em records are required in {nstdb_dir}.")

    with args.selection_csv.open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No gallery examples selected.")

    output_metadata: list[dict[str, str]] = []
    for row in selected:
        record_id = row["record_id"]
        previous_rpeak = int(row["previous_rpeak"])
        central_rpeak = int(row["central_rpeak"])
        following_rpeak = int(row["following_rpeak"])
        sequence_length = int(row["ph_resampled_length"])
        betti_resolution = int(row["betti_resolution"])
        record = wfdb.rdrecord(str(mitdb_dir / record_id))
        if int(record.fs) != 360:
            raise ValueError(f"Expected MIT-BIH sampling rate 360 Hz, got {record.fs} for record {record_id}.")
        clean = np.asarray(record.p_signal[previous_rpeak:following_rpeak, 0], dtype=np.float32)
        central_offset = central_rpeak - previous_rpeak
        clean_sublevel, clean_upper, clean_curves = ph_components(clean, sequence_length, betti_resolution)
        noisy_examples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
        starts: dict[str, int] = {}
        for noise_index, noise_type in enumerate(args.noise_types):
            augmentor = NoiseAugmentor(seed=args.seed + int(row["order"]) * 100 + noise_index)
            result = augmentor.add_noise(clean, noise_signals[noise_type], noise_type, args.snr_db)
            sublevel, upper, curves = ph_components(result.noisy, sequence_length, betti_resolution)
            noisy_examples[noise_type] = (result.noisy, sublevel, upper, curves, result.snr_db_achieved)
            starts[noise_type] = result.start_index
        output_path = output_dir / f"{int(row['order']):02d}_{row['original_symbol']}_record_{record_id}_clean_vs_noise_ph.png"
        save_dindin_noise_comparison(
            clean_waveform=clean,
            noisy_examples=noisy_examples,
            sampling_rate=int(record.fs),
            central_offset=central_offset,
            record_id=record_id,
            original_symbol=row["original_symbol"],
            mapped_class=row["mapped_class"],
            clean_sublevel=clean_sublevel,
            clean_upper=clean_upper,
            clean_betti_curves=clean_curves,
            output_path=output_path,
        )
        output_metadata.append({
            **row,
            "target_snr_db": f"{args.snr_db:.1f}",
            **{f"{noise_type}_start_index": str(starts[noise_type]) for noise_type in args.noise_types},
            "png": output_path.name,
        })
        print(f"Saved {output_path.name}")
    with (output_dir / "selection_and_noise_segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_metadata[0]))
        writer.writeheader()
        writer.writerows(output_metadata)
    print(f"Saved {len(output_metadata)} clean-vs-noise PH PNGs to {output_dir}")


if __name__ == "__main__":
    main()
