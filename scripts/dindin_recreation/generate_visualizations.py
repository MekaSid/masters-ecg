"""Recreate Dindin-style direct-1D PH visualizations for clean and noisy ECG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import resample

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.noise import NoiseAugmentor
from src.tda.betti import BettiCurveConfig, dindin_betti_curves, normalize_unit_interval
from src.tda.persistence import compute_sublevel_persistence
from src.training.zhang_dataset import ZHANG_LABEL_MAP
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir


NOISE_NAMES = {"bw": "Baseline wander", "ma": "Muscle artifact", "em": "Electrode motion"}
CLASS_NAMES = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular", "F": "Fusion", "Q": "Unclassifiable / paced"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Dindin PH clean/noise explanation figures.")
    parser.add_argument("--record-id", default="100")
    parser.add_argument("--symbol", default="A", help="Central original MIT-BIH beat symbol.")
    parser.add_argument("--occurrence", type=int, default=0, help="Zero-based matching beat occurrence in the record.")
    parser.add_argument("--snr-db", type=float, default=12.0, help="Noise level applied separately for BW, MA, and EM.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/dindin_recreation")
    return parser.parse_args()


def choose_three_beat_context(record_id: str, symbol: str, occurrence: int) -> tuple[np.ndarray, int, int, list[int], str]:
    """Return lead-I previous/current/following context centered on a selected symbol."""
    record_path = PROJECT_ROOT / "data/raw/mitdb" / record_id
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")
    indices = [index for index, value in enumerate(annotation.symbol) if value == symbol]
    if occurrence < 0 or occurrence >= len(indices):
        raise ValueError(f"Record {record_id} has {len(indices)} occurrences of symbol {symbol!r}; requested {occurrence}.")
    annotation_index = indices[occurrence]
    valid_indices = [index for index, value in enumerate(annotation.symbol) if value in ZHANG_LABEL_MAP]
    position = valid_indices.index(annotation_index)
    if position == 0 or position == len(valid_indices) - 1:
        raise ValueError("Selected beat needs both a preceding and following mapped beat.")
    previous, central, following = (int(annotation.sample[valid_indices[position + delta]]) for delta in (-1, 0, 1))
    signal = np.asarray(record.p_signal[previous:following, 0], dtype=np.float32)
    return signal, central - previous, int(record.fs), [0, central - previous, following - previous], ZHANG_LABEL_MAP[symbol]


def finite_intervals(diagram: np.ndarray, endpoint: float) -> np.ndarray:
    intervals = diagram.astype(np.float32, copy=True)
    intervals[~np.isfinite(intervals[:, 1]), 1] = endpoint
    valid = np.isfinite(intervals).all(axis=1) & (intervals[:, 1] >= intervals[:, 0])
    intervals = intervals[valid]
    return intervals[np.argsort(intervals[:, 1] - intervals[:, 0])[::-1][:25]]


def components(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample, normalize, and calculate Dindin's two H0 barcode channels."""
    resampled = resample(signal, 256).astype(np.float32)
    normalized = normalize_unit_interval(resampled)
    sublevel = compute_sublevel_persistence(normalized).diagrams[0]
    upper = compute_sublevel_persistence(-normalized).diagrams[0]
    curves = dindin_betti_curves(resampled, BettiCurveConfig(resolution=128))
    return normalized, sublevel, upper, curves


def plot_example(
    waveform: np.ndarray,
    sampling_rate: int,
    r_offsets: list[int],
    normalized: np.ndarray,
    sublevel: np.ndarray,
    upper: np.ndarray,
    curves: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Plot three ECG beats, filtration snapshot, barcodes, and Betti curves."""
    figure = plt.figure(figsize=(15, 11))
    grid = figure.add_gridspec(3, 2, height_ratios=(1.15, 1.05, 1.0), hspace=0.45, wspace=0.28)
    waveform_ax = figure.add_subplot(grid[0, :])
    threshold_ax = figure.add_subplot(grid[1, 0])
    barcode_ax = figure.add_subplot(grid[1, 1])
    betti_ax = figure.add_subplot(grid[2, :])

    time = np.arange(len(waveform)) / sampling_rate
    regions = [(r_offsets[0], r_offsets[1], "Previous beat", "#e7f0f6"), (r_offsets[1], r_offsets[2], "Central labeled beat", "#fce8e6")]
    for start, end, label, color in regions:
        waveform_ax.axvspan(start / sampling_rate, end / sampling_rate, color=color, alpha=0.8, label=label)
    waveform_ax.plot(time, waveform, color="#173f5f", linewidth=1.25, zorder=2)
    waveform_ax.scatter(r_offsets[1] / sampling_rate, waveform[r_offsets[1]], color="#c44536", edgecolor="white", s=65, zorder=3, label="Central R peak")
    waveform_ax.axvline(r_offsets[1] / sampling_rate, color="#c44536", linestyle="--", linewidth=0.9)
    waveform_ax.set_title("Input to Dindin PH: previous + central + following ECG beats", fontweight="bold")
    waveform_ax.set_xlabel("Time from preceding R peak (seconds)")
    waveform_ax.set_ylabel("ECG amplitude (mV)")
    waveform_ax.grid(alpha=0.25)
    waveform_ax.legend(loc="best", ncol=3, fontsize=8)

    filtration_x = np.linspace(0, 1, len(normalized))
    alpha = 0.50
    threshold_ax.plot(filtration_x, normalized, color="#173f5f", linewidth=1.2)
    threshold_ax.axhline(alpha, color="#c44536", linestyle="--", label=f"Threshold alpha = {alpha:.2f}")
    threshold_ax.fill_between(filtration_x, normalized, alpha, where=normalized <= alpha, color="#277da1", alpha=0.45, label="F_alpha = {t : f(t) <= alpha}")
    threshold_ax.set_title("Sublevel filtration snapshot", fontweight="bold")
    threshold_ax.set_xlabel("Resampled time position (not Betti axis)")
    threshold_ax.set_ylabel("Normalized ECG amplitude")
    threshold_ax.set_ylim(-0.05, 1.05)
    threshold_ax.grid(alpha=0.25)
    threshold_ax.legend(fontsize=7.5, loc="best")

    row = 0
    for diagram, color, label, endpoint in ((sublevel, "#277da1", "Sublevel H0: valleys", 1.0), (upper, "#f8961e", "Upper-level H0: peaks, computed on -ECG", 0.0)):
        for birth, death in finite_intervals(diagram, endpoint):
            barcode_ax.hlines(row, birth, death, color=color, linewidth=1.3)
            row += 1
        barcode_ax.plot([], [], color=color, label=label)
        row += 1
    barcode_ax.set_title("Persistence barcodes: birth -> merge/death threshold", fontweight="bold")
    barcode_ax.set_xlabel("Filtration value")
    barcode_ax.set_ylabel("H0 interval")
    barcode_ax.set_yticks([])
    barcode_ax.grid(axis="x", alpha=0.25)
    barcode_ax.legend(fontsize=7.5, loc="best")

    thresholds = np.linspace(0, 1, curves.shape[1])
    betti_ax.plot(thresholds, curves[0], color="#277da1", linewidth=1.7, label="Row 1: sublevel Betti curve")
    betti_ax.plot(thresholds, curves[1], color="#f8961e", linewidth=1.7, label="Row 2: upper-level Betti curve")
    betti_ax.set_title("Fixed Dindin PH input to the 1D CNN: shape (2, 128)", fontweight="bold")
    betti_ax.set_xlabel("128 evenly spaced filtration thresholds (not time)")
    betti_ax.set_ylabel("Number of alive H0 intervals")
    betti_ax.grid(alpha=0.25)
    betti_ax.legend(loc="best")

    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.text(0.5, 0.012, "At each filtration threshold, the Betti value counts barcode intervals alive at that threshold. "
                "Barcodes shown are the 25 most persistent intervals; Betti curves use every valid interval.", ha="center", fontsize=8.5)
    figure.subplots_adjust(bottom=0.08, top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_clean_noisy_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    sampling_rate: int,
    r_offsets: list[int],
    clean_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    noisy_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    """Save the requested clean/noisy ECG, barcode, and Betti-curve comparison."""
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    colors = {"clean": "#173f5f", "noisy": "#c44536"}

    def waveform(axis: plt.Axes, signal: np.ndarray, label: str) -> None:
        time = np.arange(len(signal)) / sampling_rate
        axis.axvspan(r_offsets[0] / sampling_rate, r_offsets[1] / sampling_rate, color="#e7f0f6", alpha=0.8, label="Previous -> central RR interval")
        axis.axvspan(r_offsets[1] / sampling_rate, r_offsets[2] / sampling_rate, color="#fce8e6", alpha=0.8, label="Central -> following RR interval")
        axis.plot(time, signal, color=colors[label], linewidth=1.2)
        peak_indices = [0, r_offsets[1], len(signal) - 1]
        axis.scatter(
            [index / sampling_rate for index in (r_offsets[0], r_offsets[1], r_offsets[2])],
            [signal[index] for index in peak_indices],
            color="#c44536", edgecolor="white", s=55, zorder=3, label="Previous / central / following R peaks",
        )
        axis.axvline(r_offsets[1] / sampling_rate, color="#c44536", linestyle="--", linewidth=0.8)
        axis.set_title(f"{label.title()} ECG: three-beat PH input", fontweight="bold")
        axis.set_xlabel("Time from preceding R peak (seconds)")
        axis.set_ylabel("ECG amplitude (mV)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")

    def barcode(axis: plt.Axes, sublevel: np.ndarray, upper: np.ndarray, label: str) -> None:
        row = 0
        for diagram, color, name, endpoint in ((sublevel, "#277da1", "Sublevel H0", 1.0), (upper, "#f8961e", "Upper-level H0 on -ECG", 0.0)):
            for birth, death in finite_intervals(diagram, endpoint):
                axis.hlines(row, birth, death, color=color, linewidth=1.2)
                row += 1
            axis.plot([], [], color=color, label=name)
            row += 1
        axis.set_title(f"{label.title()} persistence barcodes", fontweight="bold")
        axis.set_xlabel("Filtration threshold")
        axis.set_ylabel("H0 interval")
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.25)
        axis.legend(fontsize=7, loc="best")

    def betti(axis: plt.Axes, curves: np.ndarray, label: str) -> None:
        threshold = np.linspace(0, 1, curves.shape[1])
        axis.plot(threshold, curves[0], color="#277da1", linewidth=1.6, label="Sublevel: valleys")
        axis.plot(threshold, curves[1], color="#f8961e", linewidth=1.6, label="Upper-level: peaks")
        axis.set_title(f"{label.title()} Betti tensor: shape (2, 128)", fontweight="bold")
        axis.set_xlabel("128 filtration thresholds (not time)")
        axis.set_ylabel("Alive H0 intervals")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")

    waveform(axes[0, 0], clean, "clean")
    waveform(axes[0, 1], noisy, "noisy")
    _, clean_sublevel, clean_upper, clean_curves = clean_components
    _, noisy_sublevel, noisy_upper, noisy_curves = noisy_components
    barcode(axes[1, 0], clean_sublevel, clean_upper, "clean")
    barcode(axes[1, 1], noisy_sublevel, noisy_upper, "noisy")
    betti(axes[2, 0], clean_curves, "clean")
    betti(axes[2, 1], noisy_curves, "noisy")
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.text(0.5, 0.012, "Dindin direct-1D PH: each Betti value counts barcode intervals alive at a filtration threshold. "
                "Only the 25 longest intervals are drawn; all valid intervals form the 2 x 128 model input.", ha="center", fontsize=8.5)
    figure.subplots_adjust(bottom=0.08, top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.snr_db):
        raise ValueError("snr-db must be finite.")
    waveform, central_offset, sampling_rate, r_offsets, mapped_class = choose_three_beat_context(args.record_id, args.symbol, args.occurrence)
    output_dir = ensure_dir(args.output_dir)
    clean_components = components(waveform)
    normalized, sublevel, upper, curves = clean_components
    base_title = f"Dindin Direct-1D PH Recreation | MIT-BIH {args.record_id}, annotation '{args.symbol}' -> AAMI {mapped_class} ({CLASS_NAMES[mapped_class]})"
    plot_example(waveform, sampling_rate, r_offsets, normalized, sublevel, upper, curves, f"{base_title} | Clean ECG", output_dir / "01_clean_dindin_ph.png")

    metadata: dict[str, object] = {"record_id": args.record_id, "original_symbol": args.symbol, "mapped_class": mapped_class, "snr_db_target": args.snr_db, "noise": {}}
    nstdb_dir = PROJECT_ROOT / "data/raw/nstdb"
    for index, noise_type in enumerate(("bw", "ma", "em")):
        noise = np.asarray(wfdb.rdrecord(str(nstdb_dir / noise_type)).p_signal[:, 0], dtype=np.float32)
        result = NoiseAugmentor(seed=args.seed + index).add_noise(waveform, noise, noise_type, args.snr_db)
        noisy_components = components(result.noisy)
        normalized, sublevel, upper, curves = noisy_components
        plot_example(result.noisy, sampling_rate, r_offsets, normalized, sublevel, upper, curves,
                     f"{base_title} | {NOISE_NAMES[noise_type]} at {result.snr_db_achieved:.1f} dB", output_dir / f"0{index + 2}_{noise_type}_{int(args.snr_db)}db_dindin_ph.png")
        plot_clean_noisy_comparison(
            waveform, result.noisy, sampling_rate, r_offsets, clean_components, noisy_components,
            f"Clean vs. {NOISE_NAMES[noise_type]} at {result.snr_db_achieved:.1f} dB | {base_title}",
            output_dir / f"comparison_{noise_type}_{int(args.snr_db)}db.png",
        )
        metadata["noise"][noise_type] = {"start_index": result.start_index, "snr_db_achieved": result.snr_db_achieved}
    with (output_dir / "recreation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved clean and three 12 dB Dindin PH explanation figures to {output_dir}")


if __name__ == "__main__":
    main()
