from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_waveform_plot(
    clean: np.ndarray,
    noisy: Optional[np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(clean, label="clean", linewidth=1.5)
    if noisy is not None:
        ax.plot(noisy, label="noisy", linewidth=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_annotated_ecg_window(
    signal: np.ndarray,
    sampling_rate: int,
    channel_name: str,
    record_id: str,
    start_sample: int,
    annotation_samples: np.ndarray,
    annotation_symbols: list[str],
    output_path: Path,
) -> None:
    """Save a labeled ECG window and any MIT-BIH annotations inside it."""
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D ECG signal, received shape {signal.shape}.")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    time_seconds = (np.arange(len(signal)) + start_sample) / sampling_rate
    start_time = start_sample / sampling_rate
    end_time = (start_sample + len(signal)) / sampling_rate

    fig, (ax, caption_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 6.4),
        gridspec_kw={"height_ratios": [5, 1]},
    )
    ax.plot(time_seconds, signal, color="#1b4965", linewidth=1.0, label=f"{channel_name} waveform")

    window_mask = (annotation_samples >= start_sample) & (annotation_samples < start_sample + len(signal))
    plotted_samples = annotation_samples[window_mask]
    plotted_symbols = [symbol for symbol, include in zip(annotation_symbols, window_mask) if include]
    beat_mask = np.asarray([symbol != "+" for symbol in plotted_symbols], dtype=bool)

    if beat_mask.any():
        beat_samples = plotted_samples[beat_mask]
        beat_values = signal[beat_samples - start_sample]
        ax.scatter(
            beat_samples / sampling_rate,
            beat_values,
            color="#d1495b",
            edgecolor="white",
            linewidth=0.5,
            s=38,
            zorder=3,
            label="Annotated beat",
        )
        for sample, symbol, value in zip(beat_samples, np.asarray(plotted_symbols)[beat_mask], beat_values):
            ax.annotate(
                symbol,
                xy=(sample / sampling_rate, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#8c2f39",
            )

    description = (
        f"Blue trace: physical ECG amplitude from MIT-BIH record {record_id}.\n"
        "Red markers and symbols: expert beat annotations inside this window.\n"
        "This is a clean source waveform before NSTDB noise is added."
    )
    caption_ax.text(
        0.01,
        0.5,
        description,
        transform=caption_ax.transAxes,
        va="center",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f4f8fb", "edgecolor": "#9fb3c8"},
    )
    ax.set_title(f"MIT-BIH Record {record_id}: {channel_name} ECG ({start_time:.1f}-{end_time:.1f} s)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("ECG amplitude (mV)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    caption_ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_noise_window(
    signal: np.ndarray,
    sampling_rate: int,
    noise_type: str,
    channel_name: str,
    start_sample: int,
    output_path: Path,
) -> None:
    """Save a labeled NSTDB noise waveform window as a PNG."""
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D noise signal, received shape {signal.shape}.")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    noise_names = {
        "bw": "Baseline Wander",
        "ma": "Muscle Artifact",
        "em": "Electrode Motion",
    }
    start_time = start_sample / sampling_rate
    end_time = (start_sample + len(signal)) / sampling_rate
    time_seconds = (np.arange(len(signal)) + start_sample) / sampling_rate

    fig, (ax, caption_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 6.0),
        gridspec_kw={"height_ratios": [5, 1]},
    )
    ax.plot(time_seconds, signal, color="#a23e48", linewidth=1.0, label=f"{noise_type} / {channel_name}")
    ax.set_title(f"NSTDB {noise_names.get(noise_type, noise_type)} ({start_time:.1f}-{end_time:.1f} s)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Noise amplitude (mV)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    caption_ax.text(
        0.01,
        0.5,
        f"Red trace: raw NSTDB {noise_names.get(noise_type, noise_type).lower()} recording.\n"
        "This standalone noise segment is sampled, scaled to a target SNR, and added to clean MIT-BIH ECG beats.",
        transform=caption_ax.transAxes,
        va="center",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff5f5", "edgecolor": "#d9a0a7"},
    )
    caption_ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_embedding_plot(
    embedding: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(6, 6))
    if embedding.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=8, alpha=0.7)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+tau)")
        ax.set_zlabel("x(t+2tau)")
    else:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=0.7)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+tau)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_diagram_plot(diagrams: dict[int, np.ndarray], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    finite_max = 1.0
    for dim, diagram in diagrams.items():
        if diagram.size == 0:
            continue
        finite = diagram[np.isfinite(diagram).all(axis=1)]
        if finite.size == 0:
            continue
        finite_max = max(finite_max, float(np.max(finite)))
        ax.scatter(finite[:, 0], finite[:, 1], s=16, alpha=0.8, label=f"H{dim}")
    ax.plot([0.0, finite_max], [0.0, finite_max], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(0.0, finite_max * 1.05)
    ax.set_ylim(0.0, finite_max * 1.05)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    if diagrams:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_persistence_image_plot(image: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(image, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_dindin_ph_comparison(
    waveform: np.ndarray,
    sampling_rate: int,
    central_offset: int,
    record_id: str,
    original_symbol: str,
    mapped_class: str,
    sublevel_diagram: np.ndarray,
    upper_diagram: np.ndarray,
    betti_curves: np.ndarray,
    output_path: Path,
) -> None:
    """Save a waveform, H0 barcodes, and Betti curves for one Dindin PH input."""
    if waveform.ndim != 1 or len(waveform) < 2:
        raise ValueError("Expected a one-dimensional ECG sequence with at least two samples.")
    if betti_curves.shape[0] != 2:
        raise ValueError(f"Expected sublevel/upper Betti curves, received {betti_curves.shape}.")
    if not 0 <= central_offset < len(waveform):
        raise ValueError("central_offset must identify a sample in waveform.")

    def finite_intervals(diagram: np.ndarray, endpoint: float) -> np.ndarray:
        intervals = diagram.astype(np.float32, copy=True)
        if len(intervals) == 0:
            return intervals
        intervals[~np.isfinite(intervals[:, 1]), 1] = endpoint
        intervals = intervals[np.isfinite(intervals).all(axis=1) & (intervals[:, 1] >= intervals[:, 0])]
        # Keep barcodes readable; the Betti curves below still use all intervals.
        order = np.argsort(intervals[:, 1] - intervals[:, 0])[::-1]
        return intervals[order[:25]]

    time = np.arange(len(waveform)) / sampling_rate
    figure = plt.figure(figsize=(14, 7.2))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.25, 1.0), height_ratios=(1.2, 1.0))
    waveform_ax = figure.add_subplot(grid[:, 0])
    barcode_ax = figure.add_subplot(grid[0, 1])
    betti_ax = figure.add_subplot(grid[1, 1])

    waveform_ax.plot(time, waveform, color="#1b4965", linewidth=1.35)
    waveform_ax.scatter(
        central_offset / sampling_rate,
        waveform[central_offset],
        color="#d1495b",
        edgecolor="white",
        linewidth=0.6,
        s=64,
        zorder=3,
        label=f"Central beat: {original_symbol} ({mapped_class})",
    )
    waveform_ax.axvline(central_offset / sampling_rate, color="#d1495b", linestyle="--", linewidth=0.9, alpha=0.8)
    waveform_ax.set_title("ECG sequence used as PH input", fontweight="bold")
    waveform_ax.set_xlabel("Time from preceding R peak (seconds)")
    waveform_ax.set_ylabel("ECG amplitude (mV)")
    waveform_ax.grid(alpha=0.25)
    waveform_ax.legend(loc="best", fontsize=9)

    sublevel = finite_intervals(sublevel_diagram, endpoint=1.0)
    upper = finite_intervals(upper_diagram, endpoint=0.0)
    barcode_rows: list[tuple[np.ndarray, str, str]] = [
        (sublevel, "#277da1", "Sublevel H0"),
        (upper, "#f8961e", "Upper-level H0 (computed on -ECG)"),
    ]
    row = 0
    for intervals, color, label in barcode_rows:
        for birth, death in intervals:
            barcode_ax.hlines(row, birth, death, color=color, linewidth=1.5)
            row += 1
        if len(intervals):
            barcode_ax.plot([], [], color=color, label=label)
        row += 1
    barcode_ax.set_title("GUDHI H0 barcodes (25 most persistent intervals)", fontweight="bold")
    barcode_ax.set_xlabel("Filtration value")
    barcode_ax.set_ylabel("Barcode interval")
    barcode_ax.set_yticks([])
    barcode_ax.grid(axis="x", alpha=0.25)
    barcode_ax.legend(loc="best", fontsize=8)

    curve_x = np.linspace(0.0, 1.0, betti_curves.shape[1])
    betti_ax.plot(curve_x, betti_curves[0], color="#277da1", linewidth=1.5, label="Sublevel Betti curve")
    betti_ax.plot(curve_x, betti_curves[1], color="#f8961e", linewidth=1.5, label="Upper-level Betti curve")
    betti_ax.set_title("Fixed-length PH representation", fontweight="bold")
    betti_ax.set_xlabel("Normalized filtration-grid position")
    betti_ax.set_ylabel("Alive H0 components")
    betti_ax.grid(alpha=0.25)
    betti_ax.legend(loc="best", fontsize=8)

    class_names = {
        "N": "Normal",
        "S": "Supraventricular",
        "V": "Ventricular",
        "F": "Fusion",
        "Q": "Unclassifiable / paced",
    }
    figure.suptitle(
        f"MIT-BIH Record {record_id}: original annotation '{original_symbol}'\n"
        f"AAMI CLASS {mapped_class}: {class_names.get(mapped_class, mapped_class)}",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Left: raw three-beat ECG context. Right: Dindin-style H0 PH after resampling and min-max normalization. "
        "The two Betti curves are the inputs to the PH CNN.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_dindin_noise_comparison(
    clean_waveform: np.ndarray,
    noisy_examples: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    sampling_rate: int,
    central_offset: int,
    record_id: str,
    original_symbol: str,
    mapped_class: str,
    clean_sublevel: np.ndarray,
    clean_upper: np.ndarray,
    clean_betti_curves: np.ndarray,
    output_path: Path,
) -> None:
    """Save a clean-vs-NSTDB Dindin PH comparison for one ECG context.

    ``noisy_examples`` maps a noise type to its waveform, sublevel H0 barcode,
    upper-level H0 barcode, Betti curves, and achieved SNR in dB.
    """
    if clean_waveform.ndim != 1 or len(clean_waveform) < 2:
        raise ValueError("Expected a one-dimensional clean ECG sequence with at least two samples.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if not 0 <= central_offset < len(clean_waveform):
        raise ValueError("central_offset must identify a sample in clean_waveform.")
    if clean_betti_curves.shape[0] != 2:
        raise ValueError("Expected two clean Betti curves: sublevel and upper-level.")
    if not noisy_examples:
        raise ValueError("At least one noisy example is required.")

    noise_names = {
        "bw": "Baseline wander",
        "ma": "Muscle artifact",
        "em": "Electrode motion",
    }
    class_names = {
        "N": "Normal",
        "S": "Supraventricular",
        "V": "Ventricular",
        "F": "Fusion",
        "Q": "Unclassifiable / paced",
    }

    def finite_intervals(diagram: np.ndarray, endpoint: float) -> np.ndarray:
        intervals = diagram.astype(np.float32, copy=True)
        if len(intervals) == 0:
            return intervals
        intervals[~np.isfinite(intervals[:, 1]), 1] = endpoint
        intervals = intervals[np.isfinite(intervals).all(axis=1) & (intervals[:, 1] >= intervals[:, 0])]
        return intervals[np.argsort(intervals[:, 1] - intervals[:, 0])[::-1][:25]]

    columns = [("clean", clean_waveform, clean_sublevel, clean_upper, clean_betti_curves, None)]
    columns.extend((noise_type, *values) for noise_type, values in noisy_examples.items())
    figure, axes = plt.subplots(3, len(columns), figsize=(5.2 * len(columns), 10.2), squeeze=False)
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.84, wspace=0.3, hspace=0.48)

    for column_index, (noise_type, waveform, sublevel_diagram, upper_diagram, curves, achieved_snr) in enumerate(columns):
        if waveform.ndim != 1 or len(waveform) != len(clean_waveform):
            raise ValueError("Every noisy waveform must have the same one-dimensional shape as the clean waveform.")
        if curves.shape[0] != 2:
            raise ValueError("Expected sublevel and upper-level Betti curves for every column.")
        time = np.arange(len(waveform)) / sampling_rate
        waveform_ax, barcode_ax, betti_ax = axes[:, column_index]
        color = "#1b4965" if noise_type == "clean" else "#a23e48"
        waveform_ax.plot(time, waveform, color=color, linewidth=1.15)
        waveform_ax.scatter(
            central_offset / sampling_rate,
            waveform[central_offset],
            color="#d1495b",
            edgecolor="white",
            linewidth=0.55,
            s=45,
            zorder=3,
        )
        waveform_ax.axvline(central_offset / sampling_rate, color="#d1495b", linestyle="--", linewidth=0.8, alpha=0.8)
        if noise_type == "clean":
            waveform_ax.set_title("Clean MIT-BIH ECG", fontweight="bold")
        else:
            waveform_ax.set_title(f"{noise_names.get(noise_type, noise_type)}\n{achieved_snr:.1f} dB achieved SNR", fontweight="bold")
        waveform_ax.set_xlabel("Time from preceding R peak (s)")
        waveform_ax.set_ylabel("ECG amplitude (mV)" if column_index == 0 else "")
        waveform_ax.grid(alpha=0.25)

        row = 0
        for intervals, color, label, endpoint in (
            (sublevel_diagram, "#277da1", "Sublevel H0", 1.0),
            (upper_diagram, "#f8961e", "Upper-level H0", 0.0),
        ):
            for birth, death in finite_intervals(intervals, endpoint):
                barcode_ax.hlines(row, birth, death, color=color, linewidth=1.25)
                row += 1
            barcode_ax.plot([], [], color=color, label=label)
            row += 1
        barcode_ax.set_title("H0 persistence barcodes", fontweight="bold")
        barcode_ax.set_xlabel("Filtration value")
        barcode_ax.set_ylabel("Interval" if column_index == 0 else "")
        barcode_ax.set_yticks([])
        barcode_ax.grid(axis="x", alpha=0.25)
        barcode_ax.legend(loc="best", fontsize=7.5)

        curve_x = np.linspace(0.0, 1.0, curves.shape[1])
        betti_ax.plot(curve_x, curves[0], color="#277da1", linewidth=1.35, label="Sublevel")
        betti_ax.plot(curve_x, curves[1], color="#f8961e", linewidth=1.35, label="Upper-level")
        betti_ax.set_title(f"Fixed-length PH: 2 x {curves.shape[1]}", fontweight="bold")
        betti_ax.set_xlabel("Normalized filtration position")
        betti_ax.set_ylabel("Alive H0 components" if column_index == 0 else "")
        betti_ax.grid(alpha=0.25)
        betti_ax.legend(loc="best", fontsize=7.5)

    figure.suptitle(
        f"Clean vs. NSTDB Noise: MIT-BIH Record {record_id}, annotation '{original_symbol}'\n"
        f"AAMI CLASS {mapped_class}: {class_names.get(mapped_class, mapped_class)}",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Each column uses the same three-beat ECG context. PH is direct 1D H0: sublevel filtration captures valleys; "
        "upper-level filtration is computed on -ECG and captures peaks. Barcodes display the 25 most persistent intervals per filtration.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
