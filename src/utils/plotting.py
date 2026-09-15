from __future__ import annotations

from pathlib import Path
from typing import Optional

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
