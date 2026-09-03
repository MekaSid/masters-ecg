from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams


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


def save_diagram_plot(diagrams: list[np.ndarray], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_diagrams(diagrams, ax=ax, title=title, show=False)
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
