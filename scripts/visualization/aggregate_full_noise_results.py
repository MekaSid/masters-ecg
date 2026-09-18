"""Aggregate three full-DS2 noise evaluations and create uncertainty plots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir


CONDITIONS = {"bw": "Baseline Wander (BW)", "ma": "Muscle Artifact (MA)", "em": "Electrode Motion (EM)", "mix": "Combined BW + MA + EM"}
CLASS_NAMES = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular", "F": "Fusion", "Q": "Unclassifiable / paced"}
MODELS = {"raw_only": ("Raw ECG + RR", "#1b4965", "o"), "ph_only": ("PH-only", "#7a8b3a", "^"), "fusion": ("Raw ECG + RR + PH fusion", "#c44536", "s")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate full DS2 NSTDB results over matched seeds.")
    parser.add_argument("--seed-dirs", nargs="+", type=Path, default=[PROJECT_ROOT / "results/noise_robustness/full_ds2_seed42", PROJECT_ROOT / "results/noise_robustness/full_ds2_seed43", PROJECT_ROOT / "results/noise_robustness/full_ds2_seed44"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/noise_robustness/aggregate_seeds_42_44")
    return parser.parse_args()


def load_records(seed_dirs: list[Path]) -> tuple[list[dict], int]:
    grouped: dict[tuple[str, float | None, str], list[dict]] = defaultdict(list)
    for directory in seed_dirs:
        metrics_path = directory / "noise_robustness_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing completed evaluation: {metrics_path}")
        with metrics_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for condition in payload["results"]:
            for model, metrics in condition["models"].items():
                grouped[(condition["condition"], condition["snr_db"], model)].append(metrics)
    aggregate: list[dict] = []
    for (condition, snr_db, model), values in grouped.items():
        if len(values) != len(seed_dirs):
            raise ValueError(f"Expected {len(seed_dirs)} seeds for {condition}/{snr_db}/{model}, got {len(values)}.")
        row: dict[str, object] = {"condition": condition, "snr_db": snr_db, "model": model, "seeds": len(values), "total_beats": values[0]["total_beats"]}
        for metric in ("accuracy", "macro_f1"):
            scores = np.asarray([float(value[metric]) for value in values])
            row[f"{metric}_mean"] = float(scores.mean())
            row[f"{metric}_std"] = float(scores.std(ddof=1))
        for class_name in CLASS_NAMES:
            for metric in ("f1", "recall"):
                scores = np.asarray([float(value["per_class"][class_name][metric]) for value in values])
                row[f"{class_name}_{metric}_mean"] = float(scores.mean())
                row[f"{class_name}_{metric}_std"] = float(scores.std(ddof=1))
        aggregate.append(row)
    return aggregate, len(seed_dirs)


def points(rows: list[dict], condition: str, model: str) -> list[dict]:
    return sorted((row for row in rows if row["condition"] == condition and row["model"] == model), key=lambda row: float(row["snr_db"]), reverse=True)


def dashboard(rows: list[dict], metric: str, ylabel: str, output_path: Path, seed_count: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (condition, title) in zip(axes.flat, CONDITIONS.items()):
        for model, (label, color, marker) in MODELS.items():
            series = points(rows, condition, model)
            x = np.asarray([float(row["snr_db"]) for row in series])
            mean = 100 * np.asarray([float(row[f"{metric}_mean"]) for row in series])
            std = 100 * np.asarray([float(row[f"{metric}_std"]) for row in series])
            axis.plot(x, mean, label=label, color=color, marker=marker, linewidth=2.2, markersize=5)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks([24, 18, 12, 6, 0, -6])
        axis.set_xlim(25, -7)
        axis.grid(alpha=0.28)
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    for axis in axes[1, :]:
        axis.set_xlabel("Target SNR (dB): 24 light -> -6 severe")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    figure.suptitle(f"Full MIT-BIH DS2 Robustness: {ylabel} (mean +/- SD, {seed_count} matched seeds)", fontsize=16, fontweight="bold")
    figure.text(0.5, 0.075, "All 49,668 eligible DS2 contexts per seed. Shaded bands show sample standard deviation across seeds.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def class_dashboards(rows: list[dict], output_dir: Path, seed_count: int) -> None:
    for condition, condition_name in CONDITIONS.items():
        figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
        for axis, (class_name, full_name) in zip(axes.flat, CLASS_NAMES.items()):
            for model, (label, color, marker) in MODELS.items():
                series = points(rows, condition, model)
                x = np.asarray([float(row["snr_db"]) for row in series])
                mean = 100 * np.asarray([float(row[f"{class_name}_recall_mean"]) for row in series])
                std = 100 * np.asarray([float(row[f"{class_name}_recall_std"]) for row in series])
                axis.plot(x, mean, label=label, color=color, marker=marker, linewidth=2.0, markersize=5)
                axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
            axis.set_title(f"{class_name}: {full_name}", fontweight="bold")
            axis.set_xticks([24, 18, 12, 6, 0, -6])
            axis.set_xlim(25, -7)
            axis.set_ylim(-2, 102)
            axis.grid(alpha=0.28)
        axes[1, 2].axis("off")
        for axis in axes[:, 0]:
            axis.set_ylabel("Class-specific recall (%)")
        for axis in axes[1, :2]:
            axis.set_xlabel("Target SNR (dB): 24 light -> -6 severe")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
        figure.suptitle(f"Full MIT-BIH DS2: Per-Class Recall Under {condition_name} (mean +/- SD, {seed_count} seeds)", fontsize=15, fontweight="bold")
        figure.text(0.5, 0.075, "F and Q are rare classes, so their uncertainty bands should not be interpreted as stable estimates.", ha="center", fontsize=8.5)
        figure.tight_layout(rect=(0, 0.12, 1, 0.94))
        figure.savefig(output_dir / f"per_class_recall_vs_snr_{condition}_mean_sd.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows, seed_count = load_records(args.seed_dirs)
    columns = list(rows[0])
    with (output_dir / "aggregate_metrics_mean_sd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    dashboard(rows, "accuracy", "Classification accuracy (%)", output_dir / "accuracy_vs_snr_mean_sd.png", seed_count)
    dashboard(rows, "macro_f1", "Macro-F1 (%)", output_dir / "macro_f1_vs_snr_mean_sd.png", seed_count)
    class_dashboards(rows, output_dir, seed_count)
    print(f"Saved aggregate CSV and six plots to {output_dir}")


if __name__ == "__main__":
    main()
