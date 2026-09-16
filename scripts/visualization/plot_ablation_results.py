"""Plot matched multi-seed raw-versus-fusion results from saved metric JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a paired raw-only versus PH-fusion ablation.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "results/dindin_betti/paired_ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_metrics(run_dir: Path, seeds: list[int], model_name: str) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {"accuracy": [], "N": [], "S": [], "V": []}
    for seed in seeds:
        path = run_dir / f"seed_{seed}" / f"{model_name}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for seed {seed}: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        per_class = {name: result for name, result in zip(metrics["class_names"], metrics["test"]["per_class"])}
        values["accuracy"].append(float(metrics["test"]["accuracy"]))
        for class_name in ("N", "S", "V"):
            values[class_name].append(float(per_class[class_name]["f1"]))
    return {name: np.asarray(metric_values) for name, metric_values in values.items()}


def main() -> None:
    args = parse_args()
    output = args.output or args.run_dir / "paired_ablation_metrics.png"
    model_names = ["raw_only", "fusion"]
    labels = ["Raw Zhang\n(same paired subset)", "Raw + Betti\nfusion"]
    colors = ["#4C6A92", "#BB5A3C"]
    metrics = {name: load_metrics(args.run_dir, args.seeds, name) for name in model_names}

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.3))
    for axis, metric_name, title in zip(
        axes.flat,
        ("accuracy", "N", "S", "V"),
        ("Held-out DS2 Accuracy", "Normal Beat F1", "Supraventricular Beat F1", "Ventricular Beat F1"),
    ):
        values = [metrics[model][metric_name] for model in model_names]
        means = [value.mean() * 100 for value in values]
        stds = [value.std(ddof=1) * 100 for value in values]
        positions = np.arange(len(model_names))
        axis.bar(positions, means, yerr=stds, capsize=5, color=colors, width=0.58, alpha=0.9)
        for position, model_values in zip(positions, values):
            axis.scatter(np.full(len(model_values), position), model_values * 100, color="#1f1f1f", s=22, zorder=3)
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Percent")
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, 101)
        for position, mean in zip(positions, means):
            axis.text(position, min(mean + 3.5, 99), f"{mean:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    figure.suptitle("Matched Clean DS2 Ablation: Raw Zhang vs. Dindin-Style Betti Fusion", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        f"Bars show mean +/- sample standard deviation across seeds {args.seeds}. Dots show individual seed results. "
        "Both variants use the identical three-beat-filtered DS1/DS2 examples.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved ablation chart: {output}")


if __name__ == "__main__":
    main()
