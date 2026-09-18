"""Plot seed-specific per-class recall across all NSTDB noise levels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir


CONDITIONS = {"bw": "Baseline Wander (BW)", "ma": "Muscle Artifact (MA)", "em": "Electrode Motion (EM)", "mix": "Combined BW + MA + EM"}
CLASS_NAMES = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular", "F": "Fusion", "Q": "Unclassifiable / paced"}
MODELS = {"raw_only": ("Raw ECG + RR", "#1b4965", "o"), "ph_only": ("PH-only", "#7a8b3a", "^"), "fusion": ("Raw ECG + RR + PH fusion", "#c44536", "s")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot class-specific recall across NSTDB SNR levels.")
    parser.add_argument("--metrics-json", type=Path, default=PROJECT_ROOT / "results/noise_robustness/full_ds2_seed42/noise_robustness_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metrics_json.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {args.metrics_json}")
    with args.metrics_json.open(encoding="utf-8") as handle:
        records = json.load(handle)["results"]
    output_dir = ensure_dir(args.output_dir or args.metrics_json.parent / "plots")
    seed_label = args.metrics_json.parent.name.replace("full_ds2_", "").replace("_", " ").title()

    for condition, condition_name in CONDITIONS.items():
        points = sorted((item for item in records if item["condition"] == condition), key=lambda item: float(item["snr_db"]), reverse=True)
        figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
        for axis, (class_name, full_name) in zip(axes.flat, CLASS_NAMES.items()):
            for model_name, (label, color, marker) in MODELS.items():
                snr = [float(point["snr_db"]) for point in points]
                recall = [100.0 * float(point["models"][model_name]["per_class"][class_name]["recall"]) for point in points]
                axis.plot(snr, recall, label=label, color=color, marker=marker, linewidth=2.0, markersize=5)
            support = points[0]["models"]["raw_only"]["per_class"][class_name]["support"]
            axis.set_title(f"{class_name}: {full_name} (n={support:,})", fontweight="bold")
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
        figure.suptitle(f"Full MIT-BIH DS2, {seed_label}: Per-Class Recall Under {condition_name}", fontsize=16, fontweight="bold")
        figure.text(0.5, 0.075, "Recall is class-specific accuracy. F and Q have only 388 and 7 DS2 examples, respectively, so their curves are unstable.", ha="center", fontsize=8.5)
        figure.tight_layout(rect=(0, 0.12, 1, 0.94))
        output_path = output_dir / f"per_class_recall_vs_snr_{condition}.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
