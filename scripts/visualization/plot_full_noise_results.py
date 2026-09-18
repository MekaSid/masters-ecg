"""Create full-DS2 accuracy and macro-F1 noise robustness dashboards."""

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


CONDITIONS = {
    "bw": "Baseline Wander (BW)",
    "ma": "Muscle Artifact (MA)",
    "em": "Electrode Motion (EM)",
    "mix": "Combined BW + MA + EM",
}
MODELS = {
    "raw_only": ("Raw ECG + RR", "#1b4965", "o"),
    "ph_only": ("PH-only", "#7a8b3a", "^"),
    "fusion": ("Raw ECG + RR + PH fusion", "#c44536", "s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full DS2 model robustness results.")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=PROJECT_ROOT / "results/noise_robustness/full_ds2_seed42/noise_robustness_metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def create_dashboard(records: list[dict], metric_key: str, ylabel: str, output_path: Path, seed_label: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (condition, title) in zip(axes.flat, CONDITIONS.items()):
        points = sorted((record for record in records if record["condition"] == condition), key=lambda value: float(value["snr_db"]), reverse=True)
        for model_name, (label, color, marker) in MODELS.items():
            snr_values = [float(point["snr_db"]) for point in points]
            metric_values = [100.0 * float(point["models"][model_name][metric_key]) for point in points]
            axis.plot(snr_values, metric_values, label=label, color=color, marker=marker, linewidth=2.2, markersize=6)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks([24, 18, 12, 6, 0, -6])
        # Explicit limits keep shared subplot axes ordered light -> severe.
        axis.set_xlim(25, -7)
        axis.grid(alpha=0.28)
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    for axis in axes[1, :]:
        axis.set_xlabel("Target SNR (dB): 24 is light noise; -6 is severe noise")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    figure.suptitle(f"Full MIT-BIH DS2 Robustness, {seed_label}: {ylabel} vs. NSTDB Noise Level", fontsize=16, fontweight="bold")
    figure.text(0.5, 0.075, "All 49,668 eligible DS2 contexts. Each Raw/PH-only/Fusion comparison uses the identical corrupted ECG realization.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.metrics_json.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {args.metrics_json}")
    with args.metrics_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    output_dir = ensure_dir(args.output_dir or args.metrics_json.parent / "plots")
    records = payload["results"]
    seed_label = args.metrics_json.parent.name.replace("full_ds2_", "").replace("_", " ").title()
    create_dashboard(records, "accuracy", "Classification accuracy (%)", output_dir / "accuracy_vs_snr_all_conditions.png", seed_label)
    create_dashboard(records, "macro_f1", "Macro-F1 (%)", output_dir / "macro_f1_vs_snr_all_conditions.png", seed_label)
    print(f"Saved robustness dashboards to {output_dir}")


if __name__ == "__main__":
    main()
