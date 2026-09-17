"""Plot raw-only and PH-fusion accuracy curves from robustness metrics JSON."""

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


CONDITION_NAMES = {
    "bw": "Baseline Wander (BW)",
    "ma": "Muscle Artifact (MA)",
    "em": "Electrode Motion (EM)",
    "mix": "Combined BW + MA + EM",
}
MODEL_STYLE = {
    "raw_only": {"label": "Raw ECG + RR CNN", "color": "#1b4965", "marker": "o"},
    "fusion": {"label": "Raw ECG + RR + PH fusion", "color": "#c44536", "marker": "s"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy versus SNR for each NSTDB noise condition.")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=PROJECT_ROOT / "results/noise_robustness/matched_seed45_quick_ds2/noise_robustness_metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metrics_json.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {args.metrics_json}")
    with args.metrics_json.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    output_dir = ensure_dir(args.output_dir or args.metrics_json.parent / "plots")
    records = metrics["results"]
    clean = next((record for record in records if record["condition"] == "clean"), None)
    if clean is None:
        raise ValueError("Metrics must include a clean baseline record.")
    conditions = [condition for condition in CONDITION_NAMES if any(record["condition"] == condition for record in records)]
    if not conditions:
        raise ValueError("Metrics did not contain any recognized noisy conditions.")

    for condition in conditions:
        points = sorted((record for record in records if record["condition"] == condition), key=lambda record: float(record["snr_db"]))
        figure, axis = plt.subplots(figsize=(8.6, 5.5))
        for model_name, style in MODEL_STYLE.items():
            snr = [float(record["snr_db"]) for record in points]
            accuracy = [100.0 * float(record["models"][model_name]["accuracy"]) for record in points]
            axis.plot(snr, accuracy, linewidth=2.4, markersize=7, **style)
            for x_value, y_value in zip(snr, accuracy):
                axis.annotate(f"{y_value:.1f}%", (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        clean_text = ", ".join(
            f"{MODEL_STYLE[name]['label']}: {100.0 * float(clean['models'][name]['accuracy']):.1f}%"
            for name in MODEL_STYLE
        )
        axis.set_title(f"{CONDITION_NAMES[condition]}: Accuracy vs. Noise Level\nClean baseline: {clean_text}", fontweight="bold")
        axis.set_xlabel("Target SNR (dB): lower means stronger noise")
        axis.set_ylabel("Classification accuracy (%)")
        axis.set_xticks([float(record["snr_db"]) for record in points])
        axis.set_ylim(30, 70)
        axis.grid(alpha=0.28)
        axis.legend(loc="lower right")
        figure.text(
            0.5,
            0.01,
            "Matched seed-45 models; patient-disjoint, class-stratified MIT-BIH DS2 screening subset (1,207 beats).",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        output_path = output_dir / f"accuracy_vs_snr_{condition}.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
