from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.beat_extraction import BeatWindow
from src.models.raw_model import RawECGModelConfig
from src.training.dataset import build_beat_split
from src.training.train import serialize_split_summary, train_raw_ecg_model
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the record-disjoint raw ECG Conv1D baseline.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the configured epoch count.")
    parser.add_argument(
        "--run-name",
        default="full_clean_record_disjoint",
        help="Subdirectory under results/raw_cnn for this training run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_yaml(CONFIGS_DIR / "data.yaml")
    training_config = load_yaml(CONFIGS_DIR / "models" / "training.yaml")["raw_ecg_cnn"]
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    set_seed(data_config["project"]["seed"])

    classes = list(training_config["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    window_config = data_config["data"]["beat_window"]
    window = BeatWindow(pre_samples=window_config["pre_samples"], post_samples=window_config["post_samples"])
    mitdb_dir = PROJECT_ROOT / data_config["paths"]["mitdb_dir"]

    train_dataset, train_summary = build_beat_split("train", training_config["train_records"], mitdb_dir, window, class_to_index)
    val_dataset, val_summary = build_beat_split("validation", training_config["val_records"], mitdb_dir, window, class_to_index)
    test_dataset, test_summary = build_beat_split("test", training_config["test_records"], mitdb_dir, window, class_to_index)
    output_dir = ensure_dir(PROJECT_ROOT / data_config["paths"]["results_dir"] / "raw_cnn" / args.run_name)

    model_config = RawECGModelConfig(
        input_length=window.length,
        num_classes=len(classes),
        channels=tuple(training_config["channels"]),
        dropout=float(training_config["dropout"]),
    )
    _, history, results = train_raw_ecg_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_names=classes,
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
    )
    results["splits"] = {
        "train": serialize_split_summary(train_summary),
        "validation": serialize_split_summary(val_summary),
        "test": serialize_split_summary(test_summary),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_accuracy", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(history)

    print("\nRecord-disjoint split:")
    for split in (train_summary, val_summary, test_summary):
        print(f"{split.split_name}: records={list(split.record_ids)}, beats={split.retained_beats}, classes={split.class_counts}")
    print(f"\nBest validation macro-F1: {results['best_validation_macro_f1']:.4f}")
    print(f"Test accuracy: {results['test']['accuracy']:.4f}")
    print(f"Saved model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
