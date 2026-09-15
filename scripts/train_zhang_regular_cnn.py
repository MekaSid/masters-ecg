"""Train the Zhang et al. (2021) regular-CNN reproduction on clean MIT-BIH."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.zhang_dataset import ZhangECGDataset, build_zhang_split
from src.training.zhang_train import train_zhang_regular_cnn
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean Zhang et al. regular-CNN reproduction.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override the configured maximum epoch count.")
    parser.add_argument("--run-name", default="zhang_regular_clean", help="Subdirectory under results/zhang_regular_cnn.")
    return parser.parse_args()


def random_validation_split(dataset: ZhangECGDataset, fraction: float, seed: int) -> tuple[ZhangECGDataset, ZhangECGDataset]:
    """Make the paper's seeded beat-level validation subset from DS1 only."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    validation_count = int(round(len(dataset) * fraction))
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    return (
        ZhangECGDataset(dataset.features[training_indices].numpy(), dataset.labels[training_indices].numpy()),
        ZhangECGDataset(dataset.features[validation_indices].numpy(), dataset.labels[validation_indices].numpy()),
    )


def class_counts(dataset: ZhangECGDataset, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(dataset.labels.numpy(), minlength=len(class_names))
    return {name: int(count) for name, count in zip(class_names, counts)}


def main() -> None:
    args = parse_args()
    data_config = load_yaml(CONFIGS_DIR / "data.yaml")
    config = load_yaml(CONFIGS_DIR / "zhang_regular_cnn.yaml")["zhang_regular_cnn"]
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    set_seed(data_config["project"]["seed"])
    class_names = list(config["classes"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    mitdb_dir = PROJECT_ROOT / data_config["paths"]["mitdb_dir"]
    ds1_dataset, ds1_summary = build_zhang_split(config["ds1_records"], mitdb_dir, class_to_index)
    test_dataset, ds2_summary = build_zhang_split(config["ds2_records"], mitdb_dir, class_to_index)
    train_dataset, validation_dataset = random_validation_split(ds1_dataset, float(config["validation_fraction"]), int(config["split_seed"]))
    output_dir = ensure_dir(PROJECT_ROOT / data_config["paths"]["results_dir"] / "zhang_regular_cnn" / args.run_name)
    history, results, model_config = train_zhang_regular_cnn(train_dataset, validation_dataset, test_dataset, class_names, config, output_dir)
    results.update(
        {
            "class_names": class_names,
            "model_config": model_config.__dict__,
            "ds1": {"records": list(ds1_summary.record_ids), "beats": ds1_summary.retained_beats, "class_counts": ds1_summary.class_counts},
            "train": {"beats": len(train_dataset), "class_counts": class_counts(train_dataset, class_names)},
            "validation": {"beats": len(validation_dataset), "class_counts": class_counts(validation_dataset, class_names)},
            "ds2": {"records": list(ds2_summary.record_ids), "beats": ds2_summary.retained_beats, "class_counts": ds2_summary.class_counts},
            "reproduction_note": "Attention internals are not specified by Zhang et al.; this implementation uses documented channel-and-temporal attention.",
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"DS1 beats: {ds1_summary.retained_beats}, classes={ds1_summary.class_counts}")
    print(f"DS2 beats: {ds2_summary.retained_beats}, classes={ds2_summary.class_counts}")
    print(f"Best epoch: {results['best_epoch']}, held-out DS2 accuracy: {results['test']['accuracy']:.4f}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
