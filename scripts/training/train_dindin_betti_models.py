"""Train PH-only and raw-plus-PH Dindin-style Betti models on clean MIT-BIH."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tda.betti import BettiCurveConfig
from src.training.betti_dataset import BettiFusionDataset, load_or_build_betti_split
from src.training.betti_train import train_betti_model
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dindin-style Betti PH-only and fusion models.")
    parser.add_argument("--models", nargs="+", choices=["ph_only", "fusion"], default=["ph_only", "fusion"])
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--run-name", default="dindin_betti_clean")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def split_dataset(dataset: BettiFusionDataset, fraction: float, seed: int) -> tuple[BettiFusionDataset, BettiFusionDataset]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    indices = np.random.default_rng(seed).permutation(len(dataset))
    validation_count = round(len(dataset) * fraction)
    validation_indices, train_indices = indices[:validation_count], indices[validation_count:]
    return (
        BettiFusionDataset(dataset.raw_features[train_indices].numpy(), dataset.betti_features[train_indices].numpy(), dataset.labels[train_indices].numpy()),
        BettiFusionDataset(dataset.raw_features[validation_indices].numpy(), dataset.betti_features[validation_indices].numpy(), dataset.labels[validation_indices].numpy()),
    )


def counts(dataset: BettiFusionDataset, class_names: list[str]) -> dict[str, int]:
    values = np.bincount(dataset.labels.numpy(), minlength=len(class_names))
    return {name: int(value) for name, value in zip(class_names, values)}


def main() -> None:
    args = parse_args()
    data_config = load_yaml(CONFIGS_DIR / "data.yaml")
    config = load_yaml(CONFIGS_DIR / "models" / "dindin_betti_fusion.yaml")["dindin_betti_fusion"]
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    set_seed(data_config["project"]["seed"])
    class_names = list(config["classes"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    mitdb_dir = PROJECT_ROOT / data_config["paths"]["mitdb_dir"]
    cache_dir = PROJECT_ROOT / "data" / "tda" / "betti_curves"
    betti_config = BettiCurveConfig(resolution=int(config["betti_resolution"]))
    cache_suffix = f"threebeat_{config['ph_sequence_length']}samples_{config['betti_resolution']}bins"
    ds1_cache = cache_dir / f"ds1_{cache_suffix}.npz"
    ds2_cache = cache_dir / f"ds2_{cache_suffix}.npz"
    if args.rebuild_cache:
        ds1_cache.unlink(missing_ok=True)
        ds2_cache.unlink(missing_ok=True)
    ds1_dataset, ds1_summary = load_or_build_betti_split(
        ds1_cache, config["ds1_records"], mitdb_dir, class_to_index, betti_config, int(config["ph_sequence_length"]), int(config["ph_workers"])
    )
    ds2_dataset, ds2_summary = load_or_build_betti_split(
        ds2_cache, config["ds2_records"], mitdb_dir, class_to_index, betti_config, int(config["ph_sequence_length"]), int(config["ph_workers"])
    )
    train_dataset, validation_dataset = split_dataset(ds1_dataset, float(config["validation_fraction"]), int(config["split_seed"]))
    output_dir = ensure_dir(PROJECT_ROOT / data_config["paths"]["results_dir"] / "dindin_betti" / args.run_name)
    common_metadata = {
        "class_names": class_names,
        "ds1": {"beats": ds1_summary.retained_beats, "class_counts": ds1_summary.class_counts},
        "ds2": {"beats": ds2_summary.retained_beats, "class_counts": ds2_summary.class_counts},
        "train": {"beats": len(train_dataset), "class_counts": counts(train_dataset, class_names)},
        "validation": {"beats": len(validation_dataset), "class_counts": counts(validation_dataset, class_names)},
        "ph_representation": "GUDHI H0 sublevel and upper-level Betti curves from three consecutive beats on lead I.",
    }
    for kind in args.models:
        history, metrics = train_betti_model(kind, train_dataset, validation_dataset, ds2_dataset, len(class_names), config, output_dir)
        metrics.update(common_metadata)
        with (output_dir / f"{kind}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with (output_dir / f"{kind}_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"{kind} DS2 accuracy={metrics['test']['accuracy']:.4f}; results={output_dir}")


if __name__ == "__main__":
    main()
