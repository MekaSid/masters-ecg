"""Evaluate saved clean-trained Zhang and PH-fusion models under NSTDB noise.

The script intentionally keeps R-peak positions and RR features clean: NSTDB
noise changes morphology, not the annotated beat timing. The fusion model's
Betti curves are recomputed after corruption from its noisy three-beat context.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import resample

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.noise import compute_signal_power
from src.models.betti_fusion_model import ZhangBettiFusionCNN
from src.models.zhang_regular_cnn import ZhangRegularCNN, ZhangRegularCNNConfig
from src.tda.betti import BettiCurveConfig, dindin_betti_curves
from src.training.train import select_device
from src.training.zhang_dataset import ZHANG_LABEL_MAP, build_zhang_feature, dynamic_beat_bounds, rr_ratio_features
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, REPO_ROOT as PROJECT_ROOT, ensure_dir


CLASS_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class NoiseExample:
    """One labeled, three-beat ECG context, retaining both leads for raw input."""

    record_id: str
    label: int
    ph_signal: np.ndarray  # (samples, 2), preceding R peak through following R peak
    raw_start: int  # Offset within ph_signal.
    raw_end: int
    pre_rr_ratio: float
    near_pre_rr_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved ECG and PH-fusion models under NSTDB noise.")
    parser.add_argument("--raw-checkpoint", type=Path, default=PROJECT_ROOT / "results/zhang_regular_cnn/zhang_regular_clean/zhang_regular_cnn.pt")
    parser.add_argument("--fusion-checkpoint", type=Path, default=PROJECT_ROOT / "results/dindin_betti/dindin_betti_clean/fusion.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/noise_robustness/quick_ds2")
    parser.add_argument("--snr-levels", nargs="+", type=float, default=[24.0, 12.0, 0.0, -6.0])
    parser.add_argument("--noise-conditions", nargs="+", choices=["bw", "ma", "em", "mix"], default=["bw", "ma", "em", "mix"])
    parser.add_argument("--max-per-class", type=int, default=300, help="Stratified DS2 cap; use 0 for all eligible beats.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_examples(record_ids: list[str], mitdb_dir: Path, max_per_class: int, seed: int) -> list[NoiseExample]:
    """Extract a deterministic, class-stratified DS2 set aligned with fusion inputs."""
    grouped: dict[int, list[NoiseExample]] = defaultdict(list)
    for record_id in record_ids:
        record = wfdb.rdrecord(str(mitdb_dir / record_id))
        if record.p_signal.shape[1] < 2:
            raise ValueError(f"Record {record_id} does not have two leads required by the Zhang model.")
        signal = np.asarray(record.p_signal[:, :2], dtype=np.float32)
        annotation = wfdb.rdann(str(mitdb_dir / record_id), "atr")
        labels = [ZHANG_LABEL_MAP.get(symbol) for symbol in annotation.symbol]
        valid_indices = [index for index, label in enumerate(labels) if label in CLASS_TO_INDEX]
        rpeaks = np.asarray(annotation.sample[valid_indices], dtype=np.int64)
        mapped = [labels[index] for index in valid_indices]
        if len(rpeaks) < 3:
            continue
        global_rr, near_rr = rr_ratio_features(rpeaks)
        for index in range(1, len(rpeaks) - 1):
            ph_start, ph_end = int(rpeaks[index - 1]), int(rpeaks[index + 1])
            raw_start, raw_end = dynamic_beat_bounds(int(rpeaks[index - 1]), int(rpeaks[index]))
            if raw_start < ph_start or raw_end > ph_end or ph_end > len(signal):
                continue
            grouped[CLASS_TO_INDEX[mapped[index]]].append(
                NoiseExample(
                    record_id=record_id,
                    label=CLASS_TO_INDEX[mapped[index]],
                    ph_signal=signal[ph_start:ph_end],
                    raw_start=raw_start - ph_start,
                    raw_end=raw_end - ph_start,
                    pre_rr_ratio=float(global_rr[index]),
                    near_pre_rr_ratio=float(near_rr[index]),
                )
            )
    rng = np.random.default_rng(seed)
    selected: list[NoiseExample] = []
    for class_index in range(len(CLASS_NAMES)):
        candidates = grouped[class_index]
        if not candidates:
            raise ValueError(f"No eligible DS2 examples for class {CLASS_NAMES[class_index]}.")
        indices = rng.permutation(len(candidates))
        limit = len(candidates) if max_per_class == 0 else min(max_per_class, len(candidates))
        selected.extend(candidates[index] for index in indices[:limit])
    rng.shuffle(selected)
    return selected


def noise_for_condition(
    clean: np.ndarray,
    sources: dict[str, np.ndarray],
    condition: str,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample, combine, and scale NSTDB noise to a total per-lead target SNR."""
    source_names = [condition] if condition != "mix" else ["bw", "ma", "em"]
    noise = np.zeros_like(clean, dtype=np.float32)
    for source_name in source_names:
        source = sources[source_name]
        start = int(rng.integers(0, len(source) - len(clean) + 1))
        noise += source[start : start + len(clean)]
    scaled = np.empty_like(noise)
    for channel in range(clean.shape[1]):
        noise_power = compute_signal_power(noise[:, channel])
        if noise_power == 0:
            raise ValueError(f"NSTDB {condition} noise had zero power.")
        target_noise_power = compute_signal_power(clean[:, channel]) / (10.0 ** (snr_db / 10.0))
        scaled[:, channel] = noise[:, channel] * np.sqrt(target_noise_power / noise_power)
    return clean + scaled


def features_for_condition(
    examples: list[NoiseExample],
    sources: dict[str, np.ndarray] | None,
    condition: str,
    snr_db: float | None,
    seed: int,
    betti_config: BettiCurveConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create matched raw/RR and noisy-PH features for a test condition."""
    raw_features: list[np.ndarray] = []
    curves: list[np.ndarray] = []
    labels: list[int] = []
    rng = np.random.default_rng(seed)
    for example in examples:
        signal = example.ph_signal if condition == "clean" else noise_for_condition(example.ph_signal, sources or {}, condition, float(snr_db), rng)
        raw_segment = resample(signal[example.raw_start : example.raw_end], 128, axis=0).astype(np.float32)
        raw_features.append(build_zhang_feature(raw_segment, example.pre_rr_ratio, example.near_pre_rr_ratio))
        curves.append(dindin_betti_curves(resample(signal[:, 0], 256).astype(np.float32), betti_config))
        labels.append(example.label)
    return np.stack(raw_features), np.stack(curves), np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def evaluate_models(
    raw_model: ZhangRegularCNN,
    fusion_model: ZhangBettiFusionCNN,
    raw: np.ndarray,
    betti: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    raw_model.eval()
    fusion_model.eval()
    outputs: dict[str, list[np.ndarray]] = {"raw_only": [], "fusion": []}
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        raw_tensor = torch.from_numpy(raw[start:stop]).to(device)
        betti_tensor = torch.from_numpy(betti[start:stop]).to(device)
        outputs["raw_only"].append(raw_model(raw_tensor).argmax(dim=1).cpu().numpy())
        outputs["fusion"].append(fusion_model(raw_tensor, betti_tensor).argmax(dim=1).cpu().numpy())

    results: dict[str, dict[str, object]] = {}
    for name, parts in outputs.items():
        predictions = np.concatenate(parts)
        confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
        np.add.at(confusion, (labels, predictions), 1)
        per_class: dict[str, dict[str, float | int]] = {}
        f1_values: list[float] = []
        for index, class_name in enumerate(CLASS_NAMES):
            true_positive = int(confusion[index, index])
            false_positive = int(confusion[:, index].sum() - true_positive)
            support = int(confusion[index].sum())
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[class_name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
            if support:
                f1_values.append(f1)
        results[name] = {
            "accuracy": float((predictions == labels).mean()),
            "macro_f1": float(np.mean(f1_values)),
            "total_beats": int(len(labels)),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
        }
    return results


def load_models(raw_checkpoint: Path, fusion_checkpoint: Path, device: torch.device) -> tuple[ZhangRegularCNN, ZhangBettiFusionCNN]:
    if not raw_checkpoint.is_file() or not fusion_checkpoint.is_file():
        raise FileNotFoundError("Both saved raw-only and fusion checkpoints are required.")
    raw_model = ZhangRegularCNN(ZhangRegularCNNConfig(num_classes=len(CLASS_NAMES)))
    fusion_model = ZhangBettiFusionCNN(num_classes=len(CLASS_NAMES))
    raw_model.load_state_dict(torch.load(raw_checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    fusion_model.load_state_dict(torch.load(fusion_checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    return raw_model.to(device), fusion_model.to(device)


def main() -> None:
    args = parse_args()
    if args.max_per_class < 0 or args.batch_size < 1 or not args.snr_levels:
        raise ValueError("max-per-class must be >= 0, batch-size >= 1, and at least one SNR is required.")
    if not all(np.isfinite(args.snr_levels)):
        raise ValueError("SNR levels must be finite.")
    config = load_yaml(CONFIGS_DIR / "models" / "dindin_betti_fusion.yaml")["dindin_betti_fusion"]
    mitdb_dir = PROJECT_ROOT / "data/raw/mitdb"
    nstdb_dir = PROJECT_ROOT / "data/raw/nstdb"
    examples = load_examples(list(config["ds2_records"]), mitdb_dir, args.max_per_class, args.seed)
    sources = {name: np.asarray(wfdb.rdrecord(str(nstdb_dir / name)).p_signal[:, :2], dtype=np.float32) for name in ("bw", "ma", "em")}
    if any(len(source) < max(len(example.ph_signal) for example in examples) for source in sources.values()):
        raise ValueError("NSTDB recordings are shorter than an ECG context.")
    device = select_device()
    raw_model, fusion_model = load_models(args.raw_checkpoint, args.fusion_checkpoint, device)
    output_dir = ensure_dir(args.output_dir)
    betti_config = BettiCurveConfig(resolution=int(config["betti_resolution"]))
    records: list[dict[str, object]] = []
    conditions: list[tuple[str, float | None]] = [("clean", None)] + [(condition, snr) for condition in args.noise_conditions for snr in args.snr_levels]
    for condition_index, (condition, snr_db) in enumerate(conditions):
        raw, betti, labels = features_for_condition(examples, sources, condition, snr_db, args.seed + condition_index * 10_000, betti_config)
        metrics = evaluate_models(raw_model, fusion_model, raw, betti, labels, device, args.batch_size)
        record: dict[str, object] = {"condition": condition, "snr_db": snr_db, "models": metrics}
        records.append(record)
        print(f"{condition:>5} {str(snr_db):>5} dB | raw={metrics['raw_only']['accuracy']:.4f} | fusion={metrics['fusion']['accuracy']:.4f}", flush=True)
    summary = {
        "protocol": {
            "test_split": "MIT-BIH DS2, patient-disjoint from training",
            "max_per_class": args.max_per_class,
            "noise_conditions": {"bw": "baseline wander", "ma": "muscle artifact", "em": "electrode motion", "mix": "bw + ma + em, jointly scaled"},
            "snr_definition": "Total noise power is scaled per ECG lead to the requested SNR over each three-beat context.",
            "raw_features": "Noisy two-lead ECG morphology plus clean annotated RR-ratio maps.",
            "ph_features": "Noisy lead-I three-beat context -> direct H0 sublevel and upper-level PH -> 2 x 128 Betti curves.",
        },
        "class_names": CLASS_NAMES,
        "test_class_counts": {name: int(sum(example.label == index for example in examples)) for index, name in enumerate(CLASS_NAMES)},
        "results": records,
    }
    with (output_dir / "noise_robustness_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    csv_rows: list[dict[str, object]] = []
    for record in records:
        for model_name, metrics in record["models"].items():
            row: dict[str, object] = {"condition": record["condition"], "snr_db": record["snr_db"], "model": model_name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "total_beats": metrics["total_beats"]}
            for class_name, class_metrics in metrics["per_class"].items():
                row[f"{class_name}_f1"] = class_metrics["f1"]
            csv_rows.append(row)
    with (output_dir / "noise_robustness_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved metrics to {output_dir}")


if __name__ == "__main__":
    main()
