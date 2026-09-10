from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.raw_model import RawECGConvNet, RawECGModelConfig
from src.training.dataset import ECGBeatDataset, SplitSummary


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(labels: torch.Tensor, num_classes: int, power: float = 0.5) -> torch.Tensor:
    """Compute tempered inverse-frequency weights to avoid minority-class overcorrection."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError(f"Training split is missing configured classes: counts={counts.tolist()}.")
    if not 0.0 <= power <= 1.0:
        raise ValueError("class weight power must be between 0 and 1.")
    weights = (counts.mean() / counts).pow(power)
    return weights / weights.mean()


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for features, targets in loader:
        logits = model(features.to(device))
        predictions = logits.argmax(dim=1).cpu()
        targets = targets.cpu()
        correct += int((predictions == targets).sum())
        total += len(targets)
        for target, prediction in zip(targets, predictions):
            confusion[target, prediction] += 1
    per_class: list[dict[str, float | int]] = []
    supported_f1: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        support = int(confusion[class_index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            supported_f1.append(f1)
        per_class.append({"support": support, "precision": precision, "recall": recall, "f1": f1})

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0,
        "total_beats": total,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def train_raw_ecg_model(
    train_dataset: ECGBeatDataset,
    val_dataset: ECGBeatDataset,
    test_dataset: ECGBeatDataset,
    class_names: list[str],
    model_config: RawECGModelConfig,
    training_config: dict[str, Any],
    output_dir: Path,
) -> tuple[RawECGConvNet, list[dict[str, float]], dict[str, Any]]:
    """Train the raw ECG Conv1D baseline and select its checkpoint by validation accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    batch_size = int(training_config["batch_size"])
    workers = int(training_config.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = RawECGConvNet(model_config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            train_dataset.labels,
            len(class_names),
            power=float(training_config.get("class_weight_power", 0.5)),
        ).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(targets)
            seen += len(targets)

        val_metrics = evaluate_model(model, val_loader, device, len(class_names))
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": running_loss / seen,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:02d}/{training_config['epochs']}: "
            f"loss={epoch_metrics['train_loss']:.4f}, val_accuracy={epoch_metrics['val_accuracy']:.4f}, "
            f"val_macro_f1={epoch_metrics['val_macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate_model(model, test_loader, device, len(class_names))
    summary = {
        "device": str(device),
        "class_names": class_names,
        "best_validation_macro_f1": best_val_macro_f1,
        "test": test_metrics,
        "model_config": asdict(model_config),
    }
    torch.save(
        {"model_state_dict": model.cpu().state_dict(), "class_names": class_names, "model_config": asdict(model_config)},
        output_dir / "raw_ecg_conv1d.pt",
    )
    return model, history, summary


def serialize_split_summary(summary: SplitSummary) -> dict[str, Any]:
    return asdict(summary)
