"""Training utilities for controlled Dindin-style Betti experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.betti_fusion_model import DindinPHOnlyCNN, ZhangBettiFusionCNN
from src.training.betti_dataset import BettiFusionDataset
from src.training.train import select_device

ModelKind = Literal["ph_only", "fusion"]


def _forward(model: nn.Module, raw: torch.Tensor, betti: torch.Tensor, kind: ModelKind) -> torch.Tensor:
    return model(betti) if kind == "ph_only" else model(raw, betti)


@torch.no_grad()
def evaluate_betti_model(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, kind: ModelKind) -> dict[str, object]:
    """Calculate loss-independent accuracy, confusion matrix, and per-class F1."""
    model.eval()
    correct = 0
    total = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for raw, betti, targets in loader:
        predictions = _forward(model, raw.to(device), betti.to(device), kind).argmax(dim=1).cpu()
        targets = targets.cpu()
        correct += int((predictions == targets).sum())
        total += len(targets)
        for target, prediction in zip(targets, predictions):
            confusion[target, prediction] += 1
    per_class: list[dict[str, float | int]] = []
    f1_values: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        support = int(confusion[class_index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_class.append({"support": support, "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "total_beats": total,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def _loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, kind: ModelKind) -> float:
    model.eval()
    loss_sum = 0.0
    total = 0
    for raw, betti, targets in loader:
        targets = targets.to(device)
        loss_sum += float(criterion(_forward(model, raw.to(device), betti.to(device), kind), targets).item()) * len(targets)
        total += len(targets)
    return loss_sum / total


def train_betti_model(
    kind: ModelKind,
    train_dataset: BettiFusionDataset,
    validation_dataset: BettiFusionDataset,
    test_dataset: BettiFusionDataset,
    num_classes: int,
    config: dict[str, float | int],
    output_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Train a PH-only or end-to-end raw-plus-PH model using validation loss."""
    device = select_device()
    model: nn.Module = DindinPHOnlyCNN(num_classes) if kind == "ph_only" else ZhangBettiFusionCNN(num_classes)
    model = model.to(device)
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(config["lr_factor"]), patience=int(config["lr_patience"])
    )
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        total = 0
        for raw, betti, targets in train_loader:
            raw, betti, targets = raw.to(device), betti.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_forward(model, raw, betti, kind), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(targets)
            total += len(targets)
        validation_loss = _loss(model, validation_loader, device, criterion, kind)
        validation_metrics = evaluate_betti_model(model, validation_loader, device, num_classes, kind)
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": float(epoch),
            "train_loss": loss_sum / total,
            "validation_loss": validation_loss,
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
            "learning_rate": learning_rate,
        })
        print(
            f"{kind} epoch {epoch:03d}: train_loss={loss_sum / total:.4f}, val_loss={validation_loss:.4f}, "
            f"val_accuracy={validation_metrics['accuracy']:.4f}, val_macro_f1={validation_metrics['macro_f1']:.4f}, lr={learning_rate:g}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                print(f"{kind} early stop at epoch {epoch}; best epoch {best_epoch}.", flush=True)
                break
    if best_state is None:
        raise RuntimeError("No Betti-model checkpoint was created.")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate_betti_model(model, test_loader, device, num_classes, kind)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict(), "model_kind": kind}, output_dir / f"{kind}.pt")
    return history, {
        "device": str(device),
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "test": test_metrics,
    }
