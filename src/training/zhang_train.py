"""Training loop for the clean Zhang et al. regular-CNN reproduction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.zhang_regular_cnn import ZhangRegularCNN, ZhangRegularCNNConfig
from src.training.train import evaluate_model, select_device
from src.training.zhang_dataset import ZhangECGDataset


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    """Calculate unweighted validation cross-entropy for LR scheduling/checkpointing."""
    model.eval()
    loss_sum = 0.0
    total = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        loss_sum += float(criterion(model(features), targets).item()) * len(targets)
        total += len(targets)
    return loss_sum / total


def train_zhang_regular_cnn(
    train_dataset: ZhangECGDataset,
    validation_dataset: ZhangECGDataset,
    test_dataset: ZhangECGDataset,
    class_names: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, Any], ZhangRegularCNNConfig]:
    """Train regular CNN with the paper's Adam/LR-reduction/early-stop procedure."""
    device = select_device()
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model_config = ZhangRegularCNNConfig(num_classes=len(class_names))
    model = ZhangRegularCNN(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["lr_factor"]),
        patience=int(config["lr_patience"]),
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(config["max_epochs"]) + 1):
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

        val_loss = validation_loss(model, validation_loader, device, criterion)
        val_metrics = evaluate_model(model, validation_loader, device, len(class_names))
        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / seen,
                "validation_loss": val_loss,
                "validation_accuracy": float(val_metrics["accuracy"]),
                "validation_macro_f1": float(val_metrics["macro_f1"]),
                "learning_rate": learning_rate,
            }
        )
        print(
            f"Epoch {epoch:03d}: train_loss={running_loss / seen:.4f}, val_loss={val_loss:.4f}, "
            f"val_accuracy={val_metrics['accuracy']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}, lr={learning_rate:g}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["early_stopping_patience"]):
                print(f"Early stopping at epoch {epoch}; best validation loss was at epoch {best_epoch}.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No Zhang regular-CNN checkpoint was created.")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate_model(model, test_loader, device, len(class_names))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.cpu().state_dict(), "class_names": class_names, "model_config": asdict(model_config)},
        output_dir / "zhang_regular_cnn.pt",
    )
    return history, {
        "device": str(device),
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "test": test_metrics,
    }, model_config
