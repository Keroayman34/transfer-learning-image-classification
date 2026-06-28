"""Training utilities for transfer learning experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


History = dict[str, list[float]]


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot run an epoch on an empty DataLoader.")

    return running_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch."""

    return _run_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
    )


def validate_one_epoch(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model for one epoch."""

    return _run_epoch(
        model=model,
        dataloader=valid_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )


def save_checkpoint(
    model: nn.Module,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a model state dict and optional metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    checkpoint_path: Path | str | None = None,
) -> History:
    """
    Train the model and keep the best validation-accuracy checkpoint.

    The returned history includes the best epoch and best validation accuracy
    as single-item lists to keep the plotting keys simple and serializable.
    """

    history: History = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "best_epoch": [],
        "best_valid_acc": [],
    }

    best_valid_acc = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        valid_loss, valid_acc = validate_one_epoch(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

            if checkpoint is not None:
                save_checkpoint(
                    model=model,
                    path=checkpoint,
                    metadata={
                        "epoch": best_epoch,
                        "valid_accuracy": best_valid_acc,
                    },
                )

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_acc:.4f}"
        )

    model.load_state_dict(best_state)
    history["best_epoch"].append(float(best_epoch))
    history["best_valid_acc"].append(best_valid_acc)

    if checkpoint is not None and not checkpoint.exists():
        save_checkpoint(
            model=model,
            path=checkpoint,
            metadata={
                "epoch": best_epoch,
                "valid_accuracy": best_valid_acc,
            },
        )

    return history
