"""Evaluation and prediction utilities for trained models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from dataset import CLASS_NAMES
except ImportError:  # pragma: no cover - package import fallback
    from .dataset import CLASS_NAMES


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model loss and accuracy on a DataLoader."""

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    return running_loss / total, correct / total


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, predicted labels, and class probabilities."""

    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def build_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Build a confusion matrix in Cat, Dog order."""

    return confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )


def build_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> str:
    """Build a text classification report."""

    return classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )


def predict_image(
    model: nn.Module,
    image_path: Path | str,
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[str, float]:
    """Predict the class name and confidence for a single image."""

    path = Path(image_path)
    model.eval()

    try:
        image = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load image: {path}") from exc

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        confidence, prediction = probabilities.max(dim=0)

    return CLASS_NAMES[prediction.item()], confidence.item()
