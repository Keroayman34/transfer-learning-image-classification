"""General helper functions for the notebook and scripts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt

try:
    from dataset import CLASS_NAMES
except ImportError:  # pragma: no cover - package import fallback
    from .dataset import CLASS_NAMES


RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def ensure_results_dirs(
    results_dir: Path = RESULTS_DIR,
    plots_dir: Path = PLOTS_DIR,
) -> tuple[Path, Path]:
    """Create results directories and return their paths."""

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized tensor into a displayable image."""

    image_array = image.detach().cpu().permute(1, 2, 0).numpy()
    image_array = image_array * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image_array, 0, 1)


def plot_loss_curve(
    history: dict[str, list[float]],
    save_path: Path,
) -> None:
    """Plot and save training/validation loss curves."""

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["valid_loss"], marker="o", label="Validation")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_accuracy_curve(
    history: dict[str, list[float]],
    save_path: Path,
) -> None:
    """Plot and save training/validation accuracy curves."""

    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train")
    plt.plot(epochs, history["valid_acc"], marker="o", label="Validation")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(
    matrix: np.ndarray,
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """Plot and save a confusion matrix."""

    names = class_names or CLASS_NAMES

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names)
    plt.yticks(tick_marks, names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=color,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path,
    class_names: list[str] | None = None,
    max_images: int = 8,
) -> None:
    """Plot and save a grid of sample predictions."""

    names = class_names or CLASS_NAMES
    count = min(max_images, len(images))
    cols = 4
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for index in range(count):
        label = int(labels[index].item())
        prediction = int(predictions[index])
        confidence = float(probabilities[index][prediction])
        color = "green" if prediction == label else "red"

        axes[index].imshow(denormalize_image(images[index]))
        axes[index].set_title(
            f"Pred: {names[prediction]} ({confidence:.2f})\n"
            f"True: {names[label]}",
            color=color,
        )
        axes[index].axis("off")

    for index in range(count, len(axes)):
        axes[index].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
