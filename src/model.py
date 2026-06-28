"""Model utilities for ResNet18 transfer learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

try:
    from train import train_model
except ImportError:  # pragma: no cover - package import fallback
    from .train import train_model


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet18 classifier with a replaced final layer."""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_feature_extractor(model: nn.Module) -> nn.Module:
    """Freeze all layers except the final fully connected classifier."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Unfreeze every model parameter for fine-tuning."""

    for parameter in model.parameters():
        parameter.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable model parameters."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {"total": total, "trainable": trainable}


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load model weights from a checkpoint file."""

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("metadata", {})

    model.load_state_dict(checkpoint)
    return {}
