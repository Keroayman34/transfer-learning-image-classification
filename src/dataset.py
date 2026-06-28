"""
Dataset utilities for the Cats vs Dogs transfer-learning project.

The module keeps dataset work deterministic and notebook-friendly:
images are validated before splitting, class indices are fixed, and
DataLoaders receive seeded workers.
"""

from __future__ import annotations

from pathlib import Path
import random
import warnings
from typing import Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 2

CLASS_NAMES = ["Cat", "Dog"]
CLASS_TO_IDX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: class_name for class_name, index in CLASS_TO_IDX.items()}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATASET = PROJECT_ROOT / "dataset" / "raw" / "PetImages"
PROCESSED_DATASET = PROJECT_ROOT / "dataset" / "processed"


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directory(path: Path) -> None:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def is_valid_image(image_path: Path) -> bool:
    """Return True when PIL can identify and verify an image file."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(image_path) as image:
                image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def find_invalid_images(dataset_path: Path = RAW_DATASET) -> list[Path]:
    """Find corrupt or unreadable images without deleting anything."""

    invalid_images: list[Path] = []

    for class_name in CLASS_NAMES:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if not is_valid_image(image_path):
                invalid_images.append(image_path)

    return invalid_images


def collect_images(
    dataset_path: Path = RAW_DATASET,
    validate: bool = True,
) -> dict[str, list[Path]]:
    """
    Collect image paths for each class using a stable class order.

    Invalid images are skipped by default. This avoids DataLoader crashes and
    prevents train/validation/test metrics from depending on notebook state.
    """

    dataset: dict[str, list[Path]] = {}

    for class_name in CLASS_NAMES:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        image_paths = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ]

        if validate:
            image_paths = [path for path in image_paths if is_valid_image(path)]

        dataset[class_name] = image_paths

    return dataset


def split_dataset(
    dataset: dict[str, list[Path]],
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, dict[str, list[Path]]]:
    """Split every class into deterministic train, validation, and test sets."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1.")

    rng = random.Random(seed)
    splits: dict[str, dict[str, list[Path]]] = {}

    for class_name in CLASS_NAMES:
        images = list(dataset.get(class_name, []))
        rng.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        valid_end = train_end + int(total * valid_ratio)

        splits[class_name] = {
            "train": images[:train_end],
            "valid": images[train_end:valid_end],
            "test": images[valid_end:],
        }

    return splits


def get_split_counts(
    splits: dict[str, dict[str, list[Path]]],
) -> dict[str, dict[str, int]]:
    """Summarize split sizes per class."""

    return {
        class_name: {
            split_name: len(paths)
            for split_name, paths in split_data.items()
        }
        for class_name, split_data in splits.items()
    }


def get_train_transforms() -> transforms.Compose:
    """Return augmentations used for model training."""

    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic transforms used for validation and testing."""

    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class CatsDogsDataset(datasets.VisionDataset):
    """PyTorch Dataset for Cats vs Dogs image classification."""

    def __init__(
        self,
        image_paths: Iterable[Path],
        labels: Iterable[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__(root=".")
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

        if len(self.image_paths) != len(self.labels):
            raise ValueError("image_paths and labels must have the same length.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load image: {image_path}") from exc

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def prepare_dataset(
    split_data: dict[str, list[Path]],
    transform: transforms.Compose | None = None,
) -> CatsDogsDataset:
    """Convert one split dictionary into a CatsDogsDataset."""

    image_paths: list[Path] = []
    labels: list[int] = []

    for class_name in CLASS_NAMES:
        for image_path in split_data.get(class_name, []):
            image_paths.append(image_path)
            labels.append(CLASS_TO_IDX[class_name])

    return CatsDogsDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def create_dataloader(
    dataset: CatsDogsDataset,
    shuffle: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
) -> DataLoader:
    """Create a reproducible PyTorch DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
