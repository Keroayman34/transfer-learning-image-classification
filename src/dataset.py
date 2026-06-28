"""
dataset.py

This module contains all dataset-related operations:
1. Dataset splitting
2. Data augmentation
3. Image transformations
4. DataLoaders
"""

from pathlib import Path
import random
import shutil
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# Dataset Configuration

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_WORKERS = 2


# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET = PROJECT_ROOT / "dataset" / "raw" / "PetImages"

PROCESSED_DATASET = PROJECT_ROOT / "dataset" / "processed"




# Helper Functions

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)


# Directory Utilities

def create_directory(path: Path) -> None:
    """
    Create directory if it does not exist.
    """

    path.mkdir(parents=True, exist_ok=True)



# Dataset Utilities

def collect_images() -> dict[str, list[Path]]:
    """
    Collect all image paths for each class.

    Returns:
        Dictionary containing image paths grouped by class.
    """

    dataset = {}

    for class_dir in RAW_DATASET.iterdir():

        if not class_dir.is_dir():
            continue

        image_paths = sorted(class_dir.glob("*.jpg"))

        dataset[class_dir.name] = image_paths

    return dataset



# Dataset Splitting

def split_dataset(dataset: dict[str, list[Path]]) -> dict:
    """
    Split dataset into train / validation / test.
    """

    set_random_seed()

    splits = {}

    for class_name, images in dataset.items():

        images = images.copy()
        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        valid_end = train_end + int(total * VALID_RATIO)

        splits[class_name] = {
            "train": images[:train_end],
            "valid": images[train_end:valid_end],
            "test": images[valid_end:]
        }

    return splits



# Image Transformations

def get_train_transforms():
    """
    Transformations used during training.
    """

    train_transforms = transforms.Compose([

        transforms.Resize(IMAGE_SIZE),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    return train_transforms



def get_eval_transforms():
    """
    Transformations used for validation and testing.
    """

    eval_transforms = transforms.Compose([

        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    return eval_transforms


# Custom Dataset

class CatsDogsDataset(datasets.VisionDataset):
    """
    Custom Dataset for Cats vs Dogs.
    """

    def __init__(self, image_paths, labels, transform=None):

        super().__init__(root=".")

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert("RGB")

        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    



def prepare_dataset(split_data, transform=None):
    """
    Convert split dictionary into a PyTorch Dataset.
    """

    image_paths = []
    labels = []

    for class_name, images in split_data.items():

        label = 0 if class_name == "Cat" else 1

        for image in images:
            image_paths.append(image)
            labels.append(label)

    return CatsDogsDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform,
    )





# DataLoaders

def create_dataloader(
    dataset,
    shuffle=False,
):
    """
    Create PyTorch DataLoader.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

