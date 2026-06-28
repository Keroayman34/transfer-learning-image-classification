
# Model Utilities

import torch
from tqdm import tqdm
from torchvision import models
from torch import nn


def build_model(num_classes=2):
    """
    Load a pre-trained ResNet18 model and replace
    the final classification layer.
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    return model




# Transfer Learning Utilities

def freeze_feature_extractor(model):
    """
    Freeze all convolutional layers except
    the final fully connected layer.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


# Training

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    epochs=5,
):
    """
    Train the model and monitor validation performance.
    """

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    for epoch in range(epochs):

        # Training
        
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        
        # Validation
        
        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in valid_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

        valid_loss = running_loss / total
        valid_acc = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_acc:.4f}"
        )

    return history