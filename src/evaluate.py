"""
evaluate.py

Evaluation utilities for trained models.
"""

import torch


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
):
    """
    Evaluate model on the test dataset.

    Returns:
        test_loss
        test_accuracy
    """

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

    test_loss = running_loss / total
    test_accuracy = correct / total

    return test_loss, test_accuracy






from PIL import Image


def predict_image(
    model,
    image_path,
    transform,
    device,
):
    """
    Predict class for a single image.
    """

    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        prediction = outputs.argmax(dim=1).item()

    classes = ["Cat", "Dog"]

    return classes[prediction]