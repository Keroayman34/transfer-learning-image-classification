# 🐱🐶 Transfer Learning Image Classification

Image classification project using **Transfer Learning** with a pre-trained **ResNet18** model in **PyTorch** to classify images as **Cats** or **Dogs**.

---

## 📌 Project Overview

This project demonstrates how to use a pre-trained Convolutional Neural Network (ResNet18) as a feature extractor for binary image classification.

The model is trained on the Microsoft Cats vs Dogs dataset using Transfer Learning, where only the final classification layer is trained while the convolutional backbone remains frozen.

---

## ✨ Features

* Transfer Learning using ResNet18
* Pre-trained ImageNet weights
* Custom PyTorch Dataset
* Data Augmentation
* Dataset Splitting (70% / 15% / 15%)
* Training & Validation Pipeline
* Model Evaluation
* Single Image Prediction
* Training History Visualization
* Modular Project Structure

---

## 🛠️ Tech Stack

* Python
* PyTorch
* TorchVision
* Pillow (PIL)
* Matplotlib
* Jupyter Notebook

---

## 📂 Project Structure

```
transfer-learning-image-classification/
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── transfer_learning_evaluation.ipynb
│
├── results/
│   └── plots/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── evaluate.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Dataset

Microsoft Cats vs Dogs Dataset

Classes:

* Cat
* Dog

Dataset Split:

| Split      | Ratio |
| ---------- | ----- |
| Train      | 70%   |
| Validation | 15%   |
| Test       | 15%   |

---

## 🧠 Model

* ResNet18
* Pre-trained on ImageNet
* Final Fully Connected layer replaced with 2 output neurons
* Feature extractor frozen during training

---

## 🔄 Data Augmentation

Training images are augmented using:

* Resize (224×224)
* Random Horizontal Flip
* Random Rotation
* Color Jitter
* Normalization (ImageNet statistics)

---

## 🚀 Training

Optimizer

* Adam

Loss Function

* CrossEntropyLoss

Epochs

* 5

---

## 📈 Evaluation

The project evaluates the trained model on the test dataset and reports:

* Test Loss
* Test Accuracy

It also supports prediction on custom images.

---

## 📉 Training History

Training history includes:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

Visualization is generated using Matplotlib.

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/Keroayman34/transfer-learning-image-classification.git
```

Move to the project directory

```bash
cd transfer-learning-image-classification
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

Open the notebook

```
notebooks/transfer_learning_evaluation.ipynb
```

or execute the Python modules inside the `src` directory.

---

## 📷 Predict a Custom Image

```python
prediction = predict_image(
    model=model,
    image_path="path/to/image.jpg",
    transform=eval_transform,
    device=device,
)

print(prediction)
```

---

## 📜 License

This project is created for educational purposes as part of an AI Transfer Learning assignment.
