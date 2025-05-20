# Colon Cancer Image Classification 🧬🔬

This project implements a convolutional neural network using **Transfer Learning** with **VGG16** to classify medical images for **colon cancer detection**. It uses a dataset of histopathological images categorized into cancerous and non-cancerous tissue.

## 🧠 Project Overview

The goal is to train a deep learning model that can distinguish between healthy and cancerous colon tissues from high-resolution images. I leverage **data augmentation**, **VGG16 pretrained weights**, and **fine-tuning techniques** to improve generalization and performance.

---

## 📁 Dataset

- The dataset is structured in two directories:
  - `training_set/`: Contains images for training and validation (with subfolders for each class).
  - `test_set/`: Contains images for testing the model.
- Images are resized to **512x512** and normalized.
- Augmentation is applied during training to improve generalization.

> Dataset location:  
`D:/Users/PC/Documents/Universidad/colon11.zip`

---

## ⚙️ Model Architecture

The model is based on the **VGG16** convolutional base with the following additions:

- `GlobalAveragePooling2D`
- `Flatten`
- `Dense(256, activation='relu')`
- `Dropout(0.5)`
- `Dense(2, activation='softmax')`

All pretrained layers from VGG16 are **frozen** to retain the learned features from ImageNet.

---

## 🧪 Data Generators

```python
ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

🚀 Training
Optimizer: Adam

Loss: categorical_crossentropy

Epochs: 20

Batch size: 32

📊 Future Improvements
Add early stopping and model checkpointing.

Unfreeze top VGG16 layers for fine-tuning.

Add confusion matrix and classification report.

Export and deploy the model with Flask or FastAPI.


✍️ Author
Kevin Amaya 

📧 kevin.amayav@outlook.com

