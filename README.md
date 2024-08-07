# MNIST_ONNX_Classification

This repository contains a Python project for training a simple neural network on the MNIST dataset using PyTorch, exporting the model to ONNX format, and performing image classification using ONNX Runtime. The project includes steps to upload and utilize the ONNX model stored on Google Drive.

## Requirements

Make sure to install the following libraries before running the project:

```bash
pip install torch torchvision onnx onnxruntime requests matplotlib numpy
```

## Project Structure

- `train_and_save_model.py`: Script for training a simple neural network on the MNIST dataset and saving the model in ONNX format to Google Drive.
- `classify_images.py`: Script for loading the ONNX model from Google Drive and classifying images from the MNIST dataset.

## Usage

### 1. Training and Saving the Model

This script trains a simple neural network on the MNIST dataset and saves the trained model in ONNX format to your Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import onnx

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(5):  # Train for 5 epochs
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Convert the trained model to ONNX format and save to Google Drive
dummy_input = torch.randn(1, 1, 28, 28)  # Example input for the model
onnx_model_path = "/content/drive/My Drive/simple_model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'])
print(f"Model saved to {onnx_model_path}")
```

### 2. Classifying Images

This script loads the ONNX model from Google Drive and performs classification on images from the MNIST dataset.

```python
import requests
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Path to the ONNX model on Google Drive
onnx_model_path = "/content/drive/My Drive/simple_model.onnx"

# Load the ONNX model
ort_session = ort.InferenceSession(onnx_model_path)

# Prepare the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# Select a few images from the dataset
images, labels = [], []
for i, (img, lbl) in enumerate(test_loader):
    if i >= 5:
        break
    images.append(img)
    labels.append(lbl)

images = torch.cat(images)
labels = torch.cat(labels)

# Display images and perform classification
plt.figure(figsize=(10, 5))

for i in range(5):
    image = images[i].numpy().squeeze()
    label = labels[i].item()
    
    # Prepare the input for the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: images[i].numpy().reshape(1, 1, 28, 28).astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    pred_label = np.argmax(ort_outs[0])
    
    # Display the image
    plt.subplot(1, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}\nPred: {pred_label}')
    plt.axis('off')
    
    # Show result of classification
    if label == pred_label:
        plt.xlabel('Correct', color='green')
    else:
        plt.xlabel('Wrong', color='red')

plt.show()
```

## Functions

### `train_and_save_model.py`

- `SimpleModel`: Defines a simple fully connected neural network.
- `train_model`: Trains the model on the MNIST dataset.
- `save_model_to_onnx`: Converts the trained model to ONNX format and saves it to Google Drive.

### `classify_images.py`

- `load_model`: Loads the ONNX model from Google Drive.
- `classify_images`: Classifies images from the MNIST dataset using the ONNX model.
- `display_results`: Displays images along with classification results.

## File Storage

- The trained ONNX model is saved in your Google Drive at the path: `/content/drive/My Drive/simple_model.onnx`.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)

## Similar Projects

- [PyTorch to ONNX Conversion](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)

Feel free to clone this repository and use it for your own projects. Contributions and suggestions are welcome!
