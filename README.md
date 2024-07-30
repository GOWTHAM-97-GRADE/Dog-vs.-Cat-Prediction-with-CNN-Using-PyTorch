# Dog vs. Cat Prediction with CNN Using PyTorch

A PyTorch-based implementation of a Convolutional Neural Network (CNN) for predicting whether an image contains a dog or a cat. The project includes data preprocessing, CNN model definition, training, evaluation, and saving the trained model.

## Overview

This repository contains a PyTorch implementation of a CNN designed to classify images into two categories: dogs and cats. The project includes data preparation, model definition, training, and evaluation scripts.

### Key Functionalities

- **Data Preparation**: Processes images and labels into a format suitable for training and evaluation.
- **CNN Model**: Defines a Convolutional Neural Network for image classification.
- **Training Script**: Trains the CNN model and evaluates its performance on test and validation datasets.
- **Model Saving**: Saves the trained model for future use.

## Components

### Data Preparation

The `CustomDataset` class handles the loading and preprocessing of images. Images are resized, normalized, and transformed into tensors for training and evaluation.

### CNN Model

The `ConvNeuralNet` class defines a CNN with the following components:

- **Convolutional Layers**: Extract features from the input images.
- **Pooling Layers**: Reduce the spatial dimensions of feature maps.
- **Fully Connected Layers**: Classify the extracted features into the target classes (dog or cat).
- **Activation Functions**: Apply ReLU activation to introduce non-linearity.

### Training Script

The training script performs the following:

- **Data Loading**: Uses `DataLoader` to handle training, validation, and testing datasets.
- **Model Training**: Uses Cross-Entropy Loss and Stochastic Gradient Descent (SGD) for optimization.
- **Accuracy Evaluation**: Computes accuracy on test and validation datasets after each epoch.

### Model Saving

The trained model is saved to a file named `CatvsDoggo.pth` for future inference.

## Usage

### Define Hyperparameters

Set the hyperparameters for training the CNN:

```python
batch_size = 10
num_classes = 2
learning_rate = 0.001
num_epochs = 10
```

### Prepare the Data

Ensure your image data is organized in the `./train` directory, with filenames containing either "cat" or "dog". The dataset is split into training, testing, and validation sets.

### Initialize and Train the Model

Run the training script:

```python
# Model, criterion, and optimizer
model = ConvNeuralNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader: 
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    acc /= count
    print(f"Epoch {epoch+1}: Test accuracy {acc*100:.4f}%")

# Validation accuracy
model.eval()
val_acc = 0
val_count = 0
with torch.no_grad():
    for inputs, labels in validate_loader:
        y_pred = model(inputs)
        val_acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        val_count += len(labels)
val_acc /= val_count
print(f"Validation accuracy {val_acc*100:.2f}%")

# Save model
torch.save(model.state_dict(), "CatvsDoggo.pth")
```

## Running the Code

To run the training and evaluation, ensure you have the required dependencies installed and execute the script in your Python environment.

```bash
python your_script_name.py
```

## Acknowledgments

This implementation follows best practices for image classification using CNNs and PyTorch. Special thanks to the PyTorch documentation and tutorials that provided foundational knowledge.


## Dataset
- [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip)

