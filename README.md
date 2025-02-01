# CIFAR-10 Image Classification with CNN

This repository contains a Convolutional Neural Network (CNN) model implemented in TensorFlow and Keras for classifying images from the CIFAR-10 dataset. The model is trained to recognize 10 different classes of objects, such as airplanes, cars, birds, and more.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
The project demonstrates how to build, train, and evaluate a CNN model for image classification using the CIFAR-10 dataset. The model achieves competitive accuracy and includes visualization of training/validation metrics and predictions.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

## Model Architecture
The CNN model consists of the following layers:
1. **Convolutional Layer**: 32 filters, 3x3 kernel, ReLU activation.
2. **MaxPooling Layer**: 2x2 pool size.
3. **Convolutional Layer**: 64 filters, 3x3 kernel, ReLU activation.
4. **MaxPooling Layer**: 2x2 pool size.
5. **Flatten Layer**: Converts 2D outputs to 1D.
6. **Dense Layer**: 128 units, ReLU activation.
7. **Dropout Layer**: 50% dropout to prevent overfitting.
8. **Output Layer**: 10 units, softmax activation.

## Training
The model is trained for 10 epochs with a batch size of 64. 20% of the training data is used for validation. The Adam optimizer and categorical cross-entropy loss are used for training.

## Results
- **Test Accuracy**: Achieved accuracy on the test dataset.
- **Training/Validation Plots**: Visualizations of loss and accuracy over epochs.
- **Prediction Visualization**: Example predictions on random test images.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cifar10-cnn.git
   cd cifar10-cnn
