# Supervised Learning - Transfer Learning

## Description
This project focuses on applying **Transfer Learning** techniques using pre-trained convolutional neural network (CNN) architectures from `tensorflow.keras.applications` to classify images from the **CIFAR-10** dataset with a validation accuracy of **87% or higher**.

## Dataset
* **CIFAR-10**: Consists of 60,000 $32 \times 32$ color images in 10 classes (50,000 training images and 10,000 test images).

## Requirements
* Python 3.x
* TensorFlow 2.x
* Keras
* NumPy

## Files & Tasks

| File | Description |
| --- | --- |
| `0-transfer.py` | Python script that trains a DenseNet121 pre-trained model on CIFAR-10 dataset and saves the compiled model as `cifar10.h5`. Contains `preprocess_data(X, Y)` function. |
| `cifar10.h5` | Saved trained model achieving $\ge 87\%$ validation accuracy. |
| `0-main.py` | Main script used to load `cifar10.h5` and evaluate performance on CIFAR-10 validation set. |

## How to Run

1. Preprocess data and train the model:
2. Evaluate saved model accuracy:
## Author
* **Ayshen Adigozelova** - [adigozelovayshen](https://github.com/adigozelovayshen)
