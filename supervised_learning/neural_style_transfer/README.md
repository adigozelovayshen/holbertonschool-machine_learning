# Neural Style Transfer

## Overview
This directory contains implementations for **Neural Style Transfer (NST)** using TensorFlow 2.x. Neural Style Transfer is an optimization technique used to take three images—a **content image**, a **style reference image**, and an **input image** to optimize—and blend them together so the input image looks like the content image, but "painted" in the style of the style reference image.

## Tasks & Files

### 0. Initialize (`0-neural_style.py`)
Creates the `NST` class to initialize and preprocess images for style transfer:
* **Style Layers**: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, `block5_conv1`
* **Content Layer**: `block5_conv2`
* **Preprocessing**: Scales images using bicubic interpolation so the maximum dimension is 512 pixels and normalizes pixel values to `[0, 1]`.
