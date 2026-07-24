#!/usr/bin/env python3
"""
Data Augmentation - Random Crop
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image

    Parameters:
    - image: a 3D tf.Tensor containing the image to crop
    - size: a tuple containing the size of the crop

    Returns:
    - The cropped image tensor
    """
    return tf.image.random_crop(image, size=size)
