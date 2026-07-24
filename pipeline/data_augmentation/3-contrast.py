#!/usr/bin/env python3
"""
Data Augmentation - Adjust Contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image

    Parameters:
    - image: a 3D tf.Tensor containing the image to adjust
    - lower: float representing the lower bound of contrast factor range
    - upper: float representing the upper bound of contrast factor range

    Returns:
    - The contrast-adjusted image tensor
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
