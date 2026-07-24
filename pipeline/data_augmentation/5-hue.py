#!/usr/bin/env python3
"""
Data Augmentation - Adjust Hue
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    Parameters:
    - image: a 3D tf.Tensor containing the image to change
    - delta: float representing the amount the hue should change

    Returns:
    - The hue-adjusted image tensor
    """
    return tf.image.adjust_hue(image, delta=delta)
