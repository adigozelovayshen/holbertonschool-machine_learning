#!/usr/bin/env python3
"""
Data Augmentation - Random Brightness
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    Parameters:
    - image: a 3D tf.Tensor containing the image to change
    - max_delta: float representing max amount to brighten/darken

    Returns:
    - The brightness-adjusted image tensor
    """
    return tf.image.random_brightness(image, max_delta=max_delta)
