#!/usr/bin/env python3
"""
Projection Block Module for ResNet
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning (2015)

    Parameters:
    - A_prev: output from the previous layer
    - filters: tuple or list containing F11, F3, F12
    - s: stride of the first convolution in main path and shortcut connection

    Returns:
    - The activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    # First component of main path: 1x1 Conv with stride s -> BN -> ReLU
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Second component of main path: 3x3 Conv with stride 1 -> BN -> ReLU
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=-1)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Third component of main path: 1x1 Conv with stride 1 -> BN
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Shortcut path: 1x1 Conv with stride s -> BN
    conv_sc = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn_sc = K.layers.BatchNormalization(axis=-1)(conv_sc)

    # Add main path and shortcut path
    add = K.layers.Add()([bn3, bn_sc])

    # Final activation
    output = K.layers.Activation('relu')(add)

    return output
