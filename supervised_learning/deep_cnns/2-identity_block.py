#!/usr/bin/env python3
"""
Identity Block Module for ResNet
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning (2015)

    Parameters:
    - A_prev: output from the previous layer
    - filters: tuple or list containing F11, F3, F12

    Returns:
    - The activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    # First component: 1x1 Conv -> BN -> ReLU
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Second component: 3x3 Conv -> BN -> ReLU
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=-1)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Third component: 1x1 Conv -> BN
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Shortcut Connection: Add A_prev
    add = K.layers.Add()([bn3, A_prev])

    # Final activation
    output = K.layers.Activation('relu')(add)

    return output
