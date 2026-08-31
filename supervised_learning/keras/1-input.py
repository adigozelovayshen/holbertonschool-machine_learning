#!/usr/bin/env python3
"""Build model module"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with Keras"""
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=L2
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    return K.Model(inputs=inputs, outputs=x)
