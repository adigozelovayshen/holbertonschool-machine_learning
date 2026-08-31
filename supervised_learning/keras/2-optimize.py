#!/usr/bin/env python3
"""Optimize model module"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimizer for a Keras model"""
    network.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=alpha,
            beta_1=beta1,
            beta_2=beta2
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return None
