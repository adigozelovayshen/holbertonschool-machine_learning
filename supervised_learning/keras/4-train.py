#!/usr/bin/env python3
"""
4-train module
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
    - network: the model to train
    - data: numpy.ndarray of shape (m, nx) containing input data
    - labels: one-hot numpy.ndarray of shape (m, classes) containing labels
    - batch_size: size of the batch used for mini-batch gradient descent
    - epochs: number of passes through data
    - verbose: boolean that determines if output should be printed
    - shuffle: boolean that determines whether to shuffle the batches

    Returns:
    - History object generated after training the model
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
