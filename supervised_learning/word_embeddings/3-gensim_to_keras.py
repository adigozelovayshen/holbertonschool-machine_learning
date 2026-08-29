#!/usr/bin/env python3
"""
Gensim to Keras Embedding conversion module
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a trainable keras Embedding layer.

    Args:
        model: trained gensim word2vec model

    Returns:
        the trainable keras Embedding layer
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
