#!/usr/bin/env python3
"""
FastText model training module
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim fastText model.

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word
        negative: size of negative sampling
        window: maximum distance between current and predicted word
        cbow: boolean to determine training type (True for CBOW, False for SG)
        epochs: number of iterations to train over
        seed: seed for random number generator
        workers: number of worker threads to train the model

    Returns:
        the trained FastText model
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
