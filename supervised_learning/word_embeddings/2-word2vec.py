#!/usr/bin/env python3
"""
Word2Vec model module
"""
import os
import sys

# Ensure deterministic hashing BEFORE gensim/numpy are imported.
# PYTHONHASHSEED only takes effect at interpreter startup, so if it
# isn't already set to "0" we re-exec this same process with it set.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import gensim  # noqa: E402


def word2vec_model(
        sentences, vector_size=100, min_count=5, window=5,
        negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim word2vec model.

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: boolean to determine training type (True=CBOW, False=SG)
        epochs: number of iterations to train over
        seed: seed for random number generator
        workers: number of worker threads to train the model

    Returns:
        the trained Word2Vec model
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=epochs
    )

    return model
