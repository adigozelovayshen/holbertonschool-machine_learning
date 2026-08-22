#!/usr/bin/env python3
""" Semantic Search module using USE """
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents

    Args:
        corpus_path (str): path to the corpus of reference documents
        sentence (str): sentence to perform semantic search

    Returns:
        str: reference text of the document most similar to sentence
    """
    documents = []

    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    doc_embeddings = embed(documents)
    sentence_embedding = embed([sentence])

    sim_scores = np.inner(doc_embeddings, sentence_embedding)
    best_idx = np.argmax(sim_scores)

    return documents[best_idx]
