#!/usr/bin/env python3
"""Defines the Dataset class that loads and preps a translation dataset"""
from transformers import AutoTokenizer
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """Class constructor"""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset

        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple
                (pt, en)
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing the corresponding English
                sentence

        Returns:
            tokenizer_pt, tokenizer_en
        """
        base_tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        base_tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        def pt_sentences():
            """Generator yielding Portuguese sentences as strings"""
            for pt, _ in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def en_sentences():
            """Generator yielding English sentences as strings"""
            for _, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = base_tokenizer_pt.train_new_from_iterator(
            pt_sentences(), vocab_size=2 ** 13)
        tokenizer_en = base_tokenizer_en.train_new_from_iterator(
            en_sentences(), vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en
