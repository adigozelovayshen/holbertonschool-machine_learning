#!/usr/bin/env python3
"""
Dataset class for Machine Translation
"""
import tokenizers
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and pre-processes a dataset for machine translation
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        pt_pretrained = 'neuralmind/bert-base-portuguese-cased'
        en_pretrained = 'bert-base-uncased'

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            pt_pretrained,
            use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            en_pretrained,
            use_fast=True
        )

        def pt_generator():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_generator():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        trainer_pt = tokenizers.trainers.WordPieceTrainer(
            vocab_size=2**13,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        trainer_en = tokenizers.trainers.WordPieceTrainer(
            vocab_size=2**13,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )

        tokenizer_pt._tokenizer.train_from_iterator(
            pt_generator(),
            trainer=trainer_pt
        )
        tokenizer_en._tokenizer.train_from_iterator(
            en_generator(),
            trainer=trainer_en
        )

        return tokenizer_pt, tokenizer_en
