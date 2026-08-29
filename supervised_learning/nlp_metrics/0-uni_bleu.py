#!/usr/bin/env python3
"""
Calculates the unigram BLEU score for a sentence
"""
import nltk.translate.bleu_score as bleu


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    parameters:
        references [list of lists]: list of reference translations
        sentence [list]: list containing the model proposed sentence

    returns:
        the unigram BLEU score
    """
    return bleu.sentence_bleu(references, sentence, weights=(1, 0, 0, 0))
