#!/usr/bin/env python3
""" Question Answering with BERT """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question

    Args:
        question (str): The question to answer
        reference (str): The reference document containing the answer

    Returns:
        str: The answer snippet, or None if no answer is found
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] +
              reference_tokens + ['[SEP]'])
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = ([0] * (len(question_tokens) + 2) +
                   [1] * (len(reference_tokens) + 1))

    input_ids = tf.constant([input_ids], dtype=tf.int32)
    input_mask = tf.constant([input_mask], dtype=tf.int32)
    segment_ids = tf.constant([segment_ids], dtype=tf.int32)

    outputs = model([input_ids, input_mask, segment_ids])

    start_index = tf.argmax(outputs[0][0][1:-1]) + 1
    end_index = tf.argmax(outputs[1][0][1:-1]) + 1

    if start_index > end_index:
        return None

    answer_tokens = tokens[start_index: end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer if answer.strip() else None
