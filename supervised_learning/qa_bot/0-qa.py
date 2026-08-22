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
    ```"""
    # Tokenizer-in yüklənməsi
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    
    # Modelin TensorFlow Hub-dan yüklənməsi
    model = hub.load('[https://tfhub.dev/see--/bert-uncased-tf2-qa/1](https://tfhub.dev/see--/bert-uncased-tf2-qa/1)')

    # Suallar və istinad mətni üçün tokenlərin yaradılması
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # BERT formatına uyğun input tokenlərin birləşdirilməsi: [CLS] + question + [SEP] + reference + [SEP]
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(reference_tokens) + 1)

    # Tensor formatına keçirilməsi
    input_ids = tf.constant([input_ids], dtype=tf.int32)
    input_mask = tf.constant([input_mask], dtype=tf.int32)
    segment_ids = tf.constant([segment_ids], dtype=tf.int32)

    # Modelin çağırılması
    outputs = model([input_ids, input_mask, segment_ids])
    
    # Cavabın başlanğıc və bitiş indekslərinin təyini
    start_index = tf.argmax(outputs[0][0][1:-1]) + 1
    end_index = tf.argmax(outputs[1][0][1:-1]) + 1

    # Əgər uyğun cavab tapılmadısa
    if start_index > end_index:
        return None

    # Tokenləri cavab mətni kimi birləşdirmək
    answer_tokens = tokens[start_index: end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer if answer.strip() else None
