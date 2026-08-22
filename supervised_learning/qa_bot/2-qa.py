#!/usr/bin/env python3
""" Interactive QA loop with BERT """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in an interactive loop

    Args:
        reference (str): The reference document containing answers
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        try:
            question = input("Q: ").strip()
            if question.lower() in exit_words:
                print("A: Goodbye")
                break

            answer = question_answer(question, reference)
            if answer is None or answer.strip() == "":
                print("A: Sorry, I do not understand your question.")
            else:
                print("A: {}".format(answer))
        except (KeyboardInterrupt, EOFError):
            print("\nA: Goodbye")
            break
