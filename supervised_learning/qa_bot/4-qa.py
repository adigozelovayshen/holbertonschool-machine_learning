#!/usr/bin/env python3
""" Multi-reference Question Answering loop """
semantic_search = __import__('3-semantic_search').semantic_search
qa_function = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts in an interactive loop

    Args:
        corpus_path (str): path to the corpus of reference documents
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        try:
            question = input("Q: ").strip()
            if question.lower() in exit_words:
                print("A: Goodbye")
                break

            reference = semantic_search(corpus_path, question)
            answer = qa_function(question, reference)

            if answer is None or answer.strip() == "":
                print("A: Sorry, I do not understand your question.")
            else:
                print("A: {}".format(answer))
        except (KeyboardInterrupt, EOFError):
            print("\nA: Goodbye")
            break
