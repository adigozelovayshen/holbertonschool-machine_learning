# QA Bot - Question Answering with BERT

This directory contains implementations for Question Answering (QA) tasks using pre-trained BERT models, TensorFlow, and the Hugging Face `transformers` library.

## Tasks

### Task 0: Question Answering
* **File:** `0-qa.py`
* **Description:** Implements a function `question_answer(question, reference)` that extracts an answer snippet from a reference text based on a given question.
* **Models Used:**
  * `bert-uncased-tf2-qa` from TensorFlow Hub
  * `bert-large-uncased-whole-word-masking-finetuned-squad` tokenizer from Hugging Face `transformers`
