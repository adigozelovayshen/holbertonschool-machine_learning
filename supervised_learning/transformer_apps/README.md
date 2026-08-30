# Transformer Applications: Machine Translation

This directory contains implementations for building and training a Transformer model for Machine Translation (Portuguese to English) using TensorFlow 2.x and Hugging Face `transformers`.

## Project Structure

* **`0-dataset.py`**: Contains the `Dataset` class that loads the `ted_hrlr_translate/pt_to_en` dataset and trains sub-word tokenizers (`BertTokenizerFast`) for Portuguese and English with a maximum vocabulary size of $2^{13}$ (8192).
* **`setup.py`**: Utility script to load the local PT-to-EN dataset splits.

## Requirements

* Python 3.x
* TensorFlow 2.x
* TensorFlow Datasets (`tfds`)
* Hugging Face `transformers`

## Dataset Setup

Before running the scripts, download and extract the dataset into your local cache directory:

```bash
mkdir -p ~/.cache/ted_hrlr/
curl -L -O [https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/ted_hrlr_pt_to_en.tar.gz](https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/ted_hrlr_pt_to_en.tar.gz)
tar -xzf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr/