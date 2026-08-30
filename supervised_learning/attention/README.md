# Attention Mechanisms & Sequence-to-Sequence Models

This directory contains implementations of sequence-to-sequence (Seq2Seq) neural network components and attention mechanisms using TensorFlow/Keras for Machine Translation tasks.

## 📚 Project Overview

Sequence-to-sequence models map fixed-length inputs to fixed-length outputs where the lengths of input and output sequences may differ (e.g., text translation). The Attention mechanism allows the decoder to dynamically focus on different parts of the input sequence at each generation step rather than relying on a single fixed-length context vector.

## 📁 Files & Tasks

| File | Description |
| --- | --- |
| `0-rnn_encoder.py` | Implementation of an RNN Encoder layer using Keras `Embedding` and `GRU` for machine translation. |

---

## 🛠️ Requirements & Setup

- **Python Version:** `Python 3.9+`
- **Deep Learning Framework:** `TensorFlow 2.x`
- **Style Guide:** [`pycodestyle`](https://pycodestyle.pycqa.org/) (v2.5+)

### Installation

```bash
pip install tensorflow numpy