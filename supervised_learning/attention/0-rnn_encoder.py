#!/usr/bin/env python3
"""Defines the RNNEncoder class for machine translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNNEncoder layer.

        Args:
            vocab: integer representing size of input vocabulary
            embedding: integer representing vector dimensionality
            units: integer representing number of hidden units
            batch: integer representing batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """Initializes hidden states for RNN cell to zeros.

        Returns:
            Tensor of shape (batch, units) containing zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Passes input tensor through the encoder.

        Args:
            x: tensor of shape (batch, input_seq_len)
            initial: tensor of shape (batch, units)

        Returns:
            outputs: tensor of shape (batch, input_seq_len, units)
            hidden: tensor of shape (batch, units)
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden