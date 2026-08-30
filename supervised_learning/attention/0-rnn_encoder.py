#!/usr/bin/env python3
"""Defines the RNNEncoder class for machine translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNNEncoder layer.

        Args:
            vocab: integer representing the size of the input vocabulary
            embedding: integer representing the dimensionality of the embedding vector
            units: integer representing the number of hidden units in the RNN cell
            batch: integer representing the batch size
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
        """Initializes the hidden states for the RNN cell to a tensor of zeros.

        Returns:
            A tensor of shape (batch, units) containing initialized hidden states.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Passes input tensor through the encoder.

        Args:
            x: tensor of shape (batch, input_seq_len) containing word indices
            initial: tensor of shape (batch, units) containing initial hidden state

        Returns:
            outputs: tensor of shape (batch, input_seq_len, units) containing encoder outputs
            hidden: tensor of shape (batch, units) containing the last hidden state
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden