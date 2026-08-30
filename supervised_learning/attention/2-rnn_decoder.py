#!/usr/bin/env python3
"""Defines the RNNDecoder class for machine translation."""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNNDecoder layer.

        Args:
            vocab: integer representing size of output vocabulary
            embedding: integer representing vector dimensionality
            units: integer representing number of hidden units
            batch: integer representing batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Passes input tensor through the decoder.

        Args:
            x: tensor of shape (batch, 1) containing previous target word
            s_prev: tensor of shape (batch, units) containing previous hidden
            hidden_states: tensor of shape (batch, input_seq_len, units)

        Returns:
            y: tensor of shape (batch, vocab) containing output word vector
            s: tensor of shape (batch, units) containing new hidden state
        """
        # Calculate context vector and attention weights
        context, _ = self.attention(s_prev, hidden_states)

        # Convert input word indices to embeddings: (batch, 1, embedding)
        x = self.embedding(x)

        # Expand context vector shape: (batch, 1, units)
        context = tf.expand_dims(context, 1)

        # Concatenate context vector and input embedding along last axis
        # Result shape: (batch, 1, embedding + units)
        x = tf.concat([context, x], axis=-1)

        # Pass through GRU layer
        output, s = self.gru(x)

        # Reshape output to (batch, units) before fully connected layer
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass through Dense layer to get vocabulary prediction
        y = self.F(output)

        return y, s
