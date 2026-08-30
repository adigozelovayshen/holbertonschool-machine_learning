#!/usr/bin/env python3
"""Defines the SelfAttention class for machine translation."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation (Bahdanau attention).
    """

    def __init__(self, units):
        """Initializes the SelfAttention layer.

        Args:
            units: integer representing the number of hidden units in the
                   alignment model.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates attention context vector and weights.

        Args:
            s_prev: tensor of shape (batch, units) containing previous decoder
                    hidden state.
            hidden_states: tensor of shape (batch, input_seq_len, units)
                           containing outputs of the encoder.

        Returns:
            context: tensor of shape (batch, units) containing context vector.
            weights: tensor of shape (batch, input_seq_len, 1) containing
                     attention weights.
        """
        # s_prev shape: (batch, units) -> expand dim to (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Calculate score score = V(tanh(W(s_prev) + U(hidden_states)))
        # W(s_prev_expanded) shape: (batch, 1, units)
        # U(hidden_states) shape: (batch, input_seq_len, units)
        score = self.V(tf.nn.tanh(
            self.W(s_prev_expanded) + self.U(hidden_states)
        ))

        # weights shape: (batch, input_seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)

        # context shape: (batch, units)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
