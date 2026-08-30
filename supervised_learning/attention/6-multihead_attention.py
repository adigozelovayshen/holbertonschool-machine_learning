#!/usr/bin/env python3
"""
Multi Head Attention Layer Implementation
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform Multi-Head Attention inheriting from
    tensorflow.keras.layers.Layer
    """

    def __init__(self, dm, h):
        """
        Class constructor

        Args:
            dm: integer representing the dimensionality of the model
            h: integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        Transpose the result such that the shape is
        (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Calculates multi-head attention

        Args:
            Q: tensor of shape (batch, seq_len_q, dk) with input for query
            K: tensor of shape (batch, seq_len_v, dk) with input for key
            V: tensor of shape (batch, seq_len_v, dv) with input for value
            mask: always None

        Returns:
            output: tensor of shape (..., seq_len_q, dm) containing scaled
                    dot product attention output
            weights: tensor of shape (..., h, seq_len_q, seq_len_v)
                     containing attention weights
        """
        batch_size = tf.shape(Q)[0]

        # 1. Project inputs through dense layers
        q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        k = self.Wk(K)  # (batch_size, seq_len_v, dm)
        v = self.Wv(V)  # (batch_size, seq_len_v, dm)

        # 2. Split heads to shape (batch_size, h, seq_len, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 3. Apply Scaled Dot-Product Attention
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        # 4. Transpose back to (batch_size, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 5. Concatenate heads back to (batch_size, seq_len_q, dm)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # 6. Apply final linear layer projection
        output = self.linear(concat_attention)

        return output, attention_weights
