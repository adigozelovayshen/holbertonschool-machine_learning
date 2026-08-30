#!/usr/bin/env python3
"""
Transformer Encoder Block Implementation
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class to create an encoder block for a transformer model
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        Args:
            dm: integer representing the dimensionality of the model
            h: integer representing the number of heads
            hidden: integer representing the number of hidden units in the
                    fully connected layer
            drop_rate: float representing the dropout rate
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Calculates the output of the transformer encoder block

        Args:
            x: tensor of shape (batch, input_seq_len, dm) containing the
               input to the encoder block
            training: boolean to determine if the model is training
            mask: mask to be applied for multi head attention

        Returns:
            tensor of shape (batch, input_seq_len, dm) containing the
            block's output
        """
        # 1. Multi-Head Attention Sub-layer with Residual Connection & Norm
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # 2. Feed Forward Sub-layer with Residual Connection & Norm
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
