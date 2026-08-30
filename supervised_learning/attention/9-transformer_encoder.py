#!/usr/bin/env python3
"""
Transformer Encoder Implementation
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class to create the encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layer
            input_vocab: size of the input vocabulary
            max_seq_len: maximum sequence length possible
            drop_rate: dropout rate
        """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Calculates the output of the transformer encoder

        Args:
            x: tensor of shape (batch, input_seq_len) containing input tokens
            training: boolean to determine if the model is training
            mask: mask to be applied for multi-head attention

        Returns:
            tensor of shape (batch, input_seq_len, dm) containing encoder
            output
        """
        seq_len = tf.shape(x)[1]

        # 1. Obtain embeddings and scale by sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 2. Add positional encodings (sliced to current sequence length)
        x += self.positional_encoding[:seq_len, :]

        # 3. Apply dropout
        x = self.dropout(x, training=training)

        # 4. Pass through all N encoder blocks
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
