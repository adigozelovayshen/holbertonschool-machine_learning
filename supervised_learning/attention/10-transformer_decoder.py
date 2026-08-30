#!/usr/bin/env python3
"""
Transformer Decoder Implementation
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Class to create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layer
            target_vocab: size of the target vocabulary
            max_seq_len: maximum sequence length possible
            drop_rate: dropout rate
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Calculates the output of the transformer decoder

        Args:
            x: tensor of shape (batch, target_seq_len) containing input tokens
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                            containing the output of the encoder
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to the first MHA layer
            padding_mask: mask to be applied to the second MHA layer

        Returns:
            tensor of shape (batch, target_seq_len, dm) containing decoder
            output
        """
        seq_len = tf.shape(x)[1]

        # 1. Obtain target embeddings and scale by sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 2. Add positional encodings (sliced to target sequence length)
        x += self.positional_encoding[:seq_len, :]

        # 3. Apply dropout
        x = self.dropout(x, training=training)

        # 4. Pass through all N decoder blocks
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x
