#!/usr/bin/env python3
"""Calculates the scaled dot product attention using TensorFlow."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention.

    Args:
        Q: tensor with last two dimensions (..., seq_len_q, dk) (Query)
        K: tensor with last two dimensions (..., seq_len_v, dk) (Key)
        V: tensor with last two dimensions (..., seq_len_v, dv) (Value)
        mask: optional tensor broadcastable to (..., seq_len_q, seq_len_v)

    Returns:
        output: tensor of shape (..., seq_len_q, dv)
        weights: tensor of shape (..., seq_len_q, seq_len_v)
    """
    # Matrix multiplication of Q and transpose of K
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Get dimension dk (last dimension of K)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Scale the matrix multiplication by sqrt(dk)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax over the last axis (seq_len_v) to get attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply weights by V to get final output
    output = tf.matmul(weights, V)

    return output, weights
