#!/usr/bin/env python3
"""Learning Rate Decay Upgraded"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates learning rate using inverse time decay"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
