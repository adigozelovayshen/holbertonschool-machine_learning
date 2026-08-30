#!/usr/bin/env python3
"""
Module to play an episode using a trained Q-table
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.
    """
    env.unwrapped.render_mode = "ansi"
    reset_res = env.reset()
    state = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    rendered_outputs = [env.render()]
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state])

        step_res = env.step(action)
        if len(step_res) == 5:
            next_state, reward, done, truncated, info = step_res
        else:
            next_state, reward, done, info = step_res
            truncated = False

        rendered_outputs.append(env.render())
        total_rewards += reward
        state = next_state

        if done or truncated:
            break

    return total_rewards, rendered_outputs
