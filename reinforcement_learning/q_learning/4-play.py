#!/usr/bin/env python3
"""
Module to play an episode using a trained Q-table
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Parameters:
    - env: FrozenLakeEnv instance
    - Q: numpy.ndarray containing the Q-table
    - max_steps: maximum number of steps in the episode

    Returns:
    - total_rewards: total reward earned
    - rendered_outputs: list of rendered board states
    """
    res = env.reset()
    state = res[0] if isinstance(res, tuple) else res

    rendered_outputs = [env.render()]
    total_rewards = 0.0

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
