#!/usr/bin/env python3
"""
Module to perform Monte Carlo algorithm for value estimation
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.

    Parameters:
    - env: environment instance
    - V: numpy.ndarray of shape (s,) containing value estimate
    - policy: function taking a state and returning next action
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate

    Returns:
    - V: updated value estimate
    """
    for episode in range(episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, done, truncated, info = step_res
            else:
                next_state, reward, done, info = step_res
                truncated = False

            if done and reward == 0:
                reward = -1

            episode_data.append((state, reward))

            if done or truncated:
                break

            state = next_state

        episode_data = np.array(episode_data, dtype=object)
        G = 0

        for i in range(len(episode_data) - 1, -1, -1):
            s, r = episode_data[i]
            G = gamma * G + r

            if s not in episode_data[:i, 0]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
