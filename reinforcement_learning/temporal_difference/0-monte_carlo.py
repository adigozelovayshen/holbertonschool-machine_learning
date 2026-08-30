#!/usr/bin/env python3
"""
Module to perform the Monte Carlo algorithm for value estimation
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm on an environment.

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
    n_states = V.shape[0]

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

            episode_data.append((state, reward))

            if done or truncated:
                break

            state = next_state

        G = 0
        episode_states = [x[0] for x in episode_data]

        for i in range(len(episode_data) - 1, -1, -1):
            s, r = episode_data[i]
            G = gamma * G + r

            # First-visit Monte Carlo update
            if s not in episode_states[:i]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
