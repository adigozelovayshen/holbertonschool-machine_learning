#!/usr/bin/env python3
"""
Module to perform Monte Carlo value estimation
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.

    env: environment instance
    V: numpy.ndarray of shape (s,) containing the value estimate
    policy: function that takes a state and returns the next action
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns: V, the updated value estimate
    """
    for ep in range(episodes):
        # gymnasium returns (obs, info); older gym returns obs
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) \
            else reset_result

        episode = []  # list of (state, reward) pairs for this episode

        for step in range(max_steps):
            action = policy(state)
            step_result = env.step(action)

            # gymnasium: (obs, reward, terminated, truncated, info)
            # old gym:   (obs, reward, done, info)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            episode.append((state, reward))
            state = next_state

            if done:
                break

        # Compute first-visit returns, working backwards through episode
        states_seen = [s for s, _ in episode]
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            s, r = episode[t]
            G = gamma * G + r

            # first-visit check: only update if s doesn't appear earlier
            if s not in states_seen[:t]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
