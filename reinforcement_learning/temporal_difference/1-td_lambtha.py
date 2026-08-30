#!/usr/bin/env python3
"""
Module to perform TD(lambda) algorithm for value estimation
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm on an environment.

    Parameters:
    - env: environment instance
    - V: numpy.ndarray of shape (s,) containing the value estimate
    - policy: function taking a state and returning next action
    - lambtha: eligibility trace factor
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
        E = np.zeros(n_states)

        for step in range(max_steps):
            action = policy(state)
            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, done, truncated, info = step_res
            else:
                next_state, reward, done, info = step_res
                truncated = False

            # Calculate TD Error delta
            delta = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for current state (accumulating traces)
            E[state] += 1

            # Update Value table and Eligibility traces for all states
            V += alpha * delta * E
            E *= gamma * lambtha

            if done or truncated:
                break

            state = next_state

    return V
