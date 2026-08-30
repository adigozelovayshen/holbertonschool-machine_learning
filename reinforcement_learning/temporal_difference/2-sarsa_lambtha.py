#!/usr/bin/env python3
"""
Module for SARSA(lambda) algorithm implementation
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(lambda) algorithm for value estimation.

    Parameters:
    - env: environment instance
    - Q: numpy.ndarray of shape (s, a) containing Q-table
    - lambtha: eligibility trace factor
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate
    - epsilon: initial threshold for epsilon greedy
    - min_epsilon: minimum value that epsilon should decay to
    - epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
    - Q: updated Q-table
    """
    initial_epsilon = epsilon
    n_states, n_actions = Q.shape

    def epsilon_greedy(state, current_epsilon):
        """Select action using epsilon-greedy strategy."""
        p = np.random.uniform(0, 1)
        if p < current_epsilon:
            return np.random.randint(0, n_actions)
        return np.argmax(Q[state])

    for ep in range(episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        action = epsilon_greedy(state, epsilon)
        E = np.zeros((n_states, n_actions))

        for step in range(max_steps):
            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, done, truncated, _ = step_res
            else:
                next_state, reward, done, _ = step_res
                truncated = False

            next_action = epsilon_greedy(next_state, epsilon)

            # TD Error calculation (line split to pass E501)
            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])

            E[state, action] += 1

            Q += alpha * delta * E
            E *= gamma * lambtha

            if done or truncated:
                break

            state = next_state
            action = next_action

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * ep
        )

    return Q
