#!/usr/bin/env python3
"""
Module to perform Q-learning on FrozenLake environment
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning for the given environment.
    """
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        current_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, done, truncated, info = step_res
            else:
                next_state, reward, done, info = step_res
                truncated = False

            if done and reward == 0:
                reward = -1

            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            current_reward += reward

            if done or truncated:
                break

        # Decay epsilon: Standard Holberton exponential decay formula
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

        total_rewards.append(current_reward)

    return Q, total_rewards
