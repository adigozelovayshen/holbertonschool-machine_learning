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

    Parameters:
    - env: FrozenLakeEnv instance
    - Q: numpy.ndarray containing the Q-table
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate
    - epsilon: initial threshold for epsilon greedy
    - min_epsilon: minimum value that epsilon should decay to
    - epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
    - Q: updated Q-table
    - total_rewards: list containing rewards per episode
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset() if isinstance(
            env.reset(), tuple) else (env.reset(), {})
        current_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, truncated, info = (
                env.step(action) if len(env.step(action)) == 5
                else (*env.step(action), False, {})
            )

            # Qura düşdükdə mükafatı -1 edirik
            if done and reward == 0:
                reward = -1

            # Q-table yenilənməsi
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            current_reward += reward

            if done or truncated:
                break

        # Epsilon eksponensial azalması (decay update)
        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon
        ) * np.exp(-epsilon_decay * episode)

        total_rewards.append(current_reward)

    return Q, total_rewards
