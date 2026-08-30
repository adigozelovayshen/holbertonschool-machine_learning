#!/usr/bin/env python3
"""
Module to perform Monte Carlo value estimation
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.
    """
    for episode in range(episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        states = []
        rewards = []

        for step in range(max_steps):
            action = policy(state)
            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, done, truncated, info = step_res
            else:
                next_state, reward, done, info = step_res
                truncated = False

            states.append(state)
            rewards.append(reward)

            if done or truncated:
                break

            state = next_state

        G = 0
        for i in range(len(states) - 1, -1, -1):
            s = states[i]
            r = rewards[i]
            G = gamma * G + r

            if s not in states[:i]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
