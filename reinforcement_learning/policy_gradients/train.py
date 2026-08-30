#!/usr/bin/env python3
"""
Train module for Policy Gradient
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training with Policy Gradient.

    Parameters:
    - env: initial environment
    - nb_episodes: number of episodes used for training
    - alpha: the learning rate
    - gamma: the discount factor

    Returns:
    - scores: list of all score values for each episode
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    weights = np.random.rand(state_dim, action_dim)
    scores = []

    for episode in range(nb_episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        state = state.reshape(1, -1)

        episode_states = []
        episode_gradients = []
        episode_rewards = []

        done = False
        while not done:
            action, grad = policy_gradient(state, weights)
            step_res = env.step(action)

            if len(step_res) == 5:
                next_state, reward, done, truncated, _ = step_res
                done = done or truncated
            else:
                next_state, reward, done, _ = step_res

            episode_states.append(state)
            episode_gradients.append(grad)
            episode_rewards.append(reward)

            state = next_state.reshape(1, -1)

        score = sum(episode_rewards)
        scores.append(score)

        print("Episode: {} Score: {}".format(episode, score), end="\r",
              flush=True)

        # Policy Gradient çəki yenilənməsi (REINFORCE)
        for t in range(len(episode_rewards)):
            G = sum([r * (gamma ** i)
                     for i, r in enumerate(episode_rewards[t:])])
            weights += alpha * episode_gradients[t] * G

    return scores
