#!/usr/bin/env python3
"""
Script to display a game played by the trained DQN agent on Atari Breakout
"""
import gymnasium as gym
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Permute
from tensorflow.keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


class AtariWrapper(gym.Wrapper):
    """
    Wrapper to adapt Gymnasium Atari environment to Keras-RL2 standards.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return self._process_frame(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._process_frame(obs), reward, done, info

    def _process_frame(self, frame):
        img = Image.fromarray(frame)
        img = img.convert('L').resize((84, 84))
        return np.array(img, dtype=np.uint8)


def create_model(window_length, num_actions):
    """
    Creates the CNN architecture for Deep Q-Learning
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model


if __name__ == '__main__':
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    env = AtariWrapper(env)

    num_actions = env.action_space.n
    window_length = 4

    model = create_model(window_length, num_actions)
    memory = SequentialMemory(limit=100000, window_length=window_length)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        memory=memory,
        policy=policy
    )

    dqn.compile(optimizer='adam')
    dqn.load_weights('policy.h5')

    # Agentin oyunu oynaması
    dqn.test(env, nb_episodes=10, visualize=True)
