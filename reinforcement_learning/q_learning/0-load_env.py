#!/usr/bin/env python3
"""
Module to load the FrozenLake environment from Gymnasium
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False,
                     render_mode="ansi"):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.
    """
    kwargs = {
        'desc': desc,
        'map_name': map_name,
        'is_slippery': is_slippery,
        'render_mode': render_mode
    }

    env = gym.make('FrozenLake-v1', **kwargs)
    return env
