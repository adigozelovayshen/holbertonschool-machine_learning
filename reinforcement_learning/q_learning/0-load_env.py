#!/usr/bin/env python3
"""
Module to load the FrozenLake environment from Gymnasium
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False,
                     render_mode=None):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Parameters:
    - desc: list of lists containing a custom map description, or None
    - map_name: string containing pre-made map name, or None
    - is_slippery: boolean indicating if ice is slippery
    - render_mode: string specifying the render mode (e.g., 'ansi')

    Returns:
    - env: the initialized gymnasium environment
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode
    )
    return env
