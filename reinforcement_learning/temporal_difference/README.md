# Temporal Difference Learning

This directory contains implementations of Temporal Difference (TD) learning algorithms for Reinforcement Learning in Python.

## Files
- `0-monte_carlo.py`: Implements the First-Visit Monte Carlo algorithm to perform value estimation for a given policy in OpenAI Gymnasium environments.

## Requirements
- `python3`
- `numpy`
- `gymnasium`

## Usage Example
```python
import gymnasium as gym
import numpy as np
monte_carlo = __import__('0-monte_carlo').monte_carlo

env = gym.make('FrozenLake8x8-v1')
# Set initial values V and policy function, then run monte_carlo(env, V, policy)