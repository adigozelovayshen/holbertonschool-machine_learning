# Policy Gradients

This directory contains implementations of Policy Gradient algorithms in Reinforcement Learning using Python and NumPy.

## Files
- `policy_gradient.py`: Contains the `policy` function that computes action probabilities using softmax activation based on state inputs and weight matrices.

## Requirements
- `python3`
- `numpy`
- `gymnasium`

## Usage Example
```python
import numpy as np
policy = __import__('policy_gradient').policy

weight = np.random.rand(4, 2)
state = np.random.rand(1, 4)

probs = policy(state, weight)
print(probs)
