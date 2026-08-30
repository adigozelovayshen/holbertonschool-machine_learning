# Deep Q-Learning - Atari Breakout

This directory contains Python scripts to train and play Atari's Breakout using Deep Q-Networks (DQN) with `keras`, `keras-rl2`, and `gymnasium`.

## Files
- `train.py`: Trains a DQN agent on Atari Breakout using `EpsGreedyQPolicy` and saves the trained model weights to `policy.h5`.
- `play.py`: Loads the trained model weights from `policy.h5` and displays an agent playing Breakout using `GreedyQPolicy`.

## Requirements
- `gymnasium[atari]`
- `keras-rl2`
- `tensorflow`
- `pillow`
- `numpy`

## Usage
To train the agent:
```bash
python3 train.py