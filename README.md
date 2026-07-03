# Reinforcement Learning Beginner Projects

A collection of beginner-friendly reinforcement learning projects implemented with Python, Gymnasium, and Stable-Baselines3.

## Projects

### Blackjack

Three approaches to learning optimal Blackjack play:

| File | Algorithm | Description |
|------|-----------|-------------|
| `blackjack.py` | PPO (Proximal Policy Optimization) | Trains a neural network policy using Stable-Baselines3 PPO on the `Blackjack-v1` environment |
| `blackjack2.py` | Q-Learning (tabular) | Custom epsilon-greedy Q-learning agent with epsilon decay |
| `blackjack_tabular.py` | Monte Carlo (first-visit MC control) | Tabular MC method with epsilon-greedy exploration; compares learned policy against basic strategy |

### CartPole

Baseline and trained agents for the `CartPole-v1` environment:

| File | Algorithm | Description |
|------|-----------|-------------|
| `random_actions.py` | Random policy | Simple baseline that takes random actions |
| `trained_agent.py` | PPO, DQN, A2C | Benchmarks three Stable-Baselines3 algorithms and compares their performance |

### GridWorld

| File | Algorithm | Description |
|------|-----------|-------------|
| `setup.py` | Q-Learning (tabular) | Agent navigates a 10×10 grid with obstacles to reach a goal using Q-learning |

## Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `stable-baselines3` (for PPO/DQN/A2C agents)
- `torch`
- `tqdm`

## Usage

Run any script directly:

```
python BlackJack/blackjack.py
python CartPole/random_actions.py
python GridWorld/setup.py
```