import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

env = gym.make("Blackjack-v1", natural=False, sab=False)

actions = [0, 1]

Q = defaultdict(lambda: {0: 0.0, 1: 0.0})
returns_sum = defaultdict(lambda: {0: 0.0, 1: 0.0})
returns_count = defaultdict(lambda: {0: 0, 1: 0})

episodes = 500_000
epsilon_start = 1.0
epsilon_end = 0.05
gamma = 1.0


def epsilon_greedy(state, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(Q[state], key=Q[state].get)


for ep in range(episodes):
    epsilon = max(
        epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (ep / episodes)
    )

    state, _ = env.reset()
    trajectory = []
    done = False

    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        done = terminated or truncated

    G = 0.0
    visited = set()
    for state, action, reward in reversed(trajectory):
        G = gamma * G + reward
        if (state, action) not in visited:
            visited.add((state, action))
            returns_sum[state][action] += G
            returns_count[state][action] += 1
            Q[state][action] = returns_sum[state][action] / returns_count[state][action]

    if (ep + 1) % 50_000 == 0:
        print(f"Episode {ep + 1}/{episodes}  epsilon={epsilon:.3f}")


def greedy_policy(state):
    return max(Q[state], key=Q[state].get)


eval_episodes = 20_000
wins = losses = draws = 0
total_reward = 0.0

for _ in range(eval_episodes):
    state, _ = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        action = greedy_policy(state)
        state, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        done = terminated or truncated
    total_reward += ep_reward
    if ep_reward > 0:
        wins += 1
    elif ep_reward < 0:
        losses += 1
    else:
        draws += 1

mean_reward = total_reward / eval_episodes
print(f"\nEval over {eval_episodes} episodes: mean_reward={mean_reward:.4f}")
print(
    f"Win: {wins / eval_episodes:.1%}  Loss: {losses / eval_episodes:.1%}  Draw: {draws / eval_episodes:.1%}"
)

env.close()


def basic_strategy_hard(player_sum, dealer_card):
    if player_sum >= 17:
        return 0
    if player_sum <= 11:
        return 1
    if player_sum == 12:
        return 1 if dealer_card in (2, 3, 7, 8, 9, 10, 1) else 0
    if 13 <= player_sum <= 16:
        return 0 if dealer_card in (2, 3, 4, 5, 6) else 1
    return 0


def basic_strategy_soft(player_sum, dealer_card):
    if player_sum >= 19:
        return 0
    if player_sum == 18:
        return 1 if dealer_card in (9, 10, 1) else 0
    return 1


mismatches = []
total_states = 0

for player_sum in range(4, 22):
    for dealer_card in range(1, 11):
        for usable_ace in (False, True):
            state = (player_sum, dealer_card, usable_ace)
            if returns_count[state][0] == 0 and returns_count[state][1] == 0:
                continue
            learned_action = greedy_policy(state)
            optimal_action = (
                basic_strategy_soft(player_sum, dealer_card)
                if usable_ace
                else basic_strategy_hard(player_sum, dealer_card)
            )
            total_states += 1
            if learned_action != optimal_action:
                mismatches.append(
                    (
                        player_sum,
                        dealer_card,
                        usable_ace,
                        learned_action,
                        optimal_action,
                    )
                )

print(f"\nStates visited: {total_states}")
print(
    f"Mismatches vs basic strategy: {len(mismatches)} ({len(mismatches) / total_states:.1%})"
)
for m in mismatches:
    learned_str = "HIT" if m[3] else "STICK"
    optimal_str = "HIT" if m[4] else "STICK"
    print(
        f"  player={m[0]:>2} dealer={m[1]:>2} soft_ace={m[2]!s:>5}  learned={learned_str:>5}  optimal={optimal_str:>5}"
    )
