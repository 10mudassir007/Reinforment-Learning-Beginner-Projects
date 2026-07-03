import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env():
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env = FlattenObservation(env)
    env = Monitor(env)
    return env


def main():
    n_envs = 8
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=1.0,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=200_000)

    eval_env = Monitor(FlattenObservation(gym.make("Blackjack-v1")))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=2000)
    print(
        f"\nEval over 2000 episodes: mean_reward={mean_reward:.3f} +/- {std_reward:.3f}"
    )

    wins = losses = draws = 0
    for _ in range(2000):
        done = False
        obs, _ = eval_env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    total = wins + losses + draws
    print(
        f"Win: {wins / total:.1%}  Loss: {losses / total:.1%}  Draw: {draws / total:.1%}"
    )

    model.save("./ppo_blackjack")
    eval_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
